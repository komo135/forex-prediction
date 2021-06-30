import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
tnp = tf.experimental.numpy
from tensorflow.python.framework import ops

conv_args = {
    "filters": 32, "kernel_size": 1, "strides": 1, "padding": "same", "groups": 1, "kernel_initializer": "he_normal"
}  #
attention_args = {
    "num_heads": 4, "dim": 32, "dropout": 0.1, "use_bias": True, "sr_ratio": 1, "shape": None,
}
Conv1D = tf.keras.layers.Conv1D


def Inputs(input_shape, filters, kernel_size, strides, pooling):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides, "same", kernel_initializer="he_normal")(inputs)
    if pooling:
        x = tf.keras.layers.AvgPool1D(3, 2)(x)

    return inputs, x


def stack(x, function, activation, layer_args, erase_relu):
    if activation:
        x = tf.keras.layers.LayerNormalization()(x)
        if not erase_relu:
            x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    return function(**layer_args)(x)


def SE(inputs, filters, r=8):
    x = tf.keras.layers.GlobalAvgPool1D()(inputs)
    x = tf.keras.layers.Dense(filters // r, "elu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dense(filters, "sigmoid")(x)
    return tf.reshape(x, (-1, 1, filters))


class PositionAdd(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.pe = self.add_weight("pe", [input_shape[1], input_shape[2]],
                                  initializer=tf.keras.initializers.zeros())

    def call(self, inputs, **kwargs):
        return inputs + self.pe


class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads,
                 dim,
                 dropout=0.0,
                 use_bias=True,
                 sr_ratio=1,
                 shape=None,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._dim = dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._sr_ratio = int(sr_ratio)
        self._shape = shape
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = tf.keras.layers.Conv1D(dim, 1, 1, "causal", use_bias=use_bias)
        self.kv = tf.keras.layers.Conv1D(dim * 2, 1, 1, "causal", use_bias=use_bias)
        self.proj = tf.keras.layers.Conv1D(dim, 1, 1, "causal", use_bias=use_bias)
        # self.proj = tf.keras.layers.Conv1D(shape[-1], 1, 1, "causal", use_bias=use_bias)
        if dropout > 0:
            self.drop_out = tf.keras.layers.Dropout(dropout)
        if sr_ratio > 1:
            self.sr = tf.keras.layers.Conv1D(dim, sr_ratio, sr_ratio, "same", use_bias=use_bias)
            self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs: tf.Tensor, training=False, *args, **kwargs):
        B, N, C = inputs.shape
        q = self.q(inputs)
        q = tf.reshape(q, (-1, N, self._num_heads, C // self._num_heads))
        q = tf.transpose(q, [0, 2, 1, 3])

        if self._sr_ratio > 1:
            kv = self.sr(inputs)
            kv = self.norm(kv)
        else:
            kv = inputs

        kv = self.kv(kv)
        shape = kv.shape
        kv = tf.reshape(kv, (-1, shape[1], 2, self._num_heads, C // self._num_heads))
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])
        k, v = tnp.transpose(kv[0], (0, 1, 3, 2)), kv[1]

        attn = (q @ k) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.drop_out(attn, training=training)

        x = (attn @ v)
        x = tnp.reshape(tnp.transpose(x, (0, 2, 1, 3)), (-1, N, C))
        x = self.proj(x)

        return x

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "dim": self._dim,
            "dropout": self._dropout,
            "sr_ratio": self._sr_ratio,
            "shape": self._shape
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SAMModel(tf.keras.Model):
    rho = 0.05

    def train_step(self, data):
        x, y = data
        e_ws = []

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        if "_optimizer" in dir(self.optimizer): #mixed float policy
            grads_and_vars = self.optimizer._optimizer._compute_gradients(loss, var_list=self.trainable_variables, tape=tape)
        else:
            grads_and_vars = self.optimizer._compute_gradients(loss, var_list=self.trainable_variables, tape=tape)

        grads = [g for g, _ in grads_and_vars]

        grad_norm = self._grad_norm(grads)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(grads, self.trainable_variables):
            e_w = grad * scale
            e_ws.append(e_w)
            param.assign_add(e_w)

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        grads_and_vars = self.optimizer._compute_gradients(loss, var_list=self.trainable_variables, tape=tape)
        grads = [g for g, _ in grads_and_vars]

        for e_w, param in zip(e_ws, self.trainable_variables):
            param.assign_sub(e_w)

        grads_and_vars = list(zip(grads, self.trainable_variables))

        self.optimizer.apply_gradients(grads_and_vars)
        self.compiled_metrics.update_state(y, predictions)

        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm


def transformer_block(i, dim, dropout=0.1, sr=1, attention=Attention, head=None):
    x = tf.keras.layers.LayerNormalization()(i)
    # x = tf.keras.layers.MultiHeadAttention(4, dim, dim, dropout)(x, x, x)
    head = head if head is not None else 4
    x = attention(head, dim, dropout, True, 1, x.get_shape())(x)
    i = tf.keras.layers.Add()([x, i])
    x = tf.keras.layers.LayerNormalization()(i)

    x = tf.keras.layers.Conv1D(dim * 4, 1, 1, "causal", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(dim, 1, 1, "causal")(x)

    # x = tf.keras.layers.Dense(dim * 4, "relu", kernel_initializer="he_normal")(x)
    # # x = tf.keras.layers.Dense(dim, "gelu", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.Dropout(dropout)(x)
    # x = tf.keras.layers.Dense(dim)(x)
    x = tf.keras.layers.Add()([x, i])

    return x


def mix_block(inputs, filters, se, bot, groups, erase_relu):
    """
    densenetとresnetの利点を組み合わせ表現学習の特定の制限を回避する
    densenetとresnetは本質的な密なトポロジから派生している
    inner link = resnet, outer link = densenet
    """

    def block(inputs, filters, se, bot, groups):
        args = conv_args.copy()
        args.update({"filters": filters * 4, "kernel_size": 3})
        x = stack(inputs, Conv1D, True, args, erase_relu)
        if not bot:
            args.update({"filters": filters, "groups": groups})
            x = stack(x, Conv1D, True, args, False)
        else:
            args = attention_args.copy()
            args.update({"dim": filters, "shape": x.get_shape()})
            x = stack(x, Attention, True, args, False)
        if se:
            se = SE(x, filters)
            x = tf.keras.layers.Multiply()([x, se])
        return x

    r = block(inputs, filters, se, bot, groups)
    d = block(inputs, filters, se, bot, groups)

    x = tf.keras.layers.Add()([inputs[:, :, -filters:], r])
    x = tf.keras.layers.Concatenate()([inputs[:, :, :-filters], x, d])
    return x


def dense_block(inputs, filters, se, bot, groups, erase_relu):
    x = tf.keras.layers.LayerNormalization()(inputs)
    if not erase_relu:
        x = tf.keras.layers.ELU()(x)
    # x = tf.keras.layers.GaussianNoise(0.05)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv1D(filters * 4, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    # x = tf.keras.layers.GaussianNoise(0.1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    if not bot:
        x = tf.keras.layers.Conv1D(filters, 3, 1, "same", kernel_initializer="he_normal", groups=groups)(x)
        if se:
            se_ = SE(x, filters)
            x = tf.keras.layers.Multiply()([x, se_])
    else:
        x = tf.keras.layers.MultiHeadAttention(4, filters, dropout=0.1)(x, x)

    return tf.keras.layers.Concatenate()([inputs, x])


def Dense(input_shape, action_size=2, filters=32, num_layers=(6, 12, 32), se=False, dense_next=False, transformer=False,
          bot=False, mix_net=False, erase_relu=False, wide=False, sam=False, **kwargs):
    groups = 8 if dense_next else 1
    block = dense_block if not mix_net else mix_block
    model = SAMModel if sam else tf.keras.Model

    inputs, x = Inputs(input_shape, filters if not wide else filters // kwargs["k"], 7, 2, True)

    for i, l in enumerate(num_layers):
        last = i == (len(num_layers) - 1)
        bot_ = True if bot and last else False
        for _ in range(l):
            x = block(x, filters, se, False, groups, erase_relu)
        if bot_:
            channels = x.get_shape()[-1]
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Conv1D(channels // 2, 1, 1, "same", kernel_initializer="he_normal")(x)
            for _ in range(3):
                x = block(x, filters, se, True, groups, erase_relu)

        if not last:
            channels = x.get_shape()[-1]
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.AvgPool1D()(x)
            x = tf.keras.layers.Conv1D(channels // 2, 1, 1, "same", kernel_initializer="he_normal")(x)
        else:
            if transformer:
                vit_filters = 512
                num_vit_layers = 3
                x = tf.keras.layers.LayerNormalization()(x)
                x = tf.keras.layers.ELU()(x)
                x = tf.keras.layers.Conv1D(vit_filters, 1, 1, "same")(x)
                # x = Positio。nAdd()(x)

                for _ in range(num_vit_layers):
                    if "attention" not in kwargs.keys():
                        x = transformer_block(x, vit_filters)
                    else:
                        x = transformer_block(x, vit_filters, attention=kwargs["attention"])

            x = tf.keras.layers.GlobalAvgPool1D()(x)
            x = tf.keras.layers.Dense(action_size, "softmax")(x)

    return model(inputs, x)


def Transformer(input_shape, action_size=2, filters=32, num_layers=(3, 18, 3),
                attention=Attention, sr=False, sam=False):

    sr = [1 for _ in range(4)] if not sr else [8, 4, 2, 1]

    inputs, x = Inputs(input_shape, filters, 7, 2, True)
    x = tf.keras.layers.LayerNormalization()(x)

    kernel = [9, 7, 5, 3]

    for i in range(len(num_layers)):
        if i != 0:
            c = []
            for k in kernel:
                c.append(tf.keras.layers.Conv1D(filters // len(kernel), k, 2, "same")(x))
            x = tf.keras.layers.Concatenate()(c)
            x = tf.keras.layers.LayerNormalization()(x)
            filters = x.get_shape()[-1]
        x = PositionAdd()(x)

        for _ in range(num_layers[i]):
            x = transformer_block(x, filters, 0.1, sr[i], attention, head=16)

        filters *= 2

    x = tf.keras.layers.GlobalAvgPool1D()(x)
    x = tf.keras.layers.Dense(action_size, "softmax")(x)

    m = tf.keras.Model(inputs, x) if not sam else SAMModel(inputs, x)
    return m


def Pyconv(input_shape, action_size=2, filters=32, num_layers=(3, 6, 3), attention=SE, sam=False):
    def block(i, dim, k, groups):
        x = tf.keras.layers.LayerNormalization()(i)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Conv1D(dim * 4, 1, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        c = []
        for i_ in range(len(k)):
            c.append(
                tf.keras.layers.Conv1D(dim // len(k), k[i_], 1, "same", kernel_initializer="he_normal", groups=groups)(
                    x))
        x = tf.keras.layers.Concatenate()(c)
        # x = tf.keras.layers.LayerNormalization()(x)
        # x = tf.keras.layers.ELU()(x)

        # x = tf.keras.layers.Conv1D(dim, 1, 1, "same", kernel_initializer="he_normal")(x)

        att = attention(x, dim)
        x = tf.keras.layers.Multiply()([att, x])

        return tf.keras.layers.Concatenate()([x, i])

    wide = 1
    model = SAMModel if sam else tf.keras.Model

    inputs, x = Inputs(input_shape, filters, 7, 2, True)

    filters *= wide
    groups = 16
    kernel_size = [9, 7, 5, 3]

    for i in range(len(num_layers)):

        for i_ in range(num_layers[i]):
            x = block(x, filters, kernel_size, groups)

        if i != (len(num_layers) - 1):
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.AvgPool1D()(x)
            s = x.get_shape()[-1]
            x = tf.keras.layers.Conv1D(s // 2, 1, 1, "same", kernel_initializer="he_normal")(x)
        else:
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.GlobalAvgPool1D()(x)
            x = tf.keras.layers.Dense(action_size, "softmax")(x)

    return model(inputs, x)


dnl = (6, 12, 32)  # dense num layers
tnl = (3, 18, 3)  # transformer num layers
pnl = (3, 6, 3)  # pyconv
# wide net
depth = 22
n = (depth - 4) // 6
k = 5
wnf = 32 * k
wnl = tuple([n for _ in range(3)])
#

dense_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n)
dense_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, dense_next=True)

se_dense_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True)
se_dense_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True, dense_next=True)
se_wide_dense_next = lambda i, a, f=wnf, n=wnl: Dense(i, a, f, n, se=True, dense_next=True, wide=True, k=k)
se_erase_dense_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True, dense_next=True, erase_relu=True)
se_vit_dense_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True, dense_next=True, transformer=True)
se_bot_dense_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True, bot=True)
se_bot_dense_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True, dense_next=True, bot=True)
se_mix_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True, dense_next=True, mix_net=True)
se_bot_mix_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True, dense_next=True, mix_net=True, bot=True)

pvt = lambda i, a, f=64, n=tnl: Transformer(i, a, f, tnl, Attention, sr=True)  # Pyramid Vision Transformer
sam_pvt = lambda i, a, f=64, n=tnl: Transformer(i, a, f, tnl, Attention, sr=True, sam=True)
wide_pvt = lambda i, a, f=64*5, n=(3, 3, 3): Transformer(i, a, f, tnl, Attention, sr=True)

se_pyconv = lambda i, a, f=128, n=pnl: Pyconv(i, a, f, n, attention=SE)

"""
bottle neck = (1, 3)
se_dense_net = 
    val_loss: 0.1949 - val_accuracy: 0.9285
se_dense_next =
    val_loss: 0.1886 - val_accuracy: 0.9306
se_bot_dense_net =
    val_loss: 0.2058 - val_accuracy: 0.9221
se_bot_dense_next
    val_loss: 0.1888 - val_accuracy: 0.9336
se_vit_dense_next =
    val_loss: 0.5954 - val_accuracy: 0.6908
se_pyconv =
    val_loss: 0.1856 - val_accuracy: 0.9321
se_mix_next =
    val_loss: 0.1892 - val_accuracy: 0.9374
se_bot_mix_next =
    val_loss: 0.1758 - val_accuracy: 0.9346
    
bottle neck = (3, 3)
se_dense_next =
    val_loss: 0.1692 - val_accuracy: 0.9373
se_wide_dense_next =
    val_loss: 0.1762 - val_accuracy: 0.9345
    
transformer
pvt =
    val_loss: 0.5780 - val_accuracy: 0.7053  
"""


def build_model(model_name: str, input_shape: tuple, action_size: int,
                filters=None, num_layers=None) -> tf.keras.Model:
    """
    利用可能な model_name :
        dense_net, dense_next, se_dense_net, se_dense_next, se_vit_dense_next, se_bot_dense_next
    """
    args = (input_shape, action_size)
    if filters is not None:
        args += (filters,)
    if num_layers is not None:
        args += (num_layers,)

    return eval(model_name)(*args)
