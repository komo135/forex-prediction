import tensorflow as tf
try:
    from einops.layers.tensorflow import Rearrange
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'einops'])

    from einops.layers.tensorflow import Rearrange

tnp = tf.experimental.numpy

conv_args = {
    "filters": 32, "kernel_size": 1, "strides": 1, "padding": "same", "groups": 1, "kernel_initializer": "he_normal"
}  #
attention_args = {
    "num_heads": 4, "dim": 32, "dropout": 0.1, "use_bias": True, "sr_ratio": 1,
}
Conv1D = tf.keras.layers.Conv1D
l2 = tf.keras.regularizers.l2(1e-4)


def Inputs(input_shape, filters, kernel_size, strides, pooling):
    inputs = tf.keras.layers.Input(input_shape)
    # x = tf.keras.layers.GaussianNoise(0.01)(inputs)
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides, "same", kernel_initializer="he_normal")(inputs)
    if pooling:
        x = tf.keras.layers.AvgPool1D(3, 2)(x)

    return inputs, x


def stack(x, function, activation, layer_args, erase_relu):
    if activation:
        x = tf.keras.layers.LayerNormalization()(x)
        if not erase_relu:
            x = tf.keras.layers.Activation("elu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    return function(**layer_args)(x)


def SE(inputs, filters, r=8):
    x = tf.keras.layers.GlobalAvgPool1D()(inputs)
    x = tf.keras.layers.Dense(filters // r, "elu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dense(filters, "sigmoid")(x)
    return tf.reshape(x, (-1, 1, filters))


class CBAM(tf.keras.layers.Layer):
    def __init__(self, filters, r=8):
        super(CBAM, self).__init__()
        self.filters = filters
        self.r = r

        self.avg_pool = tf.keras.layers.GlobalAvgPool1D()
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()
        self.mlp = [
            tf.keras.layers.Dense(filters // r, "elu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(filters)
        ]

        self.concat = tf.keras.layers.Concatenate()
        self.conv = tf.keras.layers.Conv1D(1, 7, 1, "same", activation="sigmoid")

    def compute_mlp(self, x, pool):
        x = pool(x)
        for mlp in self.mlp:
            x = mlp(x)

        return x

    def call(self, inputs, training=None, *args, **kwargs):
        x = self.compute_mlp(inputs, self.avg_pool) + self.compute_mlp(inputs, self.max_pool)
        x = inputs * tf.reshape(tf.nn.sigmoid(x), (-1, 1, self.filters))

        conv = self.concat([tf.reduce_mean(x, -1, keepdims=True), tf.reduce_max(x, -1, keepdims=True)])
        return x * self.conv(conv)

    def get_config(self):
        new_config = {"filters":self.filters,
                  "r": self.r}
        config = super(CBAM, self).get_config()
        config.update(new_config)
        return config


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
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._dim = dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._sr_ratio = int(sr_ratio)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = tf.keras.layers.Conv1D(dim, 1, 1, "same", use_bias=use_bias)
        self.kv = tf.keras.layers.Conv1D(dim * 2, 1, 1, "same", use_bias=use_bias)
        self.proj = tf.keras.layers.Conv1D(dim, 1, 1, "same", use_bias=use_bias)
        # self.proj = tf.keras.layers.Conv1D(shape[-1], 1, 1, "causal", use_bias=use_bias)
        if dropout > 0:
            self.drop_out = tf.keras.layers.Dropout(dropout)
        if sr_ratio > 1:
            self.sr = tf.keras.layers.Conv1D(dim, sr_ratio, sr_ratio, "same", use_bias=use_bias)
            self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs: tf.Tensor, training=False, *args, **kwargs):
        q = self.q(inputs)
        q = Rearrange("b n (h k) -> b h n k", h=self._num_heads)(q)

        if self._sr_ratio > 1:
            kv = self.sr(inputs)
            kv = self.norm(kv)
        else:
            kv = inputs

        kv = self.kv(kv)
        k, v = kv[:,:,:self._dim], kv[:,:,self._dim:]
        k = Rearrange("b n (h k) -> b h k n", h=self._num_heads)(k)
        v = Rearrange("b n (h k) -> b h n k", h=self._num_heads)(v)

        attn = (q @ k) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.drop_out(attn, training=training)

        x = (attn @ v)
        x = Rearrange("b k n h -> b n (k h)")(x)
        x = self.proj(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dim

    def get_config(self):
        new_config = {
            "num_heads": self._num_heads,
            "dim": self._dim,
            "dropout": self._dropout,
            "sr_ratio": self._sr_ratio,
            "shape": self._shape
        }
        config = super(Attention, self).get_config()
        config.update(new_config)
        return config


class LambdaLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, heads=4, use_bias=False, u=4, kernel_size=7):
        super(LambdaLayer, self).__init__()

        self.out_dim = out_dim
        self.k = out_dim // 1
        self.heads = heads
        self.v = out_dim // heads
        self.u = u
        self.kernel_size = kernel_size
        self.use_bias = use_bias

        self.top_q = tf.keras.layers.Conv1D(k * heads, 1, 1, "same", use_bias=use_bias)
        self.top_k = tf.keras.layers.Conv1D(k * u, 1, 1, "same", use_bias=use_bias)
        self.top_v = tf.keras.layers.Conv1D(self.v * self.u, 1, 1, "same", use_bias=use_bias)

        self.norm_q = tf.keras.layers.LayerNormalization()
        self.norm_k = tf.keras.layers.LayerNormalization()

        self.pos_conv = tf.keras.layers.Conv2D(k, (1, self.kernel_size), padding="same")

    def call(self, inputs, *args, **kwargs):
        u, h = self.u, self.heads

        q = self.top_q(inputs)
        k = self.top_k(inputs)
        v = self.top_v(inputs)

        q = Rearrange("b n (h k) -> b h k n", h=h)(q)
        k = Rearrange("b n (u k) -> b u k n", u=u)(k)
        v = Rearrange("b n (u v) -> b u v n", u=u)(v)

        k = tf.nn.softmax(k)

        lc = tf.einsum("b u k n, b u v n -> b k v", k, v)
        yc = tf.einsum("b h k n, b k v -> b n h v", q, lc)

        v = Rearrange("b u v n -> b v n u")(v)
        lp = self.pos_conv(v)
        lp = Rearrange("b v n k -> b v k n")(lp)
        yp = tf.einsum("b h k n, b v k n -> b n h v", q, lp)

        y = yc + yp
        output = Rearrange("b n h v -> b n (h v)")(y)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dim

    def get_config(self):
        new_config = {
            "out_dim": self.out_dim,
            "head": self.heads,
            "use_bias": self.use_bias,
            "u": self.u,
            "kernel_size": self.kernel_size
        }
        config = super(LambdaLayer, self).get_config()
        config.update(new_config)

        return config


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
    x = attention(head, dim, dropout, True, 1)(x)
    i = tf.keras.layers.Add()([x, i])
    x = tf.keras.layers.LayerNormalization()(i)

    x = tf.keras.layers.Conv1D(dim * 4, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(dim, 1, 1, "same")(x)

    # x = tf.keras.layers.Dense(dim * 4, "relu", kernel_initializer="he_normal")(x)
    # # x = tf.keras.layers.Dense(dim, "gelu", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.Dropout(dropout)(x)
    # x = tf.keras.layers.Dense(dim)(x)
    x = tf.keras.layers.Add()([x, i])

    return x


def mix_block(inputs, filters, se, bot, groups, erase_relu, cbam, lambda_net):
    """
    densenetとresnetの利点を組み合わせ表現学習の特定の制限を回避する
    densenetとresnetは本質的な密なトポロジから派生している
    inner link = resnet, outer link = densenet
    """

    def block(inputs, filters, se, bot, groups):
        args = conv_args.copy()
        args.update({"filters": filters * 4, "kernel_size": 3})
        x = stack(inputs, Conv1D, True, args, erase_relu)
        if bot:
            args = attention_args.copy()
            args.update({"dim": filters})
            x = stack(x, Attention, True, args, False)
        elif lambda_net:
            args = {"out_dim": filters}
            x = stack(x, LambdaLayer, True, args, False)
        else:
            args.update({"filters": filters, "groups": groups})
            x = stack(x, Conv1D, True, args, False)
        if se:
            se = SE(x, filters)
            x = tf.keras.layers.Multiply()([x, se])
        elif cbam:
            x = CBAM(filters)(x)
        return x

    r = block(inputs, filters, se, bot, groups)
    d = block(inputs, filters, se, bot, groups)

    x = tf.keras.layers.Add()([inputs[:, :, -filters:], r])
    x = tf.keras.layers.Concatenate()([inputs[:, :, :-filters], x, d])
    return x


def dense_block(inputs, filters, se, bot, groups, erase_relu, cbam, lambda_net):
    x = tf.keras.layers.LayerNormalization()(inputs)
    if not erase_relu:
        x = tf.keras.layers.Activation("elu")(x)
    # x = tf.keras.layers.GaussianNoise(0.05)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv1D(filters * 4, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)
    # x = tf.keras.layers.GaussianNoise(0.1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    if bot:
        x = Attention(4, filters, 0.1, False, 1)(x)
    elif lambda_net:
        x = LambdaLayer(out_dim=filters)(x)
    else:
        x = tf.keras.layers.Conv1D(filters, 3, 1, "same", kernel_initializer="he_normal", groups=groups)(x)
        if se:
            se_ = SE(x, filters)
            x = tf.keras.layers.Multiply()([x, se_])
        elif cbam:
            x = CBAM(filters)(x)

    return tf.keras.layers.Concatenate()([inputs, x])


def Dense(input_shape, action_size=2, filters=32, num_layers=(6, 12, 32), se=False, dense_next=False, transformer=False,
          bot=False, mix_net=False, erase_relu=False, wide=False, sam=False, cbam=False, lambda_net=False, **kwargs):
    groups = 16 if dense_next else 1
    block = dense_block if not mix_net else mix_block
    model = SAMModel if sam else tf.keras.Model

    inputs, x = Inputs(input_shape, filters if not wide else filters // kwargs["k"], 7, 2, True)

    for i, l in enumerate(num_layers):
        last = i == (len(num_layers) - 1)
        bot_ = True if bot and last else False
        for _ in range(l):
            x = block(x, filters, se, False, groups, erase_relu, cbam, lambda_net)
        if bot_:
            channels = x.get_shape()[-1]
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Activation("elu")(x)
            x = tf.keras.layers.Conv1D(channels // 2, 1, 1, "same", kernel_initializer="he_normal")(x)
            for _ in range(3):
                x = block(x, filters, se, True, groups, erase_relu, cbam, lambda_net)

        if not last:
            channels = x.get_shape()[-1]
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Activation("elu")(x)
            x = tf.keras.layers.AvgPool1D()(x)
            x = tf.keras.layers.Conv1D(channels // 2, 1, 1, "same", kernel_initializer="he_normal")(x)
        else:
            if transformer:
                vit_filters = 512
                num_vit_layers = 3
                x = tf.keras.layers.LayerNormalization()(x)
                x = tf.keras.layers.Activation("elu")(x)
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


def Pyconv(input_shape, action_size=2, filters=32, num_layers=(3, 6, 3), attention_ = False, attention=SE, sam=False,
           erase=False, bot=False):
    def block(i, dim, k, groups, attention_, attention, erase, bot):
        x = tf.keras.layers.LayerNormalization()(i)
        if not erase:
            x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Conv1D(dim * 4, 3, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        if not bot:
            c = []
            for i_ in range(len(k)):
                c.append(
                    tf.keras.layers.Conv1D(dim // len(k), k[i_], 1, "same", kernel_initializer="he_normal", groups=groups)(
                        x))
            x = tf.keras.layers.Concatenate()(c)
        else:
            # x = Attention(4, dim, dropout=0.1)(x)
            x = tf.keras.layers.MultiHeadAttention(4, dim, dropout=0.1)(x, x)
        # x = tf.keras.layers.LayerNormalization()(x)
        # x = tf.keras.layers.ELU()(x)

        # x = tf.keras.layers.Conv1D(dim, 1, 1, "same", kernel_initializer="he_normal")(x)
        if attention_:
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
        bot_ = True if bot and (i == (len(num_layers) - 1)) else False

        for i_ in range(num_layers[i]):
            x = block(x, filters, kernel_size, groups, attention_, attention, erase, bot_)
        if bot_:
            for _ in range(3):
                x = block(x, filters, kernel_size, groups, attention_, attention, erase, bot_)

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
bot_dense_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, bot=True)
sam_dense_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, sam=True)
se_dense_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True)
erase_dense_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, erase_relu=True)
cbam_dense_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, cbam=True)

dense_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, dense_next=True)
se_dense_next = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, se=True, dense_next=True)

mix_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, mix_net=True)
sam_erase_mix_net = lambda i, a, f=32, n=dnl: Dense(i, a, f, n, mix_net=True, sam=True, erase_relu=True)

lambda_net = lambda i, a, f=128, n=pnl: Dense(i, a, f, n, lambda_net=True)
bot_lambda_net = lambda i, a, f=128, n=pnl: Dense(i, a, f, n, lambda_net=True, bot=True)
sam_erase_lambda_net = lambda i, a, f=128, n=pnl: Dense(i, a, f, n, lambda_net=True, sam=True, erase_relu=True)

pvt = lambda i, a, f=64, n=tnl: Transformer(i, a, f, tnl, Attention, sr=True)  # Pyramid Vision Transformer
sam_pvt = lambda i, a, f=64, n=tnl: Transformer(i, a, f, tnl, Attention, sr=True, sam=True)
wide_pvt = lambda i, a, f=64*5, n=(3, 3, 3): Transformer(i, a, f, tnl, Attention, sr=True)

pyconv = lambda i, a, f=128, n=pnl: Pyconv(i, a, f, n)
sam_bot_erase_pyconv = lambda i, a, f=128, n=pnl: Pyconv(i, a, f, n, erase=True, bot=True, sam=True)

"""
dense net =
    val_loss: 0.5946 - val_accuracy: 0.6925
bot dense net =
    val_loss: 0.5904 - val_accuracy: 0.6941
se dense net =
    val_loss: 0.5995 - val_accuracy: 0.6879
sam dense net =
    val_loss: 0.5915 - val_accuracy: 0.6943
erase dense net =
    val_loss: 0.5932 - val_accuracy: 0.6923
cbam dense net =
    val_loss: 0.6019 - val_accuracy: 0.6869

dense next =
    val_loss: 0.5957 - val_accuracy: 0.6900
se dense next
    val_loss: 0.5979 - val_accuracy: 0.6887

pyconv =
    val_loss: 0.5947 - val_accuracy: 0.6914
sam_bot_erase_pyconv =
    val_loss: 0.5891 - val_accuracy: 0.6955
    
mix net =
    val_loss: 0.5909 - val_accuracy: 0.6951
sam_erase_mix_net =
    val_loss: 0.5864 - val_accuracy: 0.6981
    
lambda net =
    val_loss: 0.5921 - val_accuracy: 0.6916
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
