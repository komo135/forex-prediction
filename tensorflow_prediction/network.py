import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Layer, Concatenate, MultiHeadAttention, LayerNormalization, ELU, Add, Dense

try:
    from einops.layers.tensorflow import Rearrange
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'einops'])

    from einops.layers.tensorflow import Rearrange

l2 = tf.keras.regularizers.l2(1e-4)


def Inputs(input_shape, filters, kernel_size, strides, pooling):
    inputs = tf.keras.layers.Input(input_shape)
    # x = tf.keras.layers.GaussianNoise(0.01)(inputs)
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides, "same", kernel_initializer="he_normal")(inputs)
    if pooling:
        x = tf.keras.layers.AvgPool1D(3, 2)(x)

    return inputs, x


class Pyconv(Layer):
    def __init__(self, dim, groups=16):
        super(Pyconv, self).__init__()

        self.dim = dim
        self.groups = groups

        self.k = k = [3, 5, 7, 9]
        self.conv = [
            Conv1D(dim // 4, k, 1, "same", kernel_initializer="he_normal", groups=groups) for k in k
        ]
        self.concat = Concatenate()

    def call(self, inputs, *args, **kwargs):
        x = []
        for conv in self.conv:
            x.append(conv(inputs))

        return self.concat(x)

    def get_config(self):
        new_config = {
            "dim": self.dim,
            "groups": self.groups
        }
        config = {}
        # config = super(Pyconv, self).get_config()
        config.update(new_config)

        return config


class SE(Layer):
    def __init__(self, dim, r=8):
        super(SE, self).__init__()
        self.dim = dim
        self.r = r

        self.mlp = tf.keras.Sequential([
            Dense(dim // r, "elu", kernel_initializer="he_normal"),
            Dense(dim, "sigmoid")
        ])

    def call(self, inputs, *args, **kwargs):
        x = tf.reduce_mean(inputs, axis=1, keepdims=True)
        x = self.mlp(x)
        x *= inputs

        return x

    def get_config(self):
        config = {
            "dim": self.dim,
            "r": self.r
        }

        return config


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
        new_config = {"filters": self.filters,
                      "r": self.r}
        config = {}
        # config = super(CBAM, self).get_config()
        config.update(new_config)
        return config


class PositionAdd(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.pe = self.add_weight("pe", [input_shape[1], input_shape[2]],
                                  initializer=tf.keras.initializers.zeros())

    def call(self, inputs, **kwargs):
        return inputs + self.pe


class DepthWiseConv1D(Layer):
    def __init__(self, head, dim_in, dim_out, strides, use_bias):
        self.head = head
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.strides = strides
        self.use_bias = use_bias

        super(DepthWiseConv1D, self).__init__()
        self.conv = tf.keras.Sequential([
            Conv1D(dim_in, 3, strides, "same", use_bias=use_bias, groups=dim_in),
            LayerNormalization(),
            Conv1D(dim_out * head, 1, 1, "same", use_bias=use_bias)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs)

    def get_config(self):
        config = {}
        # config = super(DepthWiseConv1D, self).get_config()
        config.update({
            "head": self.head,
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "strides": self.strides,
            "use_bias": self.use_bias
        })

        return config


class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads,
                 dim,
                 dropout=0.,
                 use_bias=True,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._dim = dim
        self.dim = dim * num_heads
        self._dropout = dropout
        self._use_bias = use_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.q = tf.keras.layers.SeparableConv1D(dim* )
        self.q = DepthWiseConv1D(num_heads, dim, dim, 1, use_bias)
        self.kv = DepthWiseConv1D(num_heads, dim, dim * 2, 2, use_bias)
        self.proj = tf.keras.layers.Conv1D(dim, 1, 1, "same", use_bias=use_bias)
        if dropout > 0:
            self.drop_out = tf.keras.layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, training=False, *args, **kwargs):
        q = self.q(inputs)
        kv = self.kv(inputs)
        k, v = kv[:, :, :self.dim], kv[:, :, self.dim:]

        q = Rearrange("b n (h k) -> b h n k", h=self._num_heads)(q)
        k = Rearrange("b n (h k) -> b h k n", h=self._num_heads)(k)
        v = Rearrange("b n (h k) -> b h n k", h=self._num_heads)(v)

        attn = (q @ k) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        if self._dropout > 0:
            attn = self.drop_out(attn, training=training)

        x = (attn @ v)
        x = Rearrange("b k n h -> b n (k h)")(x)
        x = self.proj(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dim

    def get_config(self):
        config = {}
        # config = super(Attention, self).get_config()
        new_config = {
            "num_heads": self._num_heads,
            "dim": self._dim,
            "dropout": self._dropout,
            "use_bias": self._use_bias
        }
        config.update(new_config)
        return config


class TransformerMlp(Layer):
    def __init__(self, dim):
        super(TransformerMlp, self).__init__()
        self.dim = dim

        self.norm = LayerNormalization()
        self.mlp = tf.keras.Sequential([
            Conv1D(dim * 4, 1, 1, "same", kernel_initializer="he_normal"),
            LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
            # tf.keras.layers.Dropout(0.1),
            Conv1D(dim, 1, 1, "same")

            # Dense(dim * 4, "gelu", kernel_initializer="he_normal"),
            # # tf.keras.layers.Dropout(0.1),
            # Dense(dim)
        ])

    def call(self, inputs, *args, **kwargs):
        x = self.norm(inputs)
        return self.mlp(x) + inputs

    def get_config(self):
        config = {}
        # config = super(TransformerMlp, self).get_config()
        config.update({"dim": self.dim})

        return config


class LambdaLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, heads=4, use_bias=False, u=4, kernel_size=7):
        super(LambdaLayer, self).__init__()

        self.out_dim = out_dim
        k = 5
        # k = out_dim // 1
        self.heads = heads
        self.v = out_dim // heads
        self.u = u
        self.kernel_size = kernel_size
        self.use_bias = use_bias

        self.top_q = tf.keras.layers.Conv1D(k * heads, 1, 1, "same", use_bias=use_bias)
        self.top_k = tf.keras.layers.Conv1D(k * u, 1, 1, "same", use_bias=use_bias)
        self.top_v = tf.keras.layers.Conv1D(self.v * self.u, 1, 1, "same", use_bias=use_bias)

        self.norm_q = tf.keras.layers.LayerNormalization()
        self.norm_v = tf.keras.layers.LayerNormalization()

        self.pos_conv = tf.keras.layers.Conv2D(k, (1, self.kernel_size), padding="same")

    def call(self, inputs, *args, **kwargs):
        u, h = self.u, self.heads

        q = self.top_q(inputs)
        k = self.top_k(inputs)
        v = self.top_v(inputs)

        q = self.norm_q(q)
        v = self.norm_v(v)

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
        config = {}
        # config = super(LambdaLayer, self).get_config()
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

        if "_optimizer" in dir(self.optimizer):  # mixed float policy
            grads_and_vars = self.optimizer._optimizer._compute_gradients(loss, var_list=self.trainable_variables,
                                                                          tape=tape)
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


class Model:
    def __init__(self, types: str):
        """
        Parameters

        types: "dense" or "resnet" or "transformer"
        """
        self.types = types

    def init_conv_option(self, num_layers, dim=32, se=False, dense_next=False, transformer=False, bot=False,
                         mix_net=False,erase_relu=False, sam=False, cbam=False, lambda_net=False, lambda_bot=False,
                         pyconv=False, vit=False):
        self.num_layers = num_layers
        self.dim = dim

        self.se = se
        self.groups = 16 if dense_next else 1
        self.transformer = transformer
        self.bot = bot
        self.mix = mix_net
        self.erase = erase_relu
        self.sam = sam
        self.cbam = cbam
        self.lambda_net = lambda_net
        self.lambda_bot = lambda_bot
        self.pyconv = pyconv
        self.vit = vit

        self.last = False

        self.layer_name = \
            "pyconv" if pyconv else "lambdalayer" if lambda_net else "conv1d"
        self.conv_args = {
            "filters": self.dim, "kernel_size": 3, "kernel_initializer": "he_normal", "groups": self.groups,
            "padding": "same"
        }

        self.bot_name = True if (bot or lambda_bot) else None
        if self.bot_name:
            self.bot_name = "lambdalayer" if lambda_bot else "attention"

    def init_transformer_option(self, num_layers, dim, kernel_size, heads=12, sam=False, lambda_=False):
        self.num_layers = num_layers
        self.dim = dim
        self.kernel_size = kernel_size
        self.heads = heads
        self.sam = sam
        self.lambda_ = lambda_

    def layer(self, x, layer_name):
        if layer_name == "multiheadattention":
            return MultiHeadAttention(4, self.dim, dropout=0.)(x, x)
        elif layer_name == "attention":
            return Attention(8, self.dim, 0.)(x)
        elif layer_name == "lambdalayer":
            return LambdaLayer(self.dim)(x)
        elif layer_name == "conv1d":
            self.conv_args.update({"filters": self.dim})
            return Conv1D(**self.conv_args)(x)
        elif layer_name == "pyconv":
            return Pyconv(self.dim)(x)

    def block(self, x, l):
        for _ in range(l):
            x = self.dense_block(x) if self.types == "dense" else self.res_block(x) if self.types == "resnet" else \
                self.trans_block(x)

        if not self.last:
            x = self.conv_transition(x) if self.types == "dense" or self.types == "resnet" else self.trans_transition(x)
        else:
            if self.types == "dense" or self.types == "resnet":
                if self.vit:
                    x = LayerNormalization()(x)
                    x = ELU()(x)
                    x = Conv1D(self.dim, 1, 1, "same")(x)

                    self.init_transformer_option(None, self.dim, None, 8)
                    for _ in range(3):
                        x = self.trans_block(x)

        return x

    def conv_block(self, x):
        x = LayerNormalization()(x)
        if not self.erase:
            x = ELU()(x)
        x = Conv1D(self.dim * 4 if self.types == "dense" else 1, 3, 1, "same", kernel_initializer="he_normal")(x)

        x = LayerNormalization()(x)
        x = ELU()(x)

        layer_name = self.bot_name if self.bot_name is not None and self.last else self.layer_name
        x = self.layer(x, layer_name)

        if not (self.bot_name is not None and self.last):
            if self.se:
                x = SE(self.dim)(x)
            elif self.cbam:
                x = CBAM(self.dim)(x)

        return x

    def trans_block(self, inputs):
        x = LayerNormalization()(inputs)
        # x = MultiHeadAttention(self.heads, self.dim)(x, x)
        if self.lambda_:
            x = LambdaLayer(self.dim, 4)(x)
        else:
            x = Attention(self.heads, self.dim)(x)
        x = Add()([inputs, x])
        x = TransformerMlp(self.dim)(x)

        return x

    def dense_block(self, inputs):
        if self.mix:
            r = self.conv_block(inputs)
            d = self.conv_block(inputs)
            r = Add()([inputs[:, :, -self.dim:], r])
            x = Concatenate()([inputs[:, :, :-self.dim], r, d])
        else:
            d = self.conv_block(inputs)
            x = Concatenate()([inputs, d])
        return x

    def res_block(self, inputs):
        x = self.conv_block(inputs)
        x = tf.keras.layers.Add()([inputs, x])

        return x

    def conv_transition(self, x):
        self.dim *= 2 if self.types == "resnet" else 1
        dim = x.get_shape()[-1] // 2 if self.types == "dense" else self.dim

        x = LayerNormalization()(x)
        x = ELU()(x)
        x = tf.keras.layers.AvgPool1D()(x)
        x = Conv1D(dim, 1, 1, "same", kernel_initializer="he_normal")(x)

        return x

    def trans_transition(self, x):
        # self.dim *= 2
        x = Conv1D(self.dim, 1, 2, "same")(x)
        return x

    def build_model(self, input_shape, output_size):
        bot_name = None
        num_layers = self.num_layers
        if self.types == "dense" or self.types == "resnet":
            bot_name = self.bot_name
            self.bot_name = None

        if self.types == "transformer":
            inputs, x = Inputs(input_shape, self.dim, self.kernel_size, self.kernel_size, False)
        else:
            inputs, x = Inputs(input_shape, self.dim, 7, 2, True)

        last = len(num_layers) - 1

        for i, l in enumerate(num_layers):
            self.last = True if last == i else False
            x = self.block(x, l)

        if self.types == "dense" or self.types == "resnet":
            self.bot_name = bot_name
            if self.bot_name is not None:
                x = self.block(x, 6)

        x = LayerNormalization()(x)
        x = tf.keras.layers.GlobalAvgPool1D()(x)
        x = tf.keras.layers.Dense(output_size)(x)

        return SAMModel(inputs, x) if self.sam else tf.keras.Model(inputs, x)
