import tensorflow as tf

import numpy as np
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


def conv1(x):
    x = tf.keras.layers.Conv1D(4, 7, 4, "causal", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv1D(8, 5, 2, "causal", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv1D(16, 3, 1, "causal", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def tcn(x):
    def block(i, f, d):
        x = i

        for _ in range(2):
            x = tf.keras.layers.Conv1D(f, 3, 1, "causal", dilation_rate=d, kernel_initializer="he_normal")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("elu")(x)
            # x = tf.keras.layers.Dropout(0.2)(x)

        try:
            x = tf.keras.layers.Add()([x, i])
        except:
            i = tf.keras.layers.Conv1D(f, 1, 1, "same", kernel_initializer="he_normal")(i)
            x = tf.keras.layers.Add()([x, i])

        return x

    x = tf.keras.layers.Conv1D(8, 5, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.AveragePooling1D(2, 2)(x)

    x = block(x, 8, 1)
    x = tf.keras.layers.AvgPool1D()(x)
    x = tf.keras.layers.Conv1D(16, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = block(x, 16, 1)
    # x = tf.keras.layers.AvgPool1D()(x)
    # x = tf.keras.layers.Conv1D(16, 1, 1, "same", kernel_initializer="he_normal")(x)
    # x = block(x, 32, 1)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def dense_tcn(x):
    def block(i, f, d):
        x = i

        for _ in range(2):
            x_ = tf.keras.layers.Conv1D(f, 3, 1, "causal", dilation_rate=d, kernel_initializer="he_normal")(x)
            x_ = tf.keras.layers.BatchNormalization()(x_)
            x_ = tf.keras.layers.ELU()(x_)
            x_ = tf.keras.layers.Dropout(0.3)(x_)
            x = tf.keras.layers.Concatenate()([x, x_])

        x = tf.keras.layers.Conv1D(f, 1, 1, "same", kernel_initializer="he_normal")(x_)
        return x

    x = block(x, 16, 1)
    x = block(x, 16, 2)
    x = block(x, 16, 4)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def rocket(x):
    def block(i, f, d):
        x = i

        for _ in range(2):
            k = int(np.random.choice([5, 7, 9]))
            x = tf.keras.layers.Conv1D(f, k, 1, "causal", dilation_rate=d, kernel_initializer="he_normal")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.3)(x)

        try:
            return x + i
        except:
            i = tf.keras.layers.Conv1D(f, 1, 1, "same", kernel_initializer="he_normal")(i)
            return x + i

    x = block(x, 16, 1)
    x = block(x, 16, 1)
    x = block(x, 16, 2)
    x = block(x, 16, 2)

    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def gru_tcn(x):
    def gru(x):
        x = tf.keras.layers.Permute((2, 1))(x)
        x = tf.keras.layers.GRU(16)(x)
        return x

    x_gru = gru(x)
    x_resnet = tcn(x)
    x = tf.keras.layers.Concatenate()([x_gru, x_resnet])
    return x


def resnet(x):
    def block(i, f, s):
        x = i
        if s == 2:
            i = tf.keras.layers.Conv1D(f, 1, s, "same", kernel_initializer="he_normal")(i)
            i = tf.keras.layers.LayerNormalization()(i)

        x = tf.keras.layers.Conv1D(f, 3, s, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Add()([x, i])
        x = tf.keras.layers.ELU()(x)

        # x = tf.keras.layers.GaussianDropout(0.05)(x)
        return x

    x = tf.keras.layers.Conv1D(32, 5, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.AveragePooling1D(2, 2)(x)

    for _ in range(4):
        x = block(x, 32, 1)
    for i in range(6):
        s = 1 if i != 0 else 2
        x = block(x, 64, s)

    x = tf.keras.layers.Conv1D(64, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)

    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def densenet(x):
    def block(i, f, n):
        concat = [i]
        x = i
        for _ in range(n):
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Conv1D(4 * f, 1, 1, "same", kernel_initializer="he_normal")(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
            concat.append(x)
            x = tf.keras.layers.Concatenate()(concat)

        return x

    x = tf.keras.layers.Conv1D(16, 7, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.AvgPool1D(3, 2)(x)

    x = block(x, 16, 6)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 32 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.AvgPool1D(2, 2)(x)
    x = block(x, 32, 12)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 64 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.AvgPool1D(2, 2)(x)
    x = block(x, 64, 12)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def se(x, r=8):
    c = x.get_shape().as_list()[-1]
    w0 = tf.keras.layers.Dense(c // r, "elu", kernel_initializer="he_normal")
    w1 = tf.keras.layers.Dense(c)

    avg_x = tf.keras.layers.GlobalAvgPool1D()(x)
    # max_x = tf.keras.layers.GlobalMaxPool1D()(x)

    avg_x = w0(avg_x)
    x1 = w1(avg_x)
    x1 = tf.reshape(x1, (-1, 1, c))
    x1 = tf.keras.layers.Activation("sigmoid")(x1)
    return x1


def se_resnet(x):
    def block(i, f, s):
        x = i
        if s == 2:
            i = tf.keras.layers.Conv1D(f, 1, s, "same", kernel_initializer="he_normal")(i)
            i = tf.keras.layers.LayerNormalization()(i)

        x = tf.keras.layers.Conv1D(f, 3, s, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.GaussianDropout/(0.1)(x)
        x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        se_ = se(x, 2)
        x = tf.keras.layers.Multiply()([x, se_])
        x = tf.keras.layers.Add()([x, i])
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.GaussianDropout(0.1)(x)
        # x = tf.keras.layers.Dropout(0.2)(x)

        # x = tf.keras.layers.GaussianDropout(0.05)(x)
        return x

    x = tf.keras.layers.Conv1D(16, 5, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    # x = tf.keras.layers.GaussianDropout(0.1)(x)
    x = tf.keras.layers.AveragePooling1D(3, 2)(x)

    for _ in range(4):
        x = block(x, 16, 1)
    for i in range(6):
        s = 1 if i != 0 else 2
        x = block(x, 32, s)

    x = tf.keras.layers.Conv1D(32, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    # x = tf.keras.layers.GaussianDropout(0.1)(x)

    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def xception(x):
    def block(f, i, first=False):
        x = i
        if first:
            x = tf.keras.layers.SeparableConv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.SeparableConv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        if not first:
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.SeparableConv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.MaxPool1D(3, 2, "same")(x)
        i = tf.keras.layers.Conv1D(f, 1, 2, "same", kernel_initializer="he_normal")(i)
        x = tf.keras.layers.Add()([x, i])

        return x

    x = tf.keras.layers.Conv1D(32, 3, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(64, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)

    x = block(128 * 5, x, True)
    x = block(256 * 5, x, False)

    x = tf.keras.layers.SeparableConv1D(312 * 5, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.SeparableConv1D(356 * 5, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def se_xception(x):
    def block(f, i, first=False):
        x = i
        if first:
            x = tf.keras.layers.SeparableConv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.SeparableConv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        if not first:
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.SeparableConv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.MaxPool1D(3, 2, "same")(x)
        se_ = se(x, 16)
        x = tf.keras.layers.Multiply()([x, se_])
        i = tf.keras.layers.Conv1D(f, 1, 2, "same", kernel_initializer="he_normal")(i)
        x = tf.keras.layers.Add()([x, i])

        return x

    x = tf.keras.layers.Conv1D(16, 3, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(32, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)

    x = block(64 * 5, x, True)
    x = block(128 * 5, x, False)

    x = tf.keras.layers.SeparableConv1D(192 * 5, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.SeparableConv1D(256 * 5, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def se_wide_resnet(x):
    def block(i, f, s):
        x = i
        if s == 2:
            i = tf.keras.layers.Conv1D(f, 1, s, "same", kernel_initializer="he_normal")(i)
            i = tf.keras.layers.LayerNormalization()(i)

        x = tf.keras.layers.Conv1D(f, 3, s, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        se_ = se(x, 2)
        x = tf.keras.layers.Multiply()([x, se_])
        x = tf.keras.layers.Add()([x, i])
        x = tf.keras.layers.ELU()(x)

        # x = tf.keras.layers.GaussianDropout(0.05)(x)
        return x

    x = tf.keras.layers.Conv1D(16, 7 , 2, "same", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.AveragePooling1D(3, 2)(x)

    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 16 * 5, s)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 32 * 5, s)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 64 * 5, s)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)

    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def wide_resnet(x):
    def block(i, f, s):
        x = i
        if s == 2:
            i = tf.keras.layers.Conv1D(f, 1, s, "same", kernel_initializer="he_normal")(i)

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation("elu")(x)
        x = tf.keras.layers.GaussianNoise(0.1)(x)
        x = tf.keras.layers.Conv1D(f, 3, s, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation("elu")(x)
        x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.Add()([x, i])
        return x

    x = tf.keras.layers.Conv1D(32, 5, 2, "same", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.AveragePooling1D(3, 2)(x)

    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 32 * 5, s)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 64 * 5, s)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 64 * 5, s)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)
    x = tf.keras.layers.Conv1D(64 * 5, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)

    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def resnetxt(x):
    def block(i, f, s):
        x = i
        if s == 2:
            i = tf.keras.layers.Conv1D(f, 1, s, "same", kernel_initializer="he_normal")(i)

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation("elu")(x)
        x = tf.keras.layers.Conv1D(f, 3, s, "same", kernel_initializer="he_normal", groups=32)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation("elu")(x)
        x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal", groups=32)(x)
        x = tf.keras.layers.Add()([x, i])
        return x

    x = tf.keras.layers.Conv1D(64, 7, 2, "same", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.AveragePooling1D(3, 2)(x)

    # x = tf.keras.layers.Dropout(0.3)(x)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 64, s)
    # x = tf.keras.layers.Dropout(0.3)(x)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 128, s)
    # x = tf.keras.layers.Dropout(0.3)(x)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 256, s)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)
    x = tf.keras.layers.Conv1D(256, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)

    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def erace_wide_resnet(x):
    def block(i, f, s):
        x = i
        if s == 2:
            i = tf.keras.layers.Conv1D(f, 1, s, "same", kernel_initializer="he_normal")(i)

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Conv1D(f, 3, s, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation("elu")(x)
        x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.Add()([x, i])
        return x

    x = tf.keras.layers.Conv1D(8, 5, 2, "same", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.AveragePooling1D(3, 2)(x)

    # x = tf.keras.layers.Dropout(0.3)(x)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 8 * 5, s)
    # x = tf.keras.layers.Dropout(0.3)(x)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 16 * 5, s)
    # x = tf.keras.layers.Dropout(0.3)(x)
    for i in range(5):
        s = 1 if i != 0 else 2
        x = block(x, 32 * 5, s)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)
    x = tf.keras.layers.Conv1D(32 * 5, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("elu")(x)

    x = tf.keras.layers.GlobalAvgPool1D()(x)
    # x = tf.keras.layers.GaussianNoise(0.1)(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    return x


def repvgg(x):
    def block(x, f, n):
        i = x

        for e in range(n):
            s = 2 if e == 0 else 1
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            i_ = tf.keras.layers.Conv1D(f, 1, s, "same", kernel_initializer="he_normal")(x)
            x = tf.keras.layers.Conv1D(f, 3, s, "same", kernel_initializer="he_normal")(x)
            if s == 1:
                x = tf.keras.layers.Add()([x, i_, i])
            else:
                x = tf.keras.layers.Add()([x, i_])
            i = x

        return x

    a = 2
    b = 4

    x = tf.keras.layers.Conv1D(64 * a, 7, 2, "same", kernel_initializer="he_normal")(x)

    x = block(x, 64 * a, 4)
    x = block(x, 128 * a, 8)
    x = block(x, 256 * b, 1)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def se_densenet(x):
    def block(i, f, n):
        concat = [i]
        x = i
        for _ in range(n):
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.GaussianNoise(0.1)(x)
            x = tf.keras.layers.Conv1D(4 * f, 1, 1, "same", kernel_initializer="he_normal")(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
            se_ = se(x, 8)
            x = tf.keras.layers.Multiply()([x, se_])
            concat.append(x)
            x = tf.keras.layers.Concatenate()(concat)

        return x

    x = tf.keras.layers.Conv1D(16, 7, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(3, 2)(x), tf.keras.layers.MaxPool1D(3, 2)(x)])
    # x = tf.keras.layers.AvgPool1D(3, 2)(x)

    x = block(x, 16, 6)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 32 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    # x = tf.keras.layers.AvgPool1D(2, 2)(x)
    x = block(x, 32, 12)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 64 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    # x = tf.keras.layers.AvgPool1D(2, 2)(x)
    x = block(x, 64, 12)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def choicenet(x):
    def module(x, f, k):
        i = x
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.GaussianNoise(0.1)(x)
        x = tf.keras.layers.Conv1D(f, k, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(f, k, 1, "same", kernel_initializer="he_normal")(x)
        add_x = tf.keras.layers.Add()([x, i])
        return x, add_x

    def block(x, f, n=3):
        concat = [x]
        for _ in range(n):

            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.GaussianNoise(0.1)(x)
            x = tf.keras.layers.Conv1D(f * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
            # concat.append(x)

            for k in [3, 5, 7]:
                concat.extend(module(x, f, k))
            x = tf.keras.layers.Concatenate()(concat)

        return x

    x = tf.keras.layers.Conv1D(8, 7, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(3, 2)(x), tf.keras.layers.MaxPool1D(3, 2)(x)])

    x = block(x, 8, 3)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 16 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    x = block(x, 16, 6)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 32 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    x = block(x, 32, 6)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def se_choicenet(x):
    def module(x, f, k):
        i = x
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.GaussianNoise(0.1)(x)
        x = tf.keras.layers.Conv1D(f, k, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(f, k, 1, "same", kernel_initializer="he_normal")(x)
        add_x = tf.keras.layers.Add()([x, i])
        return x, add_x

    def block(x, f, n=3):
        concat = [x]
        for _ in range(n):

            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            # x = tf.keras.layers.GaussianNoise(0.1)(x)
            x = tf.keras.layers.Conv1D(f * 1, 1, 1, "same", kernel_initializer="he_normal")(x)

            c = []

            for k in [3, 5, 7]:
                c.extend(module(x, f, k))
            x = tf.keras.layers.Concatenate()(c)
            se_ = se(x, 8)
            x = tf.keras.layers.Multiply()([x, se_])
            concat.append(x)
            x = tf.keras.layers.Concatenate()(concat)

        return x

    x = tf.keras.layers.Conv1D(16, 7, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(3, 2)(x), tf.keras.layers.MaxPool1D(3, 2)(x)])

    x = block(x, 16, 3)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 32 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    x = block(x, 32, 6)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 64 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    x = block(x, 64, 6)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def cbam(x:tf.keras.layers.Layer, r:int):
    c = x.get_shape().as_list()[-1]
    w0 = tf.keras.layers.Dense(c // r, "elu", kernel_initializer="he_normal")
    w1 = tf.keras.layers.Dense(c)

    avg_x = tf.keras.layers.GlobalAvgPool1D()(x)
    max_x = tf.keras.layers.GlobalMaxPool1D()(x)

    avg_x = w0(avg_x)
    max_x = w0(max_x)
    x_ = tf.keras.layers.Activation("sigmoid")(w1(avg_x) + w1(max_x))
    x_ = tf.reshape(x_, (-1, 1, c))
    x = tf.keras.layers.Multiply()([x, x_])

    avg_x = tf.keras.backend.mean(x, 2, keepdims=True)
    max_x = tf.keras.backend.max(x, 2, keepdims=True)

    x_ = tf.keras.layers.Concatenate(axis=2)([avg_x, max_x])
    x_ = tf.keras.layers.Conv1D(1, 7, 1, "same")(x_)
    x_ = tf.keras.layers.LayerNormalization()(x_)
    x_ = tf.keras.layers.Activation("sigmoid")(x_)
    x = tf.keras.layers.Multiply()([x, x_])
    return x


def cbam_choicenet(x):
    def module(x, f, k):
        i = x
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.GaussianNoise(0.1)(x)
        x = tf.keras.layers.Conv1D(f, k, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(f, k, 1, "same", kernel_initializer="he_normal")(x)
        add_x = tf.keras.layers.Add()([x, i])
        return x, add_x

    def block(x, f, n=3):
        concat = [x]
        for _ in range(n):

            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Conv1D(f * 1, 1, 1, "same", kernel_initializer="he_normal")(x)

            c = []

            for k in [3, 5, 7]:
                c.extend(module(x, f, k))
            x = tf.keras.layers.Concatenate()(c)
            x = cbam(x, 8)
            concat.append(x)
            x = tf.keras.layers.Concatenate()(concat)

        return x

    x = tf.keras.layers.Conv1D(16, 7, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(3, 2)(x), tf.keras.layers.MaxPool1D(3, 2)(x)])

    x = block(x, 16, 3)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 32 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    # x = cbam(x, 8)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    x = block(x, 32, 6)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 64 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    # x = cbam(x, 8)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    x = block(x, 64, 6)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def cbam_densenet(x):
    def block(i, f, n):
        concat = [i]
        x = i
        for _ in range(n):
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.GaussianNoise(0.1)(x)
            x = tf.keras.layers.Conv1D(4 * f, 1, 1, "same", kernel_initializer="he_normal")(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
            x = cbam(x, 8)
            concat.append(x)
            x = tf.keras.layers.Concatenate()(concat)

        return x

    x = tf.keras.layers.Conv1D(16, 7, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(3, 2)(x), tf.keras.layers.MaxPool1D(3, 2)(x)])

    x = block(x, 16, 6)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 32 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    x = block(x, 32, 12)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 64 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    x = block(x, 64, 12)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def bom(x, r=8):
    c = x.get_shape().as_list()[-1]
    w0 = tf.keras.layers.Dense(c // r, "elu", kernel_initializer="he_normal")
    w1 = tf.keras.layers.Dense(c)

    avg_x = tf.keras.layers.GlobalAvgPool1D()(x)
    max_x = tf.keras.layers.GlobalMaxPool1D()(x)

    avg_x = w0(avg_x)
    max_x = w0(max_x)
    x1 = w1(avg_x) + w1(max_x)
    x1 = tf.reshape(x1, (-1, 1, c))

    x2 = tf.keras.layers.Conv1D(c // r, 1, 1, "same", kernel_initializer="he_normal")(x)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tf.keras.layers.ELU()(x2)
    x2 = tf.keras.layers.Conv1D(c // r, 3, 1, "same", kernel_initializer="he_normal", dilation_rate=4)(x2)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tf.keras.layers.ELU()(x2)
    x2 = tf.keras.layers.Conv1D(c // r, 3, 1, "same", kernel_initializer="he_normal", dilation_rate=4)(x2)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tf.keras.layers.ELU()(x2)
    x2 = tf.keras.layers.Conv1D(1, 1, 1, "same", kernel_initializer="he_normal")(x2)
    x2 = tf.keras.layers.LayerNormalization()(x2)

    x_ = tf.keras.layers.Add()([x1, x2])
    x_ = tf.keras.layers.Activation("sigmoid")(x_)
    x_ = tf.keras.layers.Multiply()([x, x_])
    x = tf.keras.layers.Add()([x, x_])

    return x


def bom_repvgg(x):
    def block(x, f, n):
        i = x

        for e in range(n):
            s = 2 if e == 0 else 1
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            i_ = tf.keras.layers.Conv1D(f, 1, s, "same", kernel_initializer="he_normal")(x)
            x = tf.keras.layers.Conv1D(f, 3, s, "same", kernel_initializer="he_normal")(x)
            if s == 1:
                x = tf.keras.layers.Add()([x, i_, i])
            else:
                x = tf.keras.layers.Add()([x, i_])
            i = x

        return x

    a = 2
    b = 4

    x = tf.keras.layers.Conv1D(64 * a, 7, 2, "same", kernel_initializer="he_normal")(x)

    x = block(x, 64 * a, 4)
    x = bom(x, 16)
    x = block(x, 128 * a, 8)
    x = bom(x, 16)
    x = block(x, 256 * b, 1)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def bom_densenet(x):
    def block(i, f, n):
        concat = [i]
        x = i
        for _ in range(n):
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.GaussianNoise(0.1)(x)
            x = tf.keras.layers.Conv1D(4 * f, 1, 1, "same", kernel_initializer="he_normal")(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Conv1D(f, 3, 1, "same", kernel_initializer="he_normal")(x)
            concat.append(x)
            x = tf.keras.layers.Concatenate()(concat)

        return x

    x = tf.keras.layers.Conv1D(16, 7, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(3, 2)(x), tf.keras.layers.MaxPool1D(3, 2)(x)])
    # x = tf.keras.layers.AvgPool1D(3, 2)(x)

    x = block(x, 16, 6)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 32 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = bom(x, 16)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    # x = tf.keras.layers.AvgPool1D(2, 2)(x)
    x = block(x, 32, 12)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv1D(4 * 64 * 1, 1, 1, "same", kernel_initializer="he_normal")(x)
    x = bom(x, 16)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.AvgPool1D(2, 2)(x), tf.keras.layers.MaxPool1D(2, 2)(x)])
    # x = tf.keras.layers.AvgPool1D(2, 2)(x)
    x = block(x, 64, 12)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    return x


def model(dim: tuple, action_size: int, dueling: bool, noisy: bool) -> tf.keras.Model:
    i = tf.keras.layers.Input(dim, name="i")
    
    x = bom_densenet(i)

    x = output(x, noisy, dueling, action_size)

    return tf.keras.Model(i, x)
