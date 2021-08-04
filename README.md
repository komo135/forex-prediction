# How to use

### install tensorflow_prediction
```
python setup.py install
```

```python3
import numpy as np
import tensorflow as tf
from tensorflow_prediction import forex_data, model as build_model

pred_length = 1

## generate training data
train_x, test_x, train_y, test_y = forex_data.gen_data("EURUSD", forex_data.m1_args, pred_length)

## build model
#pattern 1
model = build_model.build_model(build_model.dense_net, (120, 1), pred_length)
#pattern 2
model = build_model.dense_model
model_opt = model.init_conv_option
model.init_conv_option(128, (3, 6, 3))
mdoel = model.build_model((120, 1), pred_length)

## compile model
opt = tf.keras.optimizers.Adam(1e-3)
model.compile(opt, "mse", ["mae"])
train_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, verbose=1
    )
]

## training model
model.fit(train_x, train_y, 512, 100, validation_data=(test_x, test_y), callbacks=train_callbacks)
```
