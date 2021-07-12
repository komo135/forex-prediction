import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import network

x, y = np.load("x.npy"), np.load("y.npy")
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=42)

model_name = Input("model name: ")

train_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, verbose=1
    )
]

train_type = Input("type GPU or CPU or TPU: ")
train_type = train_type.lower()

if train_type == "tpu":
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  tf.config.experimental_connect_to_cluster(resolver)
  # This is the TPU initialization code that has to be at the beginning.
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.TPUStrategy(resolver)
  with strategy.scope():
    model = network.build_model(model_name, x.shape[-2:], 2)
    model.compile(tf.keras.optimizers.Adam(1e-3), "SparseCategoricalCrossentropy", ["accuracy"], steps_per_execution = 100)
else:
  if train_type == "gpu"
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
  model = network.build_model(model_name, x.shape[-2:], 2)
  model.compile(tf.keras.optimizers.Adam(1e-3), "SparseCategoricalCrossentropy", ["accuracy"])

  
model.fit(train_x, train_y, 512, 100, validation_data=(test_x, test_y),callbackscallbacks=train_callbacks, workers=1000, use_multiprocessing=True)
