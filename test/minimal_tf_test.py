import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
print(tf.add(a, b))