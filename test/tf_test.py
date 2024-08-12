import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Simple TensorFlow operation
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[1, 1], [1, 1]])
print("TensorFlow operation result:")
print(tf.matmul(a, b))

# Simple NumPy operation
x = np.array([[1, 2], [3, 4]])
y = np.array([[1, 1], [1, 1]])
print("NumPy operation result:")
print(np.dot(x, y))

print("Test completed successfully!")