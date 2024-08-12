import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Generate some random data
x = np.random.random((1000, 5))
y = np.random.random((1000, 1))

# Fit the model
model.fit(x, y, epochs=1, verbose=1)

print("Test completed successfully!")