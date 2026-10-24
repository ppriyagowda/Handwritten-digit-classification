import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_test = x_test.astype("float32") / 255.0

# Load the already trained model
model = tf.keras.models.load_model("mnist_model.h5")

# Evaluate the model without training again
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print("\nModel Evaluation Results")
print("------------------------")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")