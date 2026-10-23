from tensorflow.keras.models import load_model
import numpy as np

model = load_model("mnist_model.h5")

# test with dummy input
x = np.zeros((1,28,28,1))
prediction = model.predict(x)

input("Press Enter to exit...")
print("Working ✅", prediction)
