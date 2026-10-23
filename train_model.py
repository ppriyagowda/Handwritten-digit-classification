import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ════════════════════════════════════════════════════════
#  STEP 1 — Load MNIST Dataset
# ════════════════════════════════════════════════════════
print("📦 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1,  28, 28, 1)

print(f"✅ Train samples : {x_train.shape[0]}")
print(f"✅ Test  samples : {x_test.shape[0]}")

# ════════════════════════════════════════════════════════
#  STEP 2 — Data Augmentation
# ════════════════════════════════════════════════════════
print("\n🔄 Setting up Data Augmentation...")

datagen = ImageDataGenerator(
    rotation_range     = 10,    # Rotate up to 10 degrees
    zoom_range         = 0.10,  # Zoom in/out 10%
    width_shift_range  = 0.10,  # Shift left/right 10%
    height_shift_range = 0.10,  # Shift up/down 10%
    shear_range        = 0.10,  # Slant transformation
    fill_mode          = 'nearest'
)

datagen.fit(x_train)
print("✅ Augmentation ready!")

# ════════════════════════════════════════════════════════
#  STEP 3 — Visualize Augmented Samples
# ════════════════════════════════════════════════════════
print("\n🖼️  Saving augmentation preview...")

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("Original  vs  Augmented Samples",
             fontsize=13, fontweight='bold', color='white')
fig.patch.set_facecolor('#0d0d1a')

for i in range(8):
    axes[0][i].imshow(x_train[i].reshape(28,28), cmap='gray')
    axes[0][i].axis('off')
    axes[0][i].set_title("Orig", fontsize=8, color='#aaaacc')

aug_iter = datagen.flow(x_train[:8], batch_size=8, shuffle=False)
aug_imgs = next(aug_iter)
for i in range(8):
    axes[1][i].imshow(aug_imgs[i].reshape(28,28), cmap='gray')
    axes[1][i].axis('off')
    axes[1][i].set_title("Aug", fontsize=8, color='#FFD700')

plt.tight_layout()
plt.savefig('augmentation_preview.png', dpi=120,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print("✅ Saved: augmentation_preview.png")

# ════════════════════════════════════════════════════════
#  STEP 4 — Your Same CNN + Dropout Added
# ════════════════════════════════════════════════════════
print("\n🧠 Building CNN Model...")

model = keras.Sequential([

    # ✅ Same as your original
    keras.layers.Conv2D(32, (3,3), activation='relu',
                        input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),

    # ✅ Same as your original
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    # 🆕 Added Dropout to reduce overfitting
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),

    # ✅ Same as your original
    keras.layers.Dense(128, activation='relu'),

    # 🆕 Added Dropout before output
    keras.layers.Dropout(0.3),

    # ✅ Same as your original
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

# ════════════════════════════════════════════════════════
#  STEP 5 — Compile  (✅ exact same as your original)
# ════════════════════════════════════════════════════════
model.compile(
    optimizer = 'adam',
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)

# ════════════════════════════════════════════════════════
#  STEP 6 — Callbacks
# ════════════════════════════════════════════════════════
callbacks = [

    keras.callbacks.ModelCheckpoint(
        'mnist_model.h5',            # ✅ your exact filename
        monitor        = 'val_accuracy',
        save_best_only = True,
        verbose        = 1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_accuracy',
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-6,
        verbose  = 1
    ),

    keras.callbacks.EarlyStopping(
        monitor              = 'val_accuracy',
        patience             = 8,
        restore_best_weights = True,
        verbose              = 1
    )
]

# ════════════════════════════════════════════════════════
#  STEP 7 — Train with Augmentation
# ════════════════════════════════════════════════════════
print("\n🚀 Training with Data Augmentation...")
print("   This will take a few minutes...\n")

BATCH_SIZE = 64
EPOCHS     = 15       # More than your original 5 — better accuracy

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs          = EPOCHS,
    validation_data = (x_test, y_test),
    steps_per_epoch = len(x_train) // BATCH_SIZE,
    callbacks       = callbacks,
    verbose         = 1
)

# ════════════════════════════════════════════════════════
#  STEP 8 — Evaluate
# ════════════════════════════════════════════════════════
print("\n📊 Evaluating...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy : {accuracy*100:.2f}%")
print(f"✅ Test Loss     : {loss:.4f}")

# ════════════════════════════════════════════════════════
#  STEP 9 — Plot Training History
# ════════════════════════════════════════════════════════
print("\n📈 Saving training chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d0d1a')
epochs_ran = range(1, len(history.history['accuracy']) + 1)

ax1.set_facecolor('#111122')
ax1.plot(epochs_ran, history.history['accuracy'],
         color='#FFD700', linewidth=2, label='Train Accuracy')
ax1.plot(epochs_ran, history.history['val_accuracy'],
         color='#00ff88', linewidth=2, label='Val Accuracy')
ax1.set_title('Accuracy', color='white', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', color='#aaaacc')
ax1.set_ylabel('Accuracy', color='#aaaacc')
ax1.tick_params(colors='#aaaacc')
ax1.legend(facecolor='#1a1a2e', labelcolor='white')
ax1.grid(color='#222244', linestyle='--', alpha=0.5)
for spine in ax1.spines.values():
    spine.set_edgecolor('#333355')

ax2.set_facecolor('#111122')
ax2.plot(epochs_ran, history.history['loss'],
         color='#FF6B6B', linewidth=2, label='Train Loss')
ax2.plot(epochs_ran, history.history['val_loss'],
         color='#CD7F32', linewidth=2, label='Val Loss')
ax2.set_title('Loss', color='white', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch', color='#aaaacc')
ax2.set_ylabel('Loss', color='#aaaacc')
ax2.tick_params(colors='#aaaacc')
ax2.legend(facecolor='#1a1a2e', labelcolor='white')
ax2.grid(color='#222244', linestyle='--', alpha=0.5)
for spine in ax2.spines.values():
    spine.set_edgecolor('#333355')

plt.suptitle(
    f'Training Results  |  Final Accuracy: {accuracy*100:.2f}%',
    color='#FFD700', fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('training_results.png', dpi=120,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.close()

print("✅ Saved: training_results.png")
print("\n🎉 Done! Model saved as: mnist_model.h5")
print(f"🔥 Accuracy: {accuracy*100:.2f}%")
print("file saved location:",os.getcwd())