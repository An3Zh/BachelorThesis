import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import keras

# --- CONFIG ---
model_path = "Simple_savedmodel"  # Your model file
output_plot_path = "inference_result.png"  # Output image path
# Replace these with actual paths to .TIF patches
red_path = "Test1Sample/red.TIF"
green_path = "Test1Sample/green.TIF"
blue_path = "Test1Sample/blue.TIF"
nir_path = "Test1Sample/nir.TIF"


# --- FUNCTIONS ---
def load_patch(r_path, g_path, b_path, n_path, target_size=(192, 192)):
    def load_and_resize(path):
        img = Image.open(path)
        if img.mode not in ["L", "I", "F"]:
            img = img.convert("I")  # convert to 32-bit signed integer pixels
        img = img.resize(target_size, Image.BILINEAR)
        return np.array(img, dtype=np.uint16) / 65535.0

    r = load_and_resize(r_path)
    g = load_and_resize(g_path)
    b = load_and_resize(b_path)
    n = load_and_resize(n_path)
    stacked = np.stack([r, g, b, n], axis=-1)  # [192, 192, 4]
    return np.expand_dims(stacked.astype(np.float32), axis=0)  # [1, 192, 192, 4]


def plot_input_and_output(input_tensor, prediction, save_path):

    plt.imshow(prediction[0, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


# --- MAIN EXECUTION ---
# Load model
#keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model(model_path)

# Prepare input tensor
input_tensor = load_patch(red_path, green_path, blue_path, nir_path)

# Run inference
prediction = model.predict(input_tensor)

# Plot and save
plot_input_and_output(input_tensor, prediction, output_plot_path)
