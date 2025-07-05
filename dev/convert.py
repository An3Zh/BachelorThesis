import tensorflow as tf
import numpy as np
import os
from PIL import Image

# --- CONFIG ---
MODEL_PATH = "Simple_savedmodel"
TFLITE_OUTPUT_PATH = "Simple_quant.tflite"
BASE_DIR = r"C:\Users\andre\Documents\BA\Dev\Data\38-Cloud_training"
MAX_REP_IMAGES = 100
PATCH_SIZE = (192, 192)

# --- Representative Dataset Generator ---
def representative_data_gen(red_paths, base_dir, size=(192, 192)):
    for red_path in red_paths[:MAX_REP_IMAGES]:
        name = os.path.basename(red_path).replace("red_", "")
        try:
            r = np.array(Image.open(red_path).resize(size), dtype=np.uint16)
            g = np.array(Image.open(os.path.join(base_dir, "train_green", f"green_{name}")).resize(size), dtype=np.uint16)
            b = np.array(Image.open(os.path.join(base_dir, "train_blue", f"blue_{name}")).resize(size), dtype=np.uint16)
            n = np.array(Image.open(os.path.join(base_dir, "train_nir", f"nir_{name}")).resize(size), dtype=np.uint16)

            patch = np.stack([r, g, b, n], axis=-1).astype(np.float32) / 65535.0
            patch = np.expand_dims(patch, axis=0)
            yield [patch]
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            continue

# --- Convert Model ---
def convert_model_to_tflite(model_path, output_path, red_paths, base_dir):

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(red_paths, base_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    #converter.experimental_new_converter = False
    #converter._experimental_lower_tensor_list_ops = True
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]



    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"\n✅ Quantized model saved to: {output_path}")

# --- Main ---
if __name__ == "__main__":
    red_dir = os.path.join(BASE_DIR, "train_red")
    red_paths = [os.path.join(red_dir, f) for f in os.listdir(red_dir) if f.endswith(".TIF")]

    convert_model_to_tflite(MODEL_PATH, TFLITE_OUTPUT_PATH, red_paths, BASE_DIR)
