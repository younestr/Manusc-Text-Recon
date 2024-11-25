import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.backend import ctc_decode

# Path configurations
model_path = r"C:\Users\hp\Downloads\Manusc Text Recon\new_model\C_LSTM_best_c1.hdf5"
charlist_path = r"C:\Users\hp\Downloads\Manusc Text Recon\pretrained_model\charList.txt"
test_images_path = r"C:\Users\hp\Downloads\Manusc Text Recon\test_images"

# Custom CTCLayer definition
class CTCLayer(Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, labels, predictions):
        label_length = tf.math.reduce_sum(tf.cast(labels != -1, tf.int32), axis=-1)
        pred_length = tf.fill([tf.shape(predictions)[0]], tf.shape(predictions)[1])
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=predictions,
            label_length=label_length,
            logit_length=pred_length,
            blank_index=-1
        )
        return tf.reduce_mean(loss)

# Load character list
with open(charlist_path, "r", encoding="utf-8") as f:
    char_list = f.read()

# Load the trained model
try:
    model = load_model(model_path, custom_objects={"CTCLayer": CTCLayer}, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Preprocessing function
def preprocess_image(image_path):
    """Resize and normalize the image."""
    img = load_img(image_path, target_size=(32, 128), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Decoding predictions
def decode_predictions(predictions):
    """Convert model predictions to readable text."""
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    results, _ = ctc_decode(predictions, input_length=input_len, greedy=True)
    output_texts = []
    for res in results:
        decoded = "".join([char_list[int(idx)] for idx in res.numpy()[0] if idx != -1])
        output_texts.append(decoded)
    return output_texts

# Process test images
test_images = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not test_images:
    print("No test images found in the specified directory.")
    exit()

for img_path in test_images:
    try:
        img = preprocess_image(img_path)
        preds = model.predict(img)
        decoded_text = decode_predictions(preds)
        print(f"Image: {os.path.basename(img_path)} | Predicted Text: {decoded_text[0]}")
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
