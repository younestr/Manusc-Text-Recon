import tensorflow as tf
import numpy as np
import os
import cv2
from preprocessor import Preprocessor
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, LSTM, Bidirectional, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# Function to load char list from file
def load_char_list(file_path):
    with open(file_path, 'r') as file:
        char_list = file.read().strip()  # Read and remove any extra spaces or newline characters
    return list(char_list)

# Define the model (make sure to use the exact same architecture you used during training)
def create_model(char_list):
    inputs = Input(shape=(32, 128, 1), name="image")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    conv_1 = Conv2D(32, (3,3), activation="selu", padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)

    conv_2 = Conv2D(64, (3,3), activation="selu", padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)

    conv_3 = Conv2D(128, (3,3), activation="selu", padding='same')(pool_2)
    conv_4 = Conv2D(128, (3,3), activation="selu", padding='same')(conv_3)

    conv_5 = Conv2D(512, (3,3), activation="selu", padding='same')(conv_4)
    conv_6 = Conv2D(512, (3,3), activation="selu", padding='same')(conv_5)
    drop_out = tf.keras.layers.Dropout(0.2)(conv_6)
    conv_7 = Conv2D(512, (3,3), activation="selu", padding='same')(drop_out)
    conv_8 = Conv2D(512, (3,3), activation="selu", padding='same')(conv_7)

    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_8)

    conv_5 = Conv2D(256, (3,3), activation="selu", padding='same')(pool_4)

    batch_norm_5 = BatchNormalization()(conv_5)
    conv_6 = Conv2D(256, (3,3), activation="selu", padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    conv_7 = Conv2D(64, (2,2), activation="selu")(pool_6)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # Replacing CuDNNLSTM with LSTM
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True))(squeezed)
    blstm_2 = Bidirectional(LSTM(512, return_sequences=True))(blstm_1)
    blstm_3 = Bidirectional(LSTM(512, return_sequences=True))(blstm_2)
    blstm_4 = Bidirectional(LSTM(512, return_sequences=True))(blstm_3)
    blstm_5 = Bidirectional(LSTM(128, return_sequences=True))(blstm_4)

    dense_ = Dense(128, activation='relu', name='dense_1')(blstm_5)

    # Use len(char_list) + 1 for the output layer (the +1 is for the blank character in CTC loss)
    softmax_output = Dense(len(char_list), activation='softmax', name="output_dense")(dense_)

    model = Model(inputs=[inputs, labels], outputs=softmax_output)
    return model

# Path to your char list file
char_list_path = r"C:\Users\hp\Downloads\Manusc Text Recon\pretrained_model\charList.txt"

# Load the character list from file
char_list = load_char_list(char_list_path)
print(len(char_list))

# Load the trained model and weights
model = create_model(char_list)
model.load_weights(r'C:\Users\hp\Downloads\Manusc Text Recon\new_model\C_LSTM_best_c1.hdf5')

# Initialize the preprocessor (make sure the dimensions match your model's expected input)
preprocessor = Preprocessor(img_size=(128, 32), data_augmentation=False)

# Path to test images folder
test_images_path = r"C:\Users\hp\Downloads\Manusc Text Recon\test_images"

# Get a list of test images (assuming they are in .png or .jpg format)
test_images = [f for f in os.listdir(test_images_path) if f.endswith(('.png', '.jpg'))]

# CTC Decoder function to decode the output probabilities into text
def decode_ctc(predictions):
    decoded_text = tf.strings.reduce_join(tf.argmax(predictions, axis=-1), axis=-1).numpy()
    return decoded_text

# Loop over the test images and make predictions
for test_image_name in test_images:
    # Full path to the test image
    test_image_path = os.path.join(test_images_path, test_image_name)
    
    # Read the test image (grayscale)
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error loading image: {test_image_name}")
        continue
    
    # Preprocess the image
    processed_img = preprocessor.process_img(img)

    # Expand dimensions to include batch dimension
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(processed_img)

    # Decode the predictions (using the CTC decoder)
    decoded_predictions = decode_ctc(predictions)

    # Print the predicted text
    print(f"Predicted Text for {test_image_name}: {decoded_predictions}")
