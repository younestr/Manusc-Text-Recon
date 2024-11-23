import os
import cv2
import numpy as np
import tensorflow as tf
from model import Model
from preprocessor import Preprocessor


def load_images_from_folder(folder_path, img_size):
    """
    Load and preprocess images from a folder.
    """
    preprocessor = Preprocessor(img_size, data_augmentation=False)
    images = []
    file_paths = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            processed_img = preprocessor.process_img(img)
            images.append(processed_img)
            file_paths.append(file_path)
    return np.array(images), file_paths


def predict(model, sess, images, char_list):
    """
    Run predictions on preprocessed images using the model.
    """
    predictions = []
    for img in images:
        img = img[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        feed_dict = {model.input_images: img, model.is_training: False}
        logits = sess.run(model.output, feed_dict=feed_dict)
        
        # CTC decoding
        decoded, _ = tf.nn.ctc_greedy_decoder(logits, [img.shape[2]])
        decoded_indices = sess.run(decoded[0].indices)[:, 1]  # Extract the character indices
        pred_text = ''.join([char_list[idx] for idx in decoded_indices])
        predictions.append(pred_text)

    return predictions


def remap_variables(var_list):
    """
    Remap the checkpoint variables to match the expected names.
    """
    restored_variables = {}
    for var in var_list:
        # Remap batch normalization variable names
        if 'batch_normalization_1_1/beta' in var.name:
            restored_variables['batch_normalization_1/beta'] = var
        elif 'batch_normalization_1_1/gamma' in var.name:
            restored_variables['batch_normalization_1/gamma'] = var
        elif 'batch_normalization_1_1/moving_mean' in var.name:
            restored_variables['batch_normalization_1/moving_mean'] = var
        elif 'batch_normalization_1_1/moving_variance' in var.name:
            restored_variables['batch_normalization_1/moving_variance'] = var
        else:
            restored_variables[var.name] = var

    return restored_variables


def main():
    # Paths
    model_dir = r'C:\Users\hp\Downloads\Manusc Text Recon\pretrained_model'
    test_images_dir = r'C:\Users\hp\Downloads\Manusc Text Recon\test_images'
    char_list_path = os.path.join(model_dir, "charList.txt")
    snapshot_path = os.path.join(model_dir, "snapshot-33")

    # Model and preprocessor parameters
    img_size = (128, 32)

    # Load character list
    with open(char_list_path, 'r') as f:
        char_list = f.read().strip()

    # Load images
    print("Loading and preprocessing images...")
    images, file_paths = load_images_from_folder(test_images_dir, img_size)

    # Load model
    print("Loading model...")
    model = Model(char_list=char_list)
    tf.compat.v1.disable_eager_execution()  # Disable eager execution for compatibility
    sess = tf.compat.v1.Session()

    # Remap variables during restore
    restored_variables = remap_variables(tf.compat.v1.global_variables())
    saver = tf.compat.v1.train.Saver(restored_variables)
    saver.restore(sess, snapshot_path)

    # Run predictions
    print("Running predictions...")
    predictions = predict(model, sess, images, char_list)

    # Print results
    for file_path, prediction in zip(file_paths, predictions):
        print(f"Image: {file_path}, Predicted Text: {prediction}")

    # Close session
    sess.close()


if __name__ == "__main__":
    main()
