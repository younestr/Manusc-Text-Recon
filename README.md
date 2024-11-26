# Handwritten Text Recognition with TensorFlow

Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset. The model takes **images of single words or text lines (multiple words) as input** and **outputs the recognized text**. 3/4 of the words from the validation set are correctly recognized, and the character error rate is around 10%.

![htr](./doc/htr.png)

## Difference Between the Approaches

This project has evolved through three approaches:

### First Approach : Old Model
- **Architecture**: Stripped-down version with 5 CNN layers, 2 RNN (LSTM) layers, and a CTC loss/decoding layer.
- **Performance**: Moderate accuracy; performs well on single-word recognition but struggles with text lines.
- **Focus**: Suitable for recognizing isolated word images.

### Two Approach : New Model
- **Architecture**: Enhanced CNN and RNN configurations, improved training data augmentation techniques, and optimized learning rate.
- **Performance**: Better text line recognition; increased robustness with slightly lower character error rates.
- **Focus**: More generalized performance for both single words and text lines.

### Third Approach: Flask API with Pre-trained Models in Kaggle
This approach integrates the powerful **Flask framework** with **Kaggle's pre-trained models** to provide an API for handwritten text recognition. The architecture uses `ngrok` to expose the API endpoint publicly, processes requests in Kaggle, and serves the results via a local React.js frontend.

#### Architecture Overview
1. **Backend API with Flask**:
   - The Flask server runs in a Kaggle environment and handles incoming requests.
   - `ngrok` is used to tunnel the Flask server's local address to a public URL, enabling external access.

2. **Model for Text Recognition**:
   - The pre-trained model `microsoft/trocr-base-handwritten` is used.
   - **Loading the Model in Kaggle**:
     ```python
     # Load the pre-trained model and processor
     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
     ```
   - The `VisionEncoderDecoderModel` combines a vision-based encoder with a text-based decoder, trained specifically for handwritten text recognition.
   - **How It Works**:
     - The processor converts input images into pixel values and tokenizes the textual output.
     - The encoder processes image embeddings, and the decoder generates text sequences.
     - This design enables highly accurate transcription of handwritten text in a wide variety of scenarios.

3. **Frontend with React.js**:
   - The local React.js frontend provides a user interface for uploading handwritten text images.
   - It sends requests to the Flask API, processes responses, and displays the transcribed text.

4. **Workflow**:
   - Users upload handwritten images via the React.js frontend.
   - The images are sent to the Flask API endpoint exposed by `ngrok`.
   - The Kaggle environment processes the image using the pre-trained TrOCR model.
   - The transcription results are returned to the frontend for display.

#### Tools Used
- **Kaggle**: For hosting and running the backend server.
- **Flask**: As the backend framework for the API.
- **ngrok**: To expose the Flask API to a public URL.
- **React.js**: For the user interface.
- **Pre-trained Model**: `microsoft/trocr-base-handwritten`.

#### Advantages of This Approach
- **Pre-trained Models**: Leveraging state-of-the-art models reduces the need for extensive training.
- **API Integration**: Seamless communication between the backend and frontend for real-time results.
- **Scalable Design**: Can be extended to support additional features like multilingual recognition or custom datasets.

#### Expected Outcome
- Quick and accurate recognition of handwritten text.
- Easy-to-use interface for users to interact with the system.
- Flexible architecture for deployment in various scenarios.
