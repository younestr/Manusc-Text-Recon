# Handwritten Text Recognition with TensorFlow

Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset. The model takes **images of single words or text lines (multiple words) as input** and **outputs the recognized text**. 3/4 of the words from the validation-set are correctly recognized, and the character error rate is around 10%.

![htr](./doc/htr.png)

## Difference Between the Approaches

This project has evolved through three approaches:

### Old Model
- **Architecture**: Stripped-down version with 5 CNN layers, 2 RNN (LSTM) layers, and a CTC loss/decoding layer.
- **Performance**: Moderate accuracy; performs well on single-word recognition but struggles with text lines.
- **Focus**: Suitable for recognizing isolated word images.

### New Model
- **Architecture**: Enhanced CNN and RNN configurations, improved training data augmentation techniques, and optimized learning rate.
- **Performance**: Better text line recognition; increased robustness with slightly lower character error rates.
- **Focus**: More generalized performance for both single words and text lines.

### Third Approach (In Progress)
- **Focus**: Current work emphasizes addressing limitations in long-text recognition and integrating a multilingual corpus for broader applicability.
- **Expected Outcome**: Improved generalization, reduced error rates, and support for diverse datasets.

## Run Demo

1. Download one of the pretrained models:
   - [Model trained on word images](https://www.dropbox.com/s/mya8hw6jyzqm0a3/word-model.zip?dl=1): Handles single words with better accuracy on the IAM dataset.
   - [Model trained on text line images](https://www.dropbox.com/s/7xwkcilho10rthn/line-model.zip?dl=1): Can process multiple words in one image.
2. Extract the downloaded zip file into the `model` directory of the repository.
3. Navigate to the `src` directory.
4. Run the inference code:
   - `python main.py`: Recognizes text from a word image.
   - `python main.py --img_file ../data/line.png`: Recognizes text from a text line image.

**Example Outputs**:

For a word image:
![word](./data/word.png)
