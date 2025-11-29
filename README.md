# Facial Expression Recognition (Mini-Xception, TensorFlow)

This project implements a complete Facial Expression Recognition (FER) pipeline using a custom Mini-Xception neural network.  
It includes dataset loading, preprocessing, augmentation, training, evaluation and real-time webcam inference using OpenCV.

## 1. Requirements

Install dependencies:

```
pip install -r requirements.txt
```

requirements.txt:
```
tensorflow==2.17.0
numpy
pandas
tqdm
opencv-python
scikit-learn
matplotlib
```

Python 3.10+ is recommended.

## 2. Project Structure

```
project/
│
├── code/
│   ├── main.py                 # Training entry point
│   ├── trainer.py              # Dataset pipeline, augmentations, callbacks
│   ├── eval.py                 # Evaluation utilities
│   ├── post_eval.py            # ROC curve and confusion matrix plotting
│   ├── infer_cam.py            # Realtime webcam inference
│   ├── model_mini_xception.py  # Mini-Xception model architecture
│
├── images/
│   ├── train/                  # Training images (class subfolders)
│   ├── test/                   # Test images (class subfolders)
│
├── artifacts/
│   ├── mini-xception/
│   │   ├── best.keras
│   │   ├── last.keras
│   ├── labels.json
│
└── README.md
```

Dataset format:

```
images/train/
    angry/
    happy/
    sad/
    surprise/
    fear/
    neutral/
```

## 3. Training the Model

Run inside the code/ directory:

```
python main.py
```

The script performs the following steps:

1. Loads dataset into a DataFrame.
2. Removes unwanted classes (e.g., disgust).
3. Encodes string labels into indices.
4. Loads and normalizes all images.
5. Creates training/validation datasets with augmentation.
6. Builds the Mini-Xception model.
7. Trains the model using:
   - class weights  
   - warmup learning-rate scheduler  
   - early stopping  
8. Saves:
   - best.keras  
   - last.keras  
   - labels.json  

All artifacts are stored in:

```
artifacts/mini-xception/
```

## 4. Model Evaluation (Confusion Matrix and ROC)

To evaluate the trained model on the test dataset:

```
python post_eval.py
```

This script:

- loads the test dataset  
- performs inference  
- generates a confusion matrix  
- generates one-vs-rest ROC curves  
- displays all metrics using matplotlib  

## 5. Console Evaluation (Classification Report)

If you want textual metrics in the terminal:

```
python eval.py
```

It prints:

- classification report  
- confusion matrix  
- prediction outputs  

## 6. Real-Time Webcam Inference

Run:

```
python infer_cam.py
```

The script:

- loads the HaarCascade face detector  
- loads the best trained model  
- captures webcam frames  
- detects faces  
- preprocesses face crops  
- predicts facial expression  
- overlays label and confidence  

Quit by pressing the q key.

Troubleshooting video:

- If webcam cannot open:
  - try camera index 1 or 2  
  - Windows: CAP_DSHOW  
  - Linux: CAP_V4L2  
  - ensure no other application is using the camera  

## 7. Model Overview

The Mini-Xception model includes:

- Separable convolution blocks  
- Batch Normalization  
- PReLU activations  
- MaxPooling layers  
- GlobalAveragePooling  
- Dropout  
- Dense softmax classifier  

The architecture is inspired by the Xception network but optimized for lightweight FER tasks.

## 8. Common Issues

### TensorFlow GPU not detected
- verify nvidia-smi output  
- match TensorFlow version with the appropriate CUDA and cuDNN  
- on WSL2 ensure GPU acceleration is enabled  

### Training slow
Mixed precision is enabled in the project:

```
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

Works best on modern GPUs.

### Webcam not working
- ensure correct capture backend  
- close any program using the camera  

## 9. Exporting the Model

After training, models are stored in:

```
artifacts/mini-xception/best.keras
artifacts/mini-xception/last.keras
```

Example conversion to TFLite:

```python
import tensorflow as tf
model = tf.keras.models.load_model("best.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite = converter.convert()
```

 
