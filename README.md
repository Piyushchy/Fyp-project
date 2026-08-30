# ViT V2 ONNX Facial Emotion Detection

Real-time facial emotion detection using a fine-tuned Vision Transformer (ViT V2) model exported to ONNX format.

The system detects **one face at a time** from a webcam and classifies the facial expression into one of seven emotions.

## Emotions

| Label | Emotion  |
|-------|----------|
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Neutral  |
| 5     | Sad      |
| 6     | Surprise |

## Project Structure

```
vit-emotion-v2-deployment/
│
├── emotion_config.json
├── realtime_inference_onnx.py
├── vit-emotion-v2.onnx
├── vit-emotion-v2.onnx.data
└── README.md
```

> **Important:** `vit-emotion-v2.onnx` and `vit-emotion-v2.onnx.data` are part of the same ONNX model and **must remain together** in the same folder. Do not rename, delete, or separate the `.onnx.data` file.

## Requirements

- Python 3.10 or 3.11 recommended
- Webcam
- Windows, Linux, or macOS
- CPU is sufficient — NVIDIA GPU is **not** required
- Inference runs via ONNX Runtime

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Piyushchy/Fyp-project.git
cd Fyp-project/vit-emotion-v2-deployment
```

### 2. Create a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy opencv-python onnxruntime
```

### 4. Verify installation

```bash
python -c "import cv2,numpy,onnxruntime; print('Setup successful')"
```

Expected output:
```
Setup successful
```

## Run the Application

Make sure the virtual environment is activated, then run:

```bash
python realtime_inference_onnx.py
```

The application will:

1. Open the webcam
2. Detect a face
3. Crop the detected face
4. Resize it to 224 × 224
5. Preprocess the image
6. Run the ViT V2 ONNX model
7. Predict the facial emotion
8. Display the emotion and confidence on the video

Press **Q** to close the application.

## Processing Pipeline

```
Webcam
  → Video Frame
  → OpenCV Face Detection
  → Face Crop
  → RGB Conversion
  → Resize to 224 × 224
  → Normalization
  → ViT V2 ONNX Model
  → Emotion Prediction
  → Emotion + Confidence
```

## Model Information

**Input**
- Name: `pixel_values`
- Shape: `[batch_size, 3, 224, 224]`
- Type: `float32`

**Output**
- Name: `logits`
- Shape: `[batch_size, 7]`
- Type: `float32`

**Class Mapping**

```
0 → angry
1 → disgust
2 → fear
3 → happy
4 → neutral
5 → sad
6 → surprise
```

## Image Preprocessing

1. Detect face
2. Crop face
3. Convert BGR to RGB
4. Resize to 224 × 224
5. Convert image to float32
6. Normalize pixel values
7. Convert image from HWC to CHW
8. Add batch dimension
9. Pass tensor to ONNX model

## Face Detection

The current implementation uses OpenCV's **Haar Cascade** face detector to locate the face. The ViT V2 model then classifies the facial expression. Only **one face** is processed at a time.

## Real-Time Processing

The application can skip frames to improve real-time performance:

```python
PROCESS_EVERY_N_FRAMES = 2
```

| Value | Behavior                    |
|-------|------------------------------|
| 1     | Process every frame          |
| 2     | Process every second frame   |
| 3     | Process every third frame    |

A higher value can improve performance on slower systems, but the displayed emotion will update less frequently.

## Confidence Score

The application displays the probability of the predicted emotion, e.g.:

```
happy (87.4%)
```

This means the model assigned ~87.4% probability to the "happy" class. The confidence score should **not** be interpreted as a guarantee that a person is experiencing that emotion — the system estimates emotion from visible facial features only.

## Model Performance

**Overall (ViT V2 evaluation results)**

| Metric          | Score  |
|-----------------|--------|
| Accuracy        | 67.69% |
| Macro Precision | 65.30% |
| Macro Recall    | 68.04% |
| Macro F1        | 66.46% |

**Per-Emotion Results**

| Emotion  | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Angry    | 57.92%    | 59.49% | 58.69%   |
| Disgust  | 61.15%    | 77.98% | 68.55%   |
| Fear     | 55.24%    | 49.41% | 52.16%   |
| Happy    | 88.42%    | 84.59% | 86.46%   |
| Neutral  | 62.06%    | 64.16% | 63.10%   |
| Sad      | 56.93%    | 56.79% | 56.86%   |
| Surprise | 75.39%    | 83.87% | 79.41%   |

The model performs particularly well for **Happy** and **Surprise**, and comparatively lower for **Fear**, **Sad**, and **Angry**.

## Why ONNX?

The original model was trained as a Vision Transformer and exported to ONNX for deployment:

```
Fine-Tuned ViT → ONNX Export → vit-emotion-v2.onnx → ONNX Runtime → CPU Inference → Real-Time Emotion Detection
```

ONNX Runtime allows the trained model to run without the full training environment, keeping deployment lightweight and suitable for CPU-based inference.

## Why CPU Instead of NVIDIA GPU?

This deployment targets systems without a dedicated NVIDIA GPU. ONNX Runtime performs inference on the CPU by default. The model is relatively small, and only one face is processed at a time — making CPU-based deployment practical for real-time use. A GPU execution provider could be added later if more performance is needed.

## Troubleshooting

**ONNX Runtime is missing**
```
ModuleNotFoundError: No module named 'onnxruntime'
```
Fix: `pip install onnxruntime`

**OpenCV is missing**
```
ModuleNotFoundError: No module named 'cv2'
```
Fix: `pip install opencv-python`

**NumPy is missing**
```
ModuleNotFoundError: No module named 'numpy'
```
Fix: `pip install numpy`

**CascadeClassifier error**
```
AttributeError: module 'cv2' has no attribute 'CascadeClassifier'
```
Check your OpenCV install:
```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2,'CascadeClassifier'))"
```
If necessary, reinstall OpenCV:
```bash
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python==4.10.0.84
```

**Webcam does not open**
```
Could not open webcam
```
Check that:
- The webcam is connected
- No other application is using the webcam
- Camera permissions are enabled
- Python/VS Code has camera access

The default camera is `cv2.VideoCapture(0)`. If multiple cameras are available, try `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`.

**ONNX model cannot be loaded**

Make sure `vit-emotion-v2.onnx` and `vit-emotion-v2.onnx.data` are located in the same directory.

## Inaccurate Predictions

Incorrect predictions can occur because of:

- Poor lighting
- Face angle
- Facial occlusion
- Camera quality
- Weak facial expressions
- Similar-looking emotions
- Dataset limitations
- Class imbalance
- Differences between training images and real-world webcam images

For better results:

- Face the camera directly
- Use sufficient lighting
- Keep the face reasonably close to the camera
- Avoid covering the face
- Avoid extreme head angles
- Make a clear facial expression

The model should be evaluated using test-set metrics rather than relying only on individual webcam predictions.

## Training vs. Deployment

**Training** (performed separately)
```
Facial Emotion Dataset → Data Preprocessing → ViT Fine-Tuning → Fine-Tuned ViT Model → Model Evaluation
```

**Deployment**
```
Fine-Tuned ViT → ONNX Export → ONNX Model → ONNX Runtime → Webcam → Real-Time Emotion Detection
```

The original training dataset is **not** required to run this deployment project.

## Multimodal Project Integration

This ViT V2 model is the **visual emotion detection** component of a larger multimodal emotion recognition system:

```
                    Multimodal System
                           |
          +----------------+----------------+
          |                |                |
        Image            Audio            Text
          |                |                |
       ViT V2          Audio Model       Text Model
          |                |                |
   Visual Emotion     Audio Emotion     Text Emotion
          |                |                |
          +----------------+----------------+
                           |
                     Fusion Module
                           |
                     Final Emotion
```

The ViT V2 ONNX model acts as the **visual emotion detection module**.

## Development Workflow

1. Clone the repository
2. Open the `vit-emotion-v2-deployment` folder
3. Create a virtual environment
4. Activate the virtual environment
5. Install the required dependencies
6. Verify the ONNX model files
7. Run `realtime_inference_onnx.py`
8. Test the webcam
9. Modify the code if required
10. Test again before committing changes

## Git Workflow

```bash
git status
git add .
git commit -m "Update ViT V2 ONNX deployment"
git push origin main
```

## Important for Contributors

Do **not** upload:

- Training datasets
- Kaggle temporary files
- Python virtual environments
- `__pycache__` folders
- Personal videos
- Personal images
- Temporary model checkpoints

The deployment folder should contain only the files required to run the ONNX inference application.

## Current Deployment Files

| File                        | Purpose                          |
|-----------------------------|-----------------------------------|
| `emotion_config.json`       | Emotion labels/configuration      |
| `realtime_inference_onnx.py`| Main real-time inference program  |
| `vit-emotion-v2.onnx`       | Main ONNX model                   |
| `vit-emotion-v2.onnx.data`  | Additional ONNX model data        |
| `README.md`                 | Project documentation             |
