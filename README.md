# ViT V2 ONNX Facial Emotion Detection

Real-time facial emotion detection using a fine-tuned Vision Transformer (ViT V2) model exported to ONNX format.

The system detects ONE FACE AT A TIME from a webcam and classifies the facial expression into one of seven emotions.

============================================================
EMOTIONS
============================================================

0 → Angry
1 → Disgust
2 → Fear
3 → Happy
4 → Neutral
5 → Sad
6 → Surprise


============================================================
PROJECT STRUCTURE
============================================================

vit-emotion-v2-deployment/
│
├── emotion_config.json
├── realtime_inference_onnx.py
├── vit-emotion-v2.onnx
├── vit-emotion-v2.onnx.data
└── README.md


============================================================
IMPORTANT
============================================================

The following two files are part of the same ONNX model and
MUST remain together in the same folder:

vit-emotion-v2.onnx
vit-emotion-v2.onnx.data

Do NOT rename, delete, or separate the .onnx.data file.


============================================================
REQUIREMENTS
============================================================

• Python 3.10 or 3.11 recommended
• Webcam
• Windows, Linux, or macOS
• CPU is sufficient
• NVIDIA GPU is NOT required

The deployment uses ONNX Runtime for inference.


============================================================
SETUP
============================================================

1. CLONE THE REPOSITORY
------------------------------------------------------------

git clone https://github.com/Piyushchy/Fyp-project.git

cd Fyp-project/vit-emotion-v2-deployment


2. CREATE A VIRTUAL ENVIRONMENT
------------------------------------------------------------

WINDOWS:

python -m venv .venv

.venv\Scripts\activate


LINUX / macOS:

python3 -m venv .venv

source .venv/bin/activate


3. INSTALL DEPENDENCIES
------------------------------------------------------------

pip install numpy opencv-python onnxruntime


4. VERIFY INSTALLATION
------------------------------------------------------------

python -c "import cv2,numpy,onnxruntime; print('Setup successful')"

Expected output:

Setup successful


============================================================
RUN THE APPLICATION
============================================================

Make sure the virtual environment is activated.

Run:

python realtime_inference_onnx.py


The application will:

1. Open the webcam.
2. Detect a face.
3. Crop the detected face.
4. Resize it to 224 × 224.
5. Preprocess the image.
6. Run the ViT V2 ONNX model.
7. Predict the facial emotion.
8. Display the emotion and confidence on the video.

Press Q to close the application.


============================================================
PROCESSING PIPELINE
============================================================

Webcam
   ↓
Video Frame
   ↓
OpenCV Face Detection
   ↓
Face Crop
   ↓
RGB Conversion
   ↓
Resize to 224 × 224
   ↓
Normalization
   ↓
ViT V2 ONNX Model
   ↓
Emotion Prediction
   ↓
Emotion + Confidence


============================================================
MODEL INFORMATION
============================================================

INPUT
------------------------------------------------------------

Name:

pixel_values

Shape:

[batch_size, 3, 224, 224]

Type:

float32


OUTPUT
------------------------------------------------------------

Name:

logits

Shape:

[batch_size, 7]

Type:

float32


CLASS MAPPING
------------------------------------------------------------

0 → angry
1 → disgust
2 → fear
3 → happy
4 → neutral
5 → sad
6 → surprise


============================================================
IMAGE PREPROCESSING
============================================================

The detected face is processed before being passed to the
ViT V2 ONNX model.

Processing steps:

1. Detect face
2. Crop face
3. Convert BGR to RGB
4. Resize to 224 × 224
5. Convert image to float32
6. Normalize pixel values
7. Convert image from HWC to CHW
8. Add batch dimension
9. Pass tensor to ONNX model


============================================================
FACE DETECTION
============================================================

The current implementation uses OpenCV's Haar Cascade
face detector.

The face detector is responsible for locating the face.

The ViT V2 model is responsible for classifying the
facial expression.

Only ONE FACE is processed at a time.


============================================================
REAL-TIME PROCESSING
============================================================

The application can skip frames to improve real-time
performance.

The following setting controls how frequently emotion
classification is performed:

PROCESS_EVERY_N_FRAMES = 2


Examples:

PROCESS_EVERY_N_FRAMES = 1

→ Process every frame.


PROCESS_EVERY_N_FRAMES = 2

→ Process every second frame.


PROCESS_EVERY_N_FRAMES = 3

→ Process every third frame.


A higher value can improve performance on slower systems,
but the displayed emotion will update less frequently.


============================================================
CONFIDENCE SCORE
============================================================

The application displays the probability of the predicted
emotion.

Example:

happy (87.4%)

This means the model assigned approximately 87.4%
probability to the happy class.

The confidence score should NOT be interpreted as a
guarantee that a person is experiencing that emotion.

The system estimates emotion from visible facial features.


============================================================
MODEL PERFORMANCE
============================================================

Current ViT V2 evaluation results:

Accuracy        : 67.69%
Macro Precision : 65.30%
Macro Recall    : 68.04%
Macro F1        : 66.46%


PER-EMOTION RESULTS
------------------------------------------------------------

Emotion      Precision    Recall    F1-Score

Angry        57.92%       59.49%    58.69%
Disgust      61.15%       77.98%    68.55%
Fear         55.24%       49.41%    52.16%
Happy        88.42%       84.59%    86.46%
Neutral      62.06%       64.16%    63.10%
Sad          56.93%       56.79%    56.86%
Surprise     75.39%       83.87%    79.41%


The model performs particularly well for:

• Happy
• Surprise

Performance is comparatively lower for:

• Fear
• Sad
• Angry


============================================================
WHY ONNX?
============================================================

The original model was trained using a Vision Transformer.

For deployment, the trained ViT model was exported to ONNX.

The deployment pipeline is:

Fine-Tuned ViT
      ↓
ONNX Export
      ↓
vit-emotion-v2.onnx
      ↓
ONNX Runtime
      ↓
CPU Inference
      ↓
Real-Time Emotion Detection


ONNX Runtime allows the trained model to be used without
requiring the complete training environment.

This makes deployment lightweight and suitable for
CPU-based inference.


============================================================
WHY CPU INSTEAD OF NVIDIA GPU?
============================================================

The deployment is designed to work on systems without a
dedicated NVIDIA GPU.

ONNX Runtime performs the inference on the CPU by default.

The model is relatively small and the application processes
one detected face at a time, making CPU-based deployment
practical for real-time use.

A compatible GPU execution provider can be considered in
the future if additional performance is required.


============================================================
TROUBLESHOOTING
============================================================

PROBLEM: ONNX Runtime is missing
------------------------------------------------------------

Error:

ModuleNotFoundError: No module named 'onnxruntime'

Solution:

pip install onnxruntime


PROBLEM: OpenCV is missing
------------------------------------------------------------

Error:

ModuleNotFoundError: No module named 'cv2'

Solution:

pip install opencv-python


PROBLEM: NumPy is missing
------------------------------------------------------------

Error:

ModuleNotFoundError: No module named 'numpy'

Solution:

pip install numpy


PROBLEM: CascadeClassifier error
------------------------------------------------------------

Error:

AttributeError: module 'cv2' has no attribute
'CascadeClassifier'

Check OpenCV:

python -c "import cv2; print(cv2.__version__); print(hasattr(cv2,'CascadeClassifier'))"

If necessary, reinstall OpenCV:

pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y

pip install opencv-python==4.10.0.84


PROBLEM: Webcam does not open
------------------------------------------------------------

If the program reports:

Could not open webcam

Check:

• Webcam is connected.
• No other application is using the webcam.
• Camera permissions are enabled.
• Python/VS Code has camera access.

The default camera is:

cv2.VideoCapture(0)

If multiple cameras are available, try:

cv2.VideoCapture(1)

or:

cv2.VideoCapture(2)


PROBLEM: ONNX MODEL CANNOT BE LOADED
------------------------------------------------------------

Make sure these files are together:

vit-emotion-v2.onnx
vit-emotion-v2.onnx.data

They must be located in the same directory.


============================================================
INACCURATE PREDICTIONS
============================================================

Incorrect predictions can occur because of:

• Poor lighting
• Face angle
• Facial occlusion
• Camera quality
• Weak facial expressions
• Similar-looking emotions
• Dataset limitations
• Class imbalance
• Differences between training images and real-world
  webcam images

For better results:

• Face the camera directly.
• Use sufficient lighting.
• Keep the face reasonably close to the camera.
• Avoid covering the face.
• Avoid extreme head angles.
• Make a clear facial expression.

The model should be evaluated using test-set metrics rather
than relying only on individual webcam predictions.


============================================================
TRAINING VS DEPLOYMENT
============================================================

TRAINING
------------------------------------------------------------

Training was performed separately using the facial emotion
dataset and a Vision Transformer.

Training pipeline:

Facial Emotion Dataset
        ↓
Data Preprocessing
        ↓
ViT Fine-Tuning
        ↓
Fine-Tuned ViT Model
        ↓
Model Evaluation


DEPLOYMENT
------------------------------------------------------------

The trained model was exported to ONNX for deployment.

Deployment pipeline:

Fine-Tuned ViT
        ↓
ONNX Export
        ↓
ONNX Model
        ↓
ONNX Runtime
        ↓
Webcam
        ↓
Real-Time Emotion Detection


The original training dataset is NOT required to run this
deployment project.


============================================================
MULTIMODAL PROJECT INTEGRATION
============================================================

This ViT V2 model is the VISUAL EMOTION DETECTION component
of the larger multimodal emotion recognition system.

The overall system can contain three modalities:

                    MULTIMODAL SYSTEM
                           |
          +----------------+----------------+
          |                |                |
          ↓                ↓                ↓
        IMAGE            AUDIO            TEXT
          |                |                |
          ↓                ↓                ↓
       ViT V2          Audio Model       Text Model
          |                |                |
          ↓                ↓                ↓
   Visual Emotion     Audio Emotion     Text Emotion
          |                |                |
          +----------------+----------------+
                           |
                           ↓
                    Fusion Module
                           |
                           ↓
                     Final Emotion


The ViT V2 ONNX model therefore acts as the:

VISUAL EMOTION DETECTION MODULE


============================================================
DEVELOPMENT WORKFLOW
============================================================

For a new developer:

1. Clone the repository.
2. Open the vit-emotion-v2-deployment folder.
3. Create a virtual environment.
4. Activate the virtual environment.
5. Install the required dependencies.
6. Verify the ONNX model files.
7. Run realtime_inference_onnx.py.
8. Test the webcam.
9. Modify the code if required.
10. Test again before committing changes.


============================================================
GIT WORKFLOW
============================================================

After making changes:

git status

Add the modified files:

git add .

Commit the changes:

git commit -m "Update ViT V2 ONNX deployment"

Push the changes:

git push origin main


============================================================
IMPORTANT FOR CONTRIBUTORS
============================================================

Do NOT upload:

• Training datasets
• Kaggle temporary files
• Python virtual environments
• __pycache__ folders
• Personal videos
• Personal images
• Temporary model checkpoints

The deployment folder should contain only the files required
to run the ONNX inference application.


============================================================
CURRENT DEPLOYMENT FILES
============================================================

emotion_config.json
    → Emotion labels/configuration

realtime_inference_onnx.py
    → Main real-time inference program

vit-emotion-v2.onnx
    → Main ONNX model

vit-emotion-v2.onnx.data
    → Additional ONNX model data

README.md
    → Project documentation


============================================================
END
============================================================
