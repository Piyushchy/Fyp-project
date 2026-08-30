import cv2
import numpy as np
import onnxruntime as ort
import json
import time


# ============================================================
# CONFIGURATION
# ============================================================

ONNX_PATH="vit-emotion-v2.onnx"
CONFIG_PATH="emotion_config.json"

IMAGE_SIZE=224

MEAN=np.array(
    [0.5,0.5,0.5],
    dtype=np.float32
)

STD=np.array(
    [0.5,0.5,0.5],
    dtype=np.float32
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r"
) as f:

    config=json.load(f)


ID2LABEL={
    int(key):value
    for key,value in config["id2label"].items()
}


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print("Loading ONNX model...")

session=ort.InferenceSession(
    ONNX_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)

input_name=session.get_inputs()[0].name

print("ONNX model loaded successfully.")

print("\nEmotion classes:")

for key,value in ID2LABEL.items():

    print(
        key,
        "->",
        value
    )


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():

    raise RuntimeError(
        "Could not load face detector."
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(face):

    face_rgb=cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face_resized=cv2.resize(
        face_rgb,
        (IMAGE_SIZE,IMAGE_SIZE)
    )

    image=face_resized.astype(
        np.float32
    )/255.0

    image=(
        image-MEAN
    )/STD

    image=image.transpose(
        2,
        0,
        1
    )

    image=np.expand_dims(
        image,
        axis=0
    )

    return image.astype(
        np.float32
    )


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits=logits-np.max(
        logits
    )

    probabilities=np.exp(
        logits
    )

    probabilities/=np.sum(
        probabilities
    )

    return probabilities


# ============================================================
# WEBCAM
# ============================================================

print("\nOpening webcam...")

cap=cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# VARIABLES
# ============================================================

last_label="Detecting..."
last_confidence=0.0

previous_time=time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret,frame=cap.read()

    if not ret:

        break


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    gray=cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces=face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80,80)
    )


    # --------------------------------------------------------
    # ONE FACE ONLY
    # --------------------------------------------------------

    if len(faces)>0:

        x,y,w,h=max(
            faces,
            key=lambda face:face[2]*face[3]
        )


        # ----------------------------------------------------
        # SMALL FACE MARGIN
        # ----------------------------------------------------

        margin_x=int(
            w*0.05
        )

        margin_y=int(
            h*0.05
        )

        x1=max(
            0,
            x-margin_x
        )

        y1=max(
            0,
            y-margin_y
        )

        x2=min(
            frame.shape[1],
            x+w+margin_x
        )

        y2=min(
            frame.shape[0],
            y+h+margin_y
        )


        face_roi=frame[
            y1:y2,
            x1:x2
        ]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        input_tensor=preprocess(
            face_roi
        )

        logits=session.run(
            None,
            {
                input_name:
                input_tensor
            }
        )[0][0]


        probabilities=softmax(
            logits
        )

        prediction=int(
            np.argmax(
                probabilities
            )
        )

        confidence=float(
            probabilities[
                prediction
            ]
        )


        last_label=ID2LABEL[
            prediction
        ]

        last_confidence=confidence


        # ----------------------------------------------------
        # DRAW FACE
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        # ----------------------------------------------------
        # DRAW EMOTION
        # ----------------------------------------------------

        text=(
            f"{last_label} "
            f"({last_confidence*100:.1f}%)"
        )


        cv2.putText(
            frame,
            text,
            (x1,max(30,y1-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )


    else:

        cv2.putText(
            frame,
            "No face detected",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time=time.time()

    fps=1.0/(
        current_time-previous_time
    )

    previous_time=current_time


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,0,0),
        2
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        "Real-Time Facial Emotion Detection",
        frame
    )


    # Press Q to quit

    if cv2.waitKey(1)&0xFF==ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Webcam closed.")