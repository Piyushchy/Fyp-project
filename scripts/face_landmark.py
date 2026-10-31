import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def calc_distance(p1, p2):
    """Calculates the Euclidean distance between two 2D points."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def heuristic_emotion(landmarks):
    """
    Calculates geometric ratios to determine the current emotion.
    Uses specific MediaPipe landmark indices for eyes, eyebrows, and mouth.
    """
    # Mouth Aspect Ratio (MAR): Inner top/bottom lip vs. left/right mouth corners
    mar = calc_distance(landmarks[13], landmarks[14]) / (calc_distance(landmarks[61], landmarks[291]) + 1e-6)
    
    # Eye Aspect Ratio (EAR): Top/bottom eyelids vs. inner/outer eye corners
    ear_left = calc_distance(landmarks[159], landmarks[145]) / (calc_distance(landmarks[33], landmarks[133]) + 1e-6)
    ear_right = calc_distance(landmarks[386], landmarks[374]) / (calc_distance(landmarks[362], landmarks[263]) + 1e-6)
    ear = (ear_left + ear_right) / 2.0
    
    # Distance between inner eyebrows (used for anger/sadness)
    brow_dist = calc_distance(landmarks[107], landmarks[336])
    
    # Distance between mouth corners (used for smiling/happiness)
    mouth_width = calc_distance(landmarks[61], landmarks[291])
    
    # Heuristic rules for classification
    if mar > 0.4 and ear > 0.28: 
        return "Shocked"
    elif brow_dist < 0.04 and ear < 0.25: 
        return "Angry"
    elif mar > 0.15 and ear > 0.29: 
        return "Fear"
    elif mouth_width > 0.13 and mar < 0.2: 
        return "Happy"
    elif brow_dist > 0.06 and mar < 0.1: 
        return "Sad"
    else: 
        return "Neutral"

# Set up video capture (0 for built-in webcam)
cap = cv2.VideoCapture(0)

print("Starting Heuristic Emotion Detection... (Press 'q' in the video window to quit)")

with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success: 
            print("Ignoring empty camera frame.")
            break
            
        # Flip image horizontally for a natural mirror view
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Optimize performance by marking image as not writeable
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        
        current_emotion = "Detecting..."
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the facial mesh
                mp_drawing.draw_landmarks(
                    image=image, 
                    landmark_list=face_landmarks, 
                    connections=mp_face_mesh.FACEMESH_TESSELATION, 
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Calculate emotion based on landmarks
                current_emotion = heuristic_emotion(face_landmarks.landmark)
                
                # Display the emotion on the video frame
                cv2.putText(
                    image, 
                    f"Emotion: {current_emotion}", 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, 
                    (0, 255, 0), 
                    3
                )
                
        # Show the video window
        cv2.imshow("Geometric Heuristics Emotion Recognition", image)

        # Press 'q' to break the loop and close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
