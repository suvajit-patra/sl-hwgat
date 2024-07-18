import mediapipe as mp
import cv2
import numpy as np
from typing import List, Optional, Union
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

class Model:
    def __init__(self) -> None:
        self.holistic = mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.0, min_tracking_confidence=0.0)

    def __call__(self, image:Union[np.ndarray, str]):
        if isinstance(image, str):
         # getting landmarks from a frame using mediapipe holistic
            image = cv2.imread(image)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)  
        return self.get_frame_features(results), *image.shape[:2] #return result_kps 2d with confidence, height width
        
    def get_frame_features(self, results: object) -> np.ndarray:
        """Organizes the landmarks or keypoints of a frame in a numpy ndarray extracted from mediapipe results object.

        Args:
            results (object): Medaipipe results object from a single processed frame

        Returns:
            (ndarray): A ndarray of 1662 dimension containing 3d (x, y, z) keypoints (pose keypoints contains visibility too) for a single frame from video
        """
        default_visibility = 1
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,4))
        face = np.array([[res.x, res.y, res.z, default_visibility] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468,4))
        lh = np.array([[res.x, res.y, res.z, default_visibility] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,4))
        rh = np.array([[res.x, res.y, res.z, default_visibility] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,4))
        return np.concatenate([pose, face, lh, rh])