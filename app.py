import streamlit as st
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import tempfile
import os

st.set_page_config(page_title="Deepfake Detection", layout="centered")

mp_face_mesh = mp.solutions.face_mesh

class FaceStabilityAnalyzer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.landmark_history = deque(maxlen=window_size)
        self.blinking_history = deque(maxlen=window_size)
    
    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    face_coords = np.array([
                        [lm.x, lm.y, lm.z] for lm in landmarks.landmark
                    ])
                    self.landmark_history.append(face_coords)
                    blink_score = self.detect_blink(landmarks)
                    self.blinking_history.append(blink_score)
                
                frame_count += 1
        
        cap.release()
        
        return {
            'landmark_stability': self.calculate_landmark_stability(),
            'blink_frequency': self.calculate_blink_frequency(),
            'head_movement_smoothness': self.calculate_head_smoothness(),
            'overall_deepfake_score': self.get_deepfake_score()
        }
    
    def detect_blink(self, landmarks):
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 386, 374
        LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 263, 362
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 159, 145
        RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 33, 133
        
        left_eye_top = np.array([landmarks.landmark[LEFT_EYE_TOP].x, landmarks.landmark[LEFT_EYE_TOP].y])
        left_eye_bottom = np.array([landmarks.landmark[LEFT_EYE_BOTTOM].x, landmarks.landmark[LEFT_EYE_BOTTOM].y])
        left_eye_left = np.array([landmarks.landmark[LEFT_EYE_LEFT].x, landmarks.landmark[LEFT_EYE_LEFT].y])
        left_eye_right = np.array([landmarks.landmark[LEFT_EYE_RIGHT].x, landmarks.landmark[LEFT_EYE_RIGHT].y])
        
        right_eye_top = np.array([landmarks.landmark[RIGHT_EYE_TOP].x, landmarks.landmark[RIGHT_EYE_TOP].y])
        right_eye_bottom = np.array([landmarks.landmark[RIGHT_EYE_BOTTOM].x, landmarks.landmark[RIGHT_EYE_BOTTOM].y])
        right_eye_left = np.array([landmarks.landmark[RIGHT_EYE_LEFT].x, landmarks.landmark[RIGHT_EYE_LEFT].y])
        right_eye_right = np.array([landmarks.landmark[RIGHT_EYE_RIGHT].x, landmarks.landmark[RIGHT_EYE_RIGHT].y])
        
        left_ear = self.eye_aspect_ratio(left_eye_top, left_eye_bottom, left_eye_left, left_eye_right)
        right_ear = self.eye_aspect_ratio(right_eye_top, right_eye_bottom, right_eye_left, right_eye_right)
        
        return (left_ear + right_ear) / 2
    
    def eye_aspect_ratio(self, top, bottom, left, right):
        vertical_dist = np.linalg.norm(top - bottom)
        horizontal_dist = np.linalg.norm(left - right)
        return vertical_dist / (2 * horizontal_dist + 1e-6)
    
    def calculate_landmark_stability(self):
        if len(self.landmark_history) < 2:
            return 0.5
        
        differences = []
        for i in range(1, len(self.landmark_history)):
            diff = np.linalg.norm(self.landmark_history[i] - self.landmark_history[i-1], axis=1)
            differences.append(np.mean(diff))
        
        movement_variance = np.var(differences)
        return min(movement_variance, 1.0)
      
    def calculate_blink_frequency(self):
        if len(self.blinking_history) < 10:
            return 0.5
        
        blink_history = list(self.blinking_history)
        blink_count = sum(1 for i in range(1, len(blink_history)) 
                         if blink_history[i] < 0.2 and blink_history[i-1] > 0.3)
        
        total_seconds = len(blink_history) / 30
        blinks_per_minute = (blink_count / total_seconds) * 60
        
        blink_deviation = abs(blinks_per_minute - 17) / 17
        return min(blink_deviation, 1.0)
    
    def calculate_head_smoothness(self):
        if len(self.landmark_history) < 3:
            return 0.5
        
        nose_idx = 1
        head_positions = np.array([[lm[nose_idx, 0], lm[nose_idx, 1]] for lm in self.landmark_history])
        
        velocities = np.diff(head_positions, axis=0)
        velocity_magnitude = np.linalg.norm(velocities, axis=1)
        accelerations = np.diff(velocity_magnitude)
        
        return min(np.var(accelerations), 1.0)
    
    def get_deepfake_score(self):
        stability = self.calculate_landmark_stability()
        blink_freq = self.calculate_blink_frequency()
        smoothness = self.calculate_head_smoothness()
        
        return min(0.4 * stability + 0.3 * blink_freq + 0.3 * smoothness, 1.0)

# Streamlit UI
st.image("logo.jpg", width=100)
st.markdown("# KYC-Deepfake Detection System")
st.markdown("Proposed by **Team-Alpha Coder**  | Real-time Deepfake KYC detection System")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'webm'])

with col2:
    st.markdown("### Supported Formats")
    st.markdown("- MP4\n- AVI\n- MOV\n- WebM")

if uploaded_file is not None:
    # Save temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    st.success("✅ Video uploaded!")
    
    if st.button("🔍 Analyze for Deepfakes", use_container_width=True):
        st.info("Analyzing video... This may take a moment.")
        
        analyzer = FaceStabilityAnalyzer()
        results = analyzer.analyze_video(tfile.name)
        
        os.unlink(tfile.name)
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        score = results['overall_deepfake_score']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Deepfake Score", f"{score:.3f}")
        with col2:
            st.metric("Stability", f"{results['landmark_stability']:.3f}")
        with col3:
            st.metric("Blink Freq", f"{results['blink_frequency']:.3f}")
        with col4:
            st.metric("Smoothness", f"{results['head_movement_smoothness']:.3f}")
        
        # Verdict
        st.markdown("---")
        
        if score > 0.7:
            st.error(f"❌ **LIKELY DEEPFAKE** - Score: {score:.3f}\n\nRecommendation: **BLOCK**")
        elif score > 0.5:
            st.warning(f"⚠️ **SUSPICIOUS** - Score: {score:.3f}\n\nRecommendation: **STEP-UP AUTHENTICATION REQUIRED**")
        else:
            st.success(f"✅ **LIKELY REAL** - Score: {score:.3f}\n\nRecommendation: **APPROVE**")
