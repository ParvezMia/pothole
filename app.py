import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from PIL import Image
import logging

# Set page configuration
st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Title of the app
st.title("Pothole Detection using YOLOv8")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Failed to load detection model. Please check the model file.")
        st.stop()

model = load_model()

# Improved RTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},\
    ]
})

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.5
        self.model = model
        self.frame_count = 0
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Skip frames for better performance
            if self.frame_count % 3 != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Convert to RGB for model
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model.predict(
                source=img_rgb,
                conf=self.confidence,
                verbose=False
            )
            
            # Draw detections
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    radius = max((x2 - x1) // 2, (y2 - y1) // 2)
                    cv2.circle(img, center, radius, (0, 0, 255), 3)
                    cv2.putText(img, f"{box.conf[0]:.2f}", 
                               (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, (0, 0, 255), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.5, 0.05
    )
    
    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="pothole-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "style": {"margin": "0 auto", "border": "5px solid white"},
            "controls": False,
            "autoPlay": True,
        },
    )
    
    if ctx.video_processor:
        ctx.video_processor.confidence = confidence_threshold

    # Rest of your code for image upload and UI...

if __name__ == "__main__":
    main()
