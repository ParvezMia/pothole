import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)

# Title of the app
st.title("Pothole Detection using YOLOv8")

# Load the model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Function to draw circles around potholes
def draw_circles(image, results):
    annotated_img = image.copy()
    
    if len(results.boxes) > 0:
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate center and radius of circle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = max((x2 - x1), (y2 - y1)) // 2
            
            # Draw circle
            cv2.circle(
                annotated_img,
                (center_x, center_y),
                radius,
                (0, 0, 255),  # Red color
                3  # Thickness
            )
            
            # Add confidence label
            conf = float(box.conf[0])
            label = f"Pothole: {conf:.2f}"
            cv2.putText(
                annotated_img,
                label,
                (center_x - 10, center_y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
    
    return annotated_img

# WebRTC implementation for real-time camera feed
class VideoProcessor(VideoProcessorBase):
    def __init__(self, model, confidence):
        self.model = model
        self.confidence = confidence
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame with the model
        results = self.model(img, conf=self.confidence)
        
        # Draw circles around detected potholes
        result_img = draw_circles(img, results[0])
        
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")

# Create the main app layout
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Live camera feed with automatic continuous detection
st.header("Live Pothole Detection")
st.write("Real-time pothole detection will automatically highlight detected potholes with red circles.")

# WebRTC configuration for accessing the camera
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="pothole-detection",
    video_processor_factory=lambda: VideoProcessor(model, confidence_threshold),
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,  # Enable async processing for smoother video
)

if webrtc_ctx.state.playing:
    st.info("The camera is active. Point it towards roads to detect potholes. Potholes will be automatically highlighted with red circles.")

# Instructions
st.markdown("""
### Instructions:
1. Allow camera access when prompted
2. Point your device camera at roads to detect potholes
3. Red circles will appear around detected potholes in real-time
4. Adjust the confidence threshold in the sidebar to control detection sensitivity
""")

# Additional features section
with st.expander("Additional Features"):
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        # Add a button to run detection
        if st.button("Detect Potholes"):
            with st.spinner("Processing image..."):
                # Run detection
                results = model(image_np, conf=confidence_threshold)
                result_image = draw_circles(image_np, results[0])
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("Detection Results")
                    st.image(result_image, use_column_width=True)
                
                # Get and display detection information
                if len(results[0].boxes) > 0:
                    st.success(f"Detected {len(results[0].boxes)} pothole(s)")
                else:
                    st.info("No potholes detected in this image.")

# About section
with st.expander("About this App"):
    st.write("""
    ### Pothole Detection App
    This application uses a YOLOv8 model trained specifically for pothole detection.
    
    #### Features:
    - Real-time, continuous pothole detection through your device's camera
    - Circular highlighting of detected potholes
    - Confidence scores for each detection
    - Adjustable confidence threshold for detection sensitivity
    
    #### How it works:
    The model continuously processes each frame from your camera feed to detect potholes.
    When a pothole is detected, a red circle is drawn around it with the confidence score.
    The detection happens automatically without requiring any user interaction.
    
    #### Use cases:
    - Road maintenance planning
    - Civil engineering applications
    - Public reporting of road conditions
    - Research on road quality assessment
    """)

# Footer
st.markdown("---")
st.markdown("Powered by YOLOv8 | Created with Streamlit")