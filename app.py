import streamlit as st
import os
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
from PIL import Image
import logging
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set page configuration
st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)

# Title of the app
st.title("Pothole Detection using YOLOv8")

# Fix for torch.classes error - import YOLO after setting environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Initialize session state variables
if 'webrtc_failed_attempts' not in st.session_state:
    st.session_state.webrtc_failed_attempts = 0

model = load_model()

# Function to draw circles around potholes
def draw_circles(image, results):
    annotated_img = image.copy()

    if results and len(results.boxes) > 0:
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
        try:
            img = frame.to_ndarray(format="bgr24")

            # Process the frame with the model
            if self.model:
                results = self.model(img, conf=self.confidence)
                # Draw circles around detected potholes
                result_img = draw_circles(img, results[0])
            else:
                result_img = img
                
            return av.VideoFrame.from_ndarray(result_img, format="bgr24")
        except Exception as e:
            logging.error(f"Error in video processing: {e}")
            # Return original frame if there's an error
            return frame

# Create the main app layout
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Live camera feed with automatic continuous detection
st.header("Live Pothole Detection")
st.write("Real-time pothole detection will automatically highlight detected potholes with red circles.")

# Network diagnostics option
if st.sidebar.checkbox("Enable Network Diagnostics"):
    if st.sidebar.button("Test WebRTC Connectivity"):
        st.sidebar.info("Testing STUN server connectivity...")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            s.connect(("stun.l.google.com", 19302))
            s.close()
            st.sidebar.success("STUN connectivity test passed!")
        except Exception as e:
            st.sidebar.error(f"STUN connectivity test failed: {e}")
            st.sidebar.warning("You may need to try a different network or use the image upload feature.")

# Improved WebRTC configuration with STUN and TURN servers
# Note: Replace placeholder TURN credentials with actual ones
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            # Uncomment and fill in with actual TURN server credentials
            # {
            #     "urls": "turn:your-turn-server.com:443?transport=tcp",
            #     "username": "your-username",
            #     "credential": "your-password"
            # }
        ],
        "iceTransportPolicy": "all",
        "bundlePolicy": "max-bundle",
        "sdpSemantics": "unified-plan",
    }
)

# Enable fallback option for connection issues
try:
    webrtc_ctx = webrtc_streamer(
        key="pothole-detection",
        video_processor_factory=lambda: VideoProcessor(model, confidence_threshold),
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        mode=WebRtcMode.SENDRECV,  # This ensures bidirectional communication
    )

    if webrtc_ctx.state.playing:
        st.info("The camera is active. Point it towards roads to detect potholes. Potholes will be automatically highlighted with red circles.")
    elif webrtc_ctx.state.failed:
        st.session_state.webrtc_failed_attempts += 1
        st.error("WebRTC connection failed. This may be due to network restrictions or browser compatibility.")
        
        if st.session_state.webrtc_failed_attempts > 1:
            st.warning("Multiple connection attempts have failed. Please try a different browser, network, or use the image upload feature below.")
        else:
            st.warning("Please try refreshing the page or use a different browser (Chrome or Firefox recommended).")

except Exception as e:
    st.error(f"WebRTC connection error: {e}")
    st.warning("Unable to establish camera connection. This might be due to network restrictions or firewall settings.")
    
    # Fallback to image upload only
    st.info("You can still use the image upload feature below.")

# Instructions section
st.markdown("""
### Instructions:
1. Allow camera access when prompted
2. Point your device camera at roads to detect potholes
3. Red circles will appear around detected potholes in real-time
4. Adjust the confidence threshold in the sidebar to control detection sensitivity
5. If camera connection fails, try refreshing or use the image upload feature
""")

# Additional features section - Image upload
with st.expander("Upload Image for Detection", expanded=True):
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Add a button to run detection
        if st.button("Detect Potholes"):
            if model:
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
            else:
                st.error("Model failed to load. Please try refreshing the page.")
                    
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
    - Fallback to image upload if camera connection fails
    
    #### How it works:
    The model continuously processes each frame from your camera feed to detect potholes.
    When a pothole is detected, a red circle is drawn around it with the confidence score.
    
    #### Use cases:
    - Road maintenance planning
    - Civil engineering applications
    - Public reporting of road conditions
    - Research on road quality assessment
    """)

# Handle errors explicitly and provide troubleshooting info
st.markdown("---")
st.markdown("Powered by YOLOv8 | Created with Streamlit")

# Add troubleshooting section
with st.expander("Troubleshooting"):
    st.write("""
    ### Common Issues:
    
    #### Camera Connection Errors
    - Try refreshing the page
    - Ensure you've given camera permissions in your browser
    - Try a different browser (Chrome or Firefox recommended)
    - If on a corporate network, firewall settings might block WebRTC connections
    - Use the network diagnostics in the sidebar to test connectivity
    
    #### Model Loading Errors
    - Refresh the page to reload the model
    - If persistent, try clearing your browser cache
    
    #### Performance Issues
    - For mobile devices, ensure good lighting conditions
    - Keep the device steady while detecting potholes
    
    #### Cloud Deployment Issues
    - WebRTC requires special configuration in cloud environments
    - If you're deploying on Streamlit Cloud and still having issues after using this code:
      1. Contact Streamlit support to verify WebRTC is fully supported
      2. Consider using a free TURN server provider like Twilio or OpenRelay
      3. If all else fails, you can remove the WebRTC component and focus on the image upload feature
    """)

# Update requirements.txt section at the bottom
with st.expander("Technical Information"):
    st.write("""
    ### Recommended Requirements (for best compatibility):
    ```
    streamlit==1.28.0
    streamlit-webrtc==0.47.0
    opencv-python-headless==4.8.0.74
    numpy==1.24.3
    ultralytics==8.0.145
    av==10.0.0
    pydub==0.25.1
    aiortc==1.5.0
    aioice==0.9.0
    ```
    """)
