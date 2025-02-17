import streamlit as st
import asyncio
import logging
import os
import torch
import gdown
from aiortc import RTCConfiguration, RTCIceServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the asyncio loop is correctly set
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

# Define STUN server for WebRTC (if using)
rtc_config = RTCConfiguration([
    RTCIceServer("stun:stun.l.google.com:19302"),
    RTCIceServer("stun:stun1.l.google.com:19302")
])

# Streamlit UI
st.title("🚀 Streamlit Deployment Debugger")

try:
    # Check Torch installation
    if torch.cuda.is_available():
        st.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("Running on CPU (CUDA not available)")

    # Check if `best.pt` exists, otherwise download
    model_path = "best.pt"
    google_drive_url = "https://drive.google.com/uc?id=1myS3kd5KrARwB4fFpWI09V6zFyIiEqV8"

    if not os.path.exists(model_path):
        st.info("Downloading best.pt from Google Drive...")
        gdown.download(google_drive_url, model_path, quiet=False)
        st.success("Download complete!")

    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Error: {e}")
    logger.error(f"Error encountered: {e}")

st.write("App is running fine! 🎉")
