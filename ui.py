import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:5000/predict"

st.title("Shoplifting Detection")
st.write("Upload a video and get video with shoplifting detection!")
uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi"])

if uploaded_video is not None:
    if st.button("Predict"):
        with open(uploaded_video.name, "wb") as f:
            f.write(uploaded_video.getbuffer())
        files = {
            "video": (
                uploaded_video.name,
                uploaded_video.read(),
            )
        }
        response = requests.post(BACKEND_URL, files=files)
        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            output_path = result["output_path"]
            video_info = result.get("video_info")
            st.success("Prediction complete!")
            with open(output_path, "rb") as video_bytes:
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name=f"processed_{uploaded_video.name}",
                )
