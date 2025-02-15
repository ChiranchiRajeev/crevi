import os
import cv2
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
import streamlit as st

# Function to create a video from images
def create_video_from_images(images, output_path, frame_rate=1):
    if not images:
        st.error("No images provided.")
        return

    # Load the first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Add all images to the video
    for image in images:
        frame = cv2.imread(image)
        if frame is not None:
            video.write(frame)
        else:
            st.warning(f"Could not load image: {image}")

    video.release()

# Function to add audio to a video
def add_audio_to_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Streamlit App
def main():
    st.title("Image to Video Creator with Audio")

    # Step 1: Upload Images
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)
    temp_image_paths = []

    if uploaded_images:
        for img in uploaded_images:
            temp_path = os.path.join("temp", img.name)
            with open(temp_path, "wb") as f:
                f.write(img.read())
            temp_image_paths.append(temp_path)

    # Step 2: Upload Audio
    uploaded_audio = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    audio_path = None

    if uploaded_audio:
        audio_path = os.path.join("temp", uploaded_audio.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())

    # Step 3: Generate Video Button
    if st.button("Generate Video"):
        if not temp_image_paths or not audio_path:
            st.error("Please upload both images and an audio file before generating the video.")
        else:
            output_video_path = "final_output_video.mp4"
            temp_video_path = "temp_video.mp4"

            # Create video from images
            create_video_from_images(temp_image_paths, temp_video_path)

            # Add audio to the video
            add_audio_to_video(temp_video_path, audio_path, output_video_path)

            # Display download link
            st.success("Video generated successfully!")
            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="Download Video",
                    data=video_file,
                    file_name="final_output_video.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)  # Ensure temp directory exists
    main()
