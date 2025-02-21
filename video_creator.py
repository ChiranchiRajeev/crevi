import os
import cv2
import numpy as np
import streamlit as st
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
import librosa
import soundfile as sf
import tempfile
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Temporary directory
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Function to detect beats in audio
def detect_beats(audio_path):
    logger.debug(f"Detecting beats in {audio_path}")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times, tempo

# Function to resize media
def resize_media(frame_or_clip, target_width, target_height):
    if isinstance(frame_or_clip, np.ndarray):
        return cv2.resize(frame_or_clip, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        return frame_or_clip.resize((target_width, target_height))

# Function to create beat-synced video
def create_beat_synced_video(media_files, audio_path, output_path, progress_bar):
    logger.debug("Starting beat-synced video creation")
    beat_times, tempo = detect_beats(audio_path)
    audio = AudioFileClip(audio_path)
    clips = []
    media_index = 0
    total_beats = len(beat_times)

    # Get dimensions from first media
    first_media = media_files[0]
    if first_media.endswith((".jpg", ".png")):
        frame = cv2.imread(first_media)
        height, width = frame.shape[:2]
    else:
        clip = VideoFileClip(first_media)
        width, height = clip.size
        clip.close()

    st.write(f"Detected {total_beats} beats at {tempo:.2f} BPM. Syncing media...")

    for i, beat_time in enumerate(beat_times):
        if media_index >= len(media_files):
            break
        progress_bar.progress((i + 1) / total_beats)
        next_beat = beat_times[i + 1] if i + 1 < len(beat_times) else audio.duration
        duration = max(next_beat - beat_time, 0.5)

        media = media_files[media_index]
        logger.debug(f"Processing media: {media}")
        if media.endswith((".jpg", ".png")):
            img = cv2.imread(media)
            if img is None:
                logger.error(f"Failed to load image: {media}")
                continue
            img = resize_media(img, width, height)
            clip = ImageClip(img, duration=duration).set_start(beat_time)
        else:
            video = VideoFileClip(media)
            video = resize_media(video, width, height)
            clip = video.subclip(0, min(duration, video.duration)).set_start(beat_time)
            video.close()

        clips.append(clip)
        media_index += 1

    # Add countdown if media runs out
    if media_index < len(media_files):
        remaining_time = audio.duration - beat_times[media_index - 1]
        logger.debug(f"Adding countdown for {remaining_time}s")
        countdown_clip = create_countdown_clip(remaining_time, width, height)
        countdown_clip = countdown_clip.set_start(beat_times[media_index - 1])
        clips.append(countdown_clip)

    logger.debug("Concatenating clips")
    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.set_audio(audio)
    logger.debug(f"Writing video to {output_path}")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)

# Function to create countdown clip
def create_countdown_clip(duration, width, height):
    logger.debug(f"Creating countdown clip for {duration}s, {width}x{height}")
    frames = int(duration * 24)  # 24 FPS
    countdown_frames = []
    for i in range(frames):
        time_left = duration - (i / 24)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        text = "Ending in {:.1f}s".format(float(time_left))  # Explicit float conversion
        logger.debug(f"Adding text: {text}")
        cv2.putText(frame, text, (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        countdown_frames.append(frame.copy())  # Avoid reference issues
    return ImageClip(countdown_frames, fps=24)  # Pass as list, not np.array directly

# Streamlit App
def main():
    st.title("🎵 Advanced Beat-Synced Video Editor 🎥")
    st.markdown("Create dynamic videos synced to music beats!")

    st.sidebar.header("Settings")
    frame_rate = st.sidebar.slider("Default Frame Rate (for images)", 1, 30, 24)
    preview_enabled = st.sidebar.checkbox("Enable Preview", value=False)

    st.subheader("1. Upload Images or Videos")
    uploaded_media = st.file_uploader("Upload Images/Videos", type=["jpg", "png", "mp4"],
                                      accept_multiple_files=True)
    media_paths = []

    if uploaded_media:
        for media in uploaded_media:
            temp_path = os.path.join(TEMP_DIR, media.name)
            with open(temp_path, "wb") as f:
                f.write(media.read())
            media_paths.append(temp_path)
        st.write(f"Uploaded {len(media_paths)} files.")

    st.subheader("2. Upload Audio")
    uploaded_audio = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    audio_path = None

    if uploaded_audio:
        audio_path = os.path.join(TEMP_DIR, uploaded_audio.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())
        st.audio(audio_path)

    if st.button("Generate Beat-Synced Video"):
        if not media_paths or not audio_path:
            st.error("Please upload media files and an audio file.")
        else:
            output_video_path = os.path.join(TEMP_DIR, "final_output.mp4")
            progress_bar = st.progress(0)

            with st.spinner("Processing video..."):
                try:
                    create_beat_synced_video(media_paths, audio_path, output_video_path, progress_bar)
                    st.success("Video generated successfully!")
                except Exception as e:
                    st.error(f"Video generation failed: {str(e)}")
                    logger.exception("Video generation error")
                    return

            if preview_enabled:
                st.subheading("Preview")
                st.video(output_video_path)

            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="Download Video",
                    data=video_file,
                    file_name="beat_synced_video.mp4",
                    mime="video/mp4"
                )

    if st.button("Clean Up Temporary Files"):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.success("Temporary files cleared.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Main execution error")
