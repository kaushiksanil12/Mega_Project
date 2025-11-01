from moviepy.editor import VideoFileClip, AudioFileClip
import os


def replace_video_audio_moviepy(video_path, audio_path, output_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading video...")
    video = VideoFileClip(video_path)

    print("Loading audio...")
    audio = AudioFileClip(audio_path)

    # Truncate the audio if longer; pad if shorter (optional)
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
        print("Clipped audio to match video duration.")
    elif video.duration > audio.duration:
        print("Warning: video is longer than audio. Silence will be appended at end.")

    print("Replacing audio in video...")
    final = video.set_audio(audio)
    print(f"Writing output to {output_path} ... (this may take a minute)")
    
    # Add temp_audiofile parameter for Windows compatibility
    final.write_videofile(
        output_path, 
        codec='libx264', 
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

    video.close()
    audio.close()
    final.close()
    print("✅ Completed. Check the output video.")


if __name__ == "__main__":
    video_in = r"C:/Users/kaush/OneDrive/Desktop/projects/mega project/video-audio-pipeline/audio-processor/input/original.mp4"
    audio_in = r"C:/Users/kaush/OneDrive/Desktop/projects/mega project/video-audio-pipeline/audio-alignment/output/final_dubbed_audio_stereo.wav"
    video_out = r"./output/final_dubbed_video_moviepy.mp4"
    replace_video_audio_moviepy(video_in, audio_in, video_out)
