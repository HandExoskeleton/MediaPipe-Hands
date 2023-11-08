import moviepy.editor as mp


def create_side_by_side_video(num_videos):
    video_clips = []

    for i in range(num_videos):
        video_path = input(f"Enter the path of video {i + 1}: ")
        video_clip = mp.VideoFileClip(video_path)
        video_clips.append(video_clip)

    # Find the minimum height and width among the videos
    min_height = min([clip.h for clip in video_clips])
    min_width = min([clip.w for clip in video_clips])

    # Resize all clips to have the same dimensions
    video_clips = [clip.resize((min_width, min_height)) for clip in video_clips]

    # Combine video clips side by side
    final_clip = mp.clips_array([video_clips])

    # Write the combined video to a file
    output_path = input("Enter the output video file path: ")
    final_clip.write_videofile(output_path, codec="libx264")


if __name__ == "__main__":
    num_videos = int(input("Enter the number of videos: "))
    create_side_by_side_video(num_videos)
