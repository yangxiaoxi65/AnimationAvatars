import argparse

from utils.ffmpeg_utils import extract_frames_from_video, merge_images_to_video
from run_openpose import run_openpose

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run Animattion Avatars Pipeline (stage 1-3)")
    parser.add_argument(
        "video_path", type=str, help="Path to the input video file.")
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    video_path = args.video_path
    frames_dir = extract_frames_from_video(args.video_path)
    run_openpose(frames_dir)
    
