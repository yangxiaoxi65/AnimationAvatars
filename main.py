import argparse
import logging
import os

from config.openpose_config import OpenposeConfig
from utils.ffmpeg_utils import extract_frames_from_video
from utils.convert_openpose_json_to_npy import convert
from run_openpose import run_openpose


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Animattion Avatars Pipeline (stage 1-3)")
    parser.add_argument(
        '-f', '--force', action='store_true',
        help="Force re-run the pipeline even if output files exist.")
    
    parser.add_argument(
        "video_path", type=str, help="Path to the input video file.")
    return parser.parse_args()


def main(args):
    print("""
        **********************************
        *   Stage 0: Extracting frames.. *
        **********************************
        """)
    frames_dir = extract_frames_from_video(args.video_path)

    print("""
        **********************************
        *   Stage 1: Running OpenPose..  *
        **********************************
        """)
    base_dir = os.path.abspath(os.path.dirname(frames_dir))
    config = OpenposeConfig(base_dir)
    if os.path.exists(os.path.join(base_dir, "keypoints.npy")):
        print("Keypoints already exist, skipping OpenPose.")
    else:
        run_openpose(config)
        convert(config.output_json_folder)



if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args)
