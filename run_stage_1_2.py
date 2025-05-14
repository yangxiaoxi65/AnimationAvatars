import argparse
import logging
import os

import cv2
import numpy as np
from config.openpose_config import OpenposeConfig
from config.sam_config import SamConfig
from utils.ffmpeg_utils import extract_frames_from_video
from utils.convert_openpose_json_to_npy import convert
from utils.gen_camera import gen_camera_params
from run_openpose import run_openpose
from run_sam import process_video


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
    openpose_config = OpenposeConfig(base_dir)
    if os.path.exists(os.path.join(base_dir, "keypoints.npy")):
        print("Keypoints already exist, skipping OpenPose.")
    else:
        run_openpose(openpose_config)
        convert(openpose_config.output_json_folder)
    print("""
        **********************************
        *   Stage 2: Running  SAM..      *
        **********************************
        """)
    sam_config = SamConfig(base_dir)
    process_video(sam_config)
    print("""
        **********************************
        *   Stage 3: Generating camera parameters  *
        **********************************
        """)
    
    # save camera parameters
    sample_image_name = os.listdir(openpose_config.input_folder)[0]
    sample_image_path = os.path.join(
        openpose_config.input_folder, sample_image_name)
    img = cv2.imread(sample_image_path)
    cam_parms = gen_camera_params(img, fov=60)
    np.save(base_dir + "/cameras.npz", cam_parms)
    
    # use simplify-x



if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args)
