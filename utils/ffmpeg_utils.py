from ast import Str
import ffmpeg
import os


def extract_frames_from_video(video_path: str) -> str:
    """
    Extract frames from a video file and save them as images in a specified folder.

    Args:
        video_path (str): Path to the input video file.

    Raises:
        FileNotFoundError: If the input video file does not exist.
    
    Returns:
        str: Path to the folder containing the extracted images.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_folder = os.path.join(video_dir, video_name, 'images')
    os.makedirs(output_folder, exist_ok=True)

    ffmpeg.input(video_path).output(os.path.join(
        output_folder, '%04d.jpg')).run(overwrite_output=True)
    return output_folder


def merge_images_to_video(image_folder: str, image_filename_structure: str, output_video_path: str, fps: int = 30) -> None:
    """
    Merge images from a folder into a video file.

    Args:
        image_folder (str): Path to the folder containing input images.
        output_video_path (str): Path to the output video file.
        fps (int): Frames per second for the output video. Default is 30.

    Raises:
        FileNotFoundError: If the input image folder does not exist or is empty.
    """
    if not os.path.isdir(image_folder) or not os.listdir(image_folder):
        raise FileNotFoundError(
            f"Image folder {image_folder} is missing or empty.")

    ffmpeg.input(os.path.join(image_folder, image_filename_structure),
                 framerate=fps).output(output_video_path).run(overwrite_output=True)
