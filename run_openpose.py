from email.mime import image
import os
import subprocess
import logging
import shutil

from config.openpose_config import OpenposeConfig

"""
First stage of the pipeline: run OpenPose on a folder of images.
"""


def find_openpose_path() -> str:
    """
    Finds the OpenPose binary in the system PATH.

    Returns:
        str: The absolute path to the OpenPose binary.

    Raises:
        FileNotFoundError: If the OpenPose binary is not found.
    """
    openpose_path = shutil.which("openpose.bin")
    openpose_root_path = os.path.abspath(os.path.join(openpose_path, "../../../.."))
    if openpose_root_path:
        return openpose_root_path
    else:
        raise FileNotFoundError(
            "OpenPose binary not found. Ensure it is installed and in your PATH.")
        
def validate_folder(folder):
    """
    Validates that a folder exists and is not empty.

    Args:
        folder (str): The path to the folder.

    Raises:
        AssertionError: If the folder does not exist or is empty.
    """
    if not os.path.isdir(folder) or not os.listdir(folder):
        raise AssertionError(f"Folder {folder} is missing or empty.")


def run_openpose(image_folder)-> None:
    """
    Runs OpenPose on a folder of images.

    Args:
        image_folder (str): Path to the folder containing input images.
        config (dict): Configuration for OpenPose parameters.

    Raises:
        Exception: If OpenPose execution or post-processing fails.
    """
    print("""
            **********************************
            *   Stage 1: Running OpenPose..  *
            **********************************
          """)
    logging.info(f"Running on folder: {image_folder}")
    validate_folder(image_folder)
    image_folder = os.path.abspath(image_folder)
    openpose_root_path = find_openpose_path()
    logging.info(f"OpenPose path: {openpose_root_path}")
    config = OpenposeConfig(image_folder)
    os.makedirs(config.output_json_folder, exist_ok=True)
    os.makedirs(config.output_images_folder, exist_ok=True)
    logging.info(f"OpenPose configuration: {config}")
    cwd = os.getcwd()
    try:
        os.chdir(openpose_root_path)
        subprocess.run([
                    './build/examples/openpose/openpose.bin',
                    '--image_dir', image_folder,
                    '--display', config.display,
                    '--write_json', config.output_json_folder,
                    '--write_images', config.output_images_folder,
                    '--write_images_format', config.write_images_format,
                    '--render_pose', config.render_pose,
                    '--render_threshold', config.render_threshold,
                    '--number_people_max', config.number_people_max,
                    '--model_pose', config.model_pose,
                ])
        validate_folder(config.output_json_folder)
        validate_folder(config.output_images_folder)
    except Exception as e:
            logging.error(f"OpenPose execution failed: {e}")
    finally:
        logging.info("OpenPose completed successfully.")
        os.chdir(cwd)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    image_folder = "./assets/7/images"
    run_openpose(image_folder)