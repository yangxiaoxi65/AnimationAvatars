import numpy as np


def gen_camera_params(img, fov=60):
    """
    Generate camera parameters based on the image size and field of view (FOV).
    """
    f = max(img.shape[:2]) / 2 * 1 / np.tan(np.radians(fov/2))
    K = np.eye(3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = img.shape[1] / 2
    K[1, 2] = img.shape[0] / 2
    return {
        "intrinsic": K,
        "extrinsic": np.eye(4),
        "height": img.shape[0],
        "width": img.shape[1],
    }
