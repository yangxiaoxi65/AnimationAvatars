from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def process_video(data_dir):
    sam = sam_model_registry["vit_h"](checkpoint="/school/Computer_Vision/final_proj/segment-anything/checkpoints/sam_vit_h_4b8939.pth")
    sam.to("cuda")
    predictor = SamPredictor(sam)
    img_lists = sorted(glob.glob(f"{data_dir}/images/*.jpg"))
    keypoints = np.load(f"{data_dir}/keypoints.npy")
    os.makedirs(f"{data_dir}/masks_sam", exist_ok=True)
    os.makedirs(f"{data_dir}/masks_sam_images", exist_ok=True)
    for fn, pts in tqdm(zip(img_lists, keypoints), total=len(img_lists)):
        img = cv2.imread(fn)
        predictor.set_image(img)
        m = pts[..., -1] > 0.5
        pts = pts[m]
        masks, _, _ = predictor.predict(pts[:, :2], np.ones_like(pts[:, 0]))
        mask = masks.sum(axis=0) > 0
        cv2.imwrite(fn.replace("images", "masks_sam"), mask * 255)

        img[~mask] = 0
        cv2.imwrite(fn.replace("images", "masks_sam_images"), img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    process_video(args.data_dir)
