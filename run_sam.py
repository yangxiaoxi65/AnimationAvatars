from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

from config.sam_config import SamConfig

def process_video(config: SamConfig):
    """
    Process video frames using SAM (Segment Anything Model) to generate masks.
    
    Args:
        config (SamConfig): configuration object containing model and input/output settings.
    """
    
    # Create output directories
    os.makedirs(config.mask_dir, exist_ok=True)
    os.makedirs(config.mask_image_dir, exist_ok=True)
    
    # Load SAM model
    sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint_dir)
    sam.to(config.device)
    predictor = SamPredictor(sam)
    
    # Get input files
    img_lists = sorted(glob.glob(config.image_pattern))
    
    # Check if necessary files exist
    if not img_lists:
        raise FileNotFoundError(f"No matching for: {config.image_pattern}")
    if not os.path.exists(config.keypoints_path):
        raise FileNotFoundError(f"Keypoints file not found: {config.keypoints_path}")
        
    keypoints = np.load(config.keypoints_path, allow_pickle=True)
    
    # Process each image
    for file_name, pts in tqdm(zip(img_lists, keypoints), total=len(img_lists)):
        img = cv2.imread(file_name)
        if img is None:
            print(f"Warning: Could not read image {file_name}, skipping")
            continue
            
        predictor.set_image(img)
        
        # Filter keypoints by confidence
        valid_mask = pts[..., -1] > 0.5
        valid_pts = pts[valid_mask]
        
        if len(valid_pts) == 0:
            print(f"Warning: No valid keypoints found for {file_name}, skipping")
            continue
        
        # Generate masks
        masks, _, _ = predictor.predict(
            point_coords=valid_pts[:, :2],
            point_labels=np.ones_like(valid_pts[:, 0])
        )
        
        # Combine all masks
        combined_mask = masks.sum(axis=0) > 0
        
        # Save mask
        mask_filename = os.path.join(config.mask_dir, os.path.basename(file_name))
        cv2.imwrite(mask_filename, combined_mask.astype(np.uint8) * 255)
        
        # Apply mask to image and save
        masked_img = img.copy()
        masked_img[~combined_mask] = 0
        masked_img_filename = os.path.join(config.mask_image_dir, os.path.basename(file_name))
        cv2.imwrite(masked_img_filename, masked_img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    config = SamConfig(args.data_dir)
    process_video(config)
