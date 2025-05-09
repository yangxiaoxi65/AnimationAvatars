import os

class SamConfig:
    def __init__(self, data_dir):
        # model settings
        self.model_type = "vit_h"
        self.checkpoint_dir = "./segment-anything/checkpoints/sam_vit_h_4b8939.pth"
        self.device = "cuda"
        
        #input settings
        self.image_pattern = os.path.join(data_dir, "images", "*.jpg")
        self.keypoints_path = os.path.join(data_dir, "keypoints.npy")
        # output settings
        self.mask_dir = os.path.join(data_dir, "masks")
        self.mask_image_dir = os.path.join(data_dir, "mask_images")