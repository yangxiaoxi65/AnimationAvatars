import json
import numpy as np
from pathlib import Path

# Code adapted from https://github.com/tijiang13/InstantAvatar/blob/master/scripts/custom/convert_openpose_json_to_npy.py

def convert(json_folder, output_file='keypoints.npy'):
    json_path = Path(json_folder)
    npy_path = json_path.parent / output_file
    
    # Get all JSON files and sort them
    json_files = sorted([f for f in json_path.glob('*.json')])
    
    # Extract pose data from each file
    pose_data_list = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract pose keypoints and reshape
        pose_data = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        pose_data_list.append(pose_data)
    
    # Stack arrays and save
    if pose_data_list:
        np.save(npy_path, np.stack(pose_data_list))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert OpenPose JSON files to NPY file")
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="keypoints.npy")
    args = parser.parse_args()
    
    convert(args.json_dir, args.output_file)