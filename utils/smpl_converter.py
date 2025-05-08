import os
import pickle
import numpy as np
from typing import Optional, List, Dict, Any

def load_pkl(pkl_file: str) -> Dict[str, Any]:
    """
    Helper function to load SMPL data from a pkl file.
    
    Args:
        pkl_file (str): Path to the source pkl file with SMPL data.
    
    Returns:
        Dict[str, Any]: Dictionary containing the loaded data from the pkl file.
    """
    # Check if input file exists
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Input pkl file not found: {pkl_file}")
    
    # Load the pkl data
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded pickle file from {pkl_file}")
        
        # Ensure the loaded data is a dictionary
        if not isinstance(data, dict):
            data = {'data': data}
            print("Warning: Loaded data is not a dictionary. Wrapping it in a dictionary with key 'data'.")
        
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file: {str(e)}")

def pkl_to_npz(
    pkl_file: str, 
    video_name: str,
    output_dir: str = "assets",
    seq_length: int = 801,
    specific_keys: Optional[List[str]] = None
) -> str:
    """
    Convert SMPL data from pkl format to npz format for the Animation Avatar project.
    
    This function loads a pkl file containing SMPL data and creates a poses.npz file
    with the proper structure where all arrays (except betas) have their first 
    dimension representing the frame count.
    
    Args:
        pkl_file (str): Path to the source pkl file with SMPL data.
        video_name (str): Name of the video (folder name under assets).
        output_dir (str): Base directory for outputs (default: "assets").
        seq_length (int): Number of frames in the sequence (default: 801).
        specific_keys (List[str], optional): Specific SMPL data keys to include.
                                           If None, all relevant keys are included.
    
    Returns:
        str: Path to the created poses.npz file.
    """
    # Load the data using the helper function
    data = load_pkl(pkl_file)
    
    # Validate video_name is provided
    if not video_name:
        raise ValueError("video_name must be provided and cannot be empty")
    
    # Prepare output directory and path
    output_subdir = os.path.join(output_dir, video_name)
    os.makedirs(output_subdir, exist_ok=True)
    npz_path = os.path.join(output_subdir, "poses.npz")
    
    print(f"Converting data to {npz_path}...")
    
    # Default SMPL keys if not specified
    if specific_keys is None:
        specific_keys = [
            'betas', 'body_pose', 'global_orient', 'transl'
        ]
    
    # Process data for the npz file
    processed_data = {}
    
    # Process each key
    for key in specific_keys:
        if key in data:
            value = data[key]
            
            # Convert to numpy array if not already
            if not isinstance(value, np.ndarray):
                try:
                    value = np.array(value)
                except:
                    print(f"Warning: Could not convert '{key}' to numpy array. Skipping this key.")
                    continue
            
            # Process based on key type
            if key == 'betas':
                # betas should maintain its original shape (typically (10,))
                if value.shape == (1, 10):
                    processed_data[key] = value[0]  # Remove batch dimension
                else:
                    processed_data[key] = value
            else:
                # For all other arrays, first dimension should be seq_length
                if len(value.shape) == 1:
                    # Single vector to sequence
                    processed_data[key] = np.repeat(value.reshape(1, -1), seq_length, axis=0)
                elif value.shape[0] == 1:
                    # Has batch dimension, expand to sequence
                    processed_data[key] = np.repeat(value, seq_length, axis=0)
                else:
                    # Already has a first dimension, might need to adjust
                    if value.shape[0] != seq_length:
                        # Repeat or truncate to match seq_length
                        if value.shape[0] == 1:
                            processed_data[key] = np.repeat(value, seq_length, axis=0)
                        else:
                            print(f"Warning: {key} has {value.shape[0]} frames, expected {seq_length}.")
                            # If too few frames, repeat the last frame
                            if value.shape[0] < seq_length:
                                padding = np.repeat(value[-1:], seq_length - value.shape[0], axis=0)
                                processed_data[key] = np.concatenate([value, padding], axis=0)
                            # If too many frames, truncate
                            else:
                                processed_data[key] = value[:seq_length]
                    else:
                        processed_data[key] = value
        
        elif key == 'transl' and 'camera_translation' in data:
            # If transl is missing but camera_translation exists, use that
            transl = data['camera_translation']
            if isinstance(transl, np.ndarray):
                # Make sure it's the right shape (seq_length, 3)
                if len(transl.shape) == 1 and transl.shape[0] == 3:
                    processed_data[key] = np.repeat(transl.reshape(1, 3), seq_length, axis=0)
                elif transl.shape == (1, 3):
                    processed_data[key] = np.repeat(transl, seq_length, axis=0)
                else:
                    processed_data[key] = transl
        
        elif key == 'poses' and 'body_pose' in data and 'global_orient' in data:
            # If poses is missing but we have body_pose and global_orient, create it
            try:
                body_pose = data['body_pose']
                global_orient = data['global_orient']
                
                # Convert to numpy if needed
                if not isinstance(body_pose, np.ndarray):
                    body_pose = np.array(body_pose)
                if not isinstance(global_orient, np.ndarray):
                    global_orient = np.array(global_orient)
                
                # Reshape if needed
                if len(body_pose.shape) == 1:
                    body_pose = body_pose.reshape(1, -1)
                if len(global_orient.shape) == 1:
                    global_orient = global_orient.reshape(1, 3)
                
                # Make sure both arrays have batch dimension 1 or seq_length
                if body_pose.shape[0] == 1:
                    body_pose = np.repeat(body_pose, seq_length, axis=0)
                if global_orient.shape[0] == 1:
                    global_orient = np.repeat(global_orient, seq_length, axis=0)
                
                # Concatenate global_orient and body_pose
                poses = np.concatenate([global_orient, body_pose], axis=1)
                processed_data['poses'] = poses
                print(f"Created 'poses' by concatenating global_orient and body_pose")
            except Exception as e:
                print(f"Warning: Failed to create 'poses': {str(e)}")
    
    # Check if we're missing any keys
    missing_keys = set(specific_keys) - set(processed_data.keys())
    if missing_keys:
        print(f"Warning: Could not find or process these keys: {missing_keys}")
        # For missing keys, try to create placeholder data
        for key in missing_keys:
            if key == 'transl':
                # Create a zero translation sequence
                processed_data[key] = np.zeros((seq_length, 3), dtype=np.float32)
                print(f"Created placeholder zero data for '{key}'")
    
    # Save the npz file
    try:
        np.savez(npz_path, **processed_data)
        print(f"Successfully saved npz file to {npz_path}")
        return npz_path
    except Exception as e:
        raise RuntimeError(f"Failed to save npz file: {str(e)}")