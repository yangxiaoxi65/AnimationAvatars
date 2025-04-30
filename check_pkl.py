import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

def check_pkl(pkl_path, output_dir=None):
    """
    Analyze SMPL pose parameters in a PKL file
    
    Args:
        pkl_path: Path to the PKL file
        output_dir: Directory to save visualizations (optional)
    """
    print(f"Analyzing PKL file: {pkl_path}")
    
    # Create output directory for visualizations
    if output_dir is None:
        output_dir = os.path.dirname(pkl_path)
        output_dir = os.path.join(output_dir, "pkl_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load PKL file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict):
        print(f"ERROR: PKL file does not contain a dictionary, found {type(data)} instead")
        return
    
    # Print basic information
    print("\nPKL Contents:")
    print("=" * 50)
    print(f"Keys: {data.keys()}")
    
    # Check each key in the data
    for key in data.keys():
        value = data[key]
        if isinstance(value, np.ndarray):
            print(f"\n{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Type: {value.dtype}")
            print(f"  Stats: min={np.min(value)}, max={np.max(value)}, mean={np.mean(value)}")
            
            # Check for extreme values
            if value.size > 0:
                extreme_values = np.where(np.abs(value) > 2.0)
                if extreme_values[0].size > 0:
                    print(f"  WARNING: Found {extreme_values[0].size} extreme values (|x| > 2.0)")
                    if extreme_values[0].size < 10:  # Only show if not too many
                        print(f"  Extreme positions: {extreme_values}")
                        print(f"  Extreme values: {value[extreme_values]}")
            
            # Create visualization for body_pose or global_orient
            if key in ['body_pose', 'global_orient'] and value.size > 0:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.hist(value.flatten(), bins=50)
                plt.title(f"{key} Histogram")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                
                plt.subplot(1, 2, 2)
                if key == 'body_pose' and value.shape[1] > 10:
                    # Only plot first 10 dimensions if too many
                    plt.plot(value[0, :30])
                    plt.title(f"{key} First 10 Joints (x,y,z)")
                else:
                    plt.plot(value[0])
                    plt.title(f"{key} Values")
                plt.xlabel("Index")
                plt.ylabel("Value")
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{key}_analysis.png"))
                plt.close()
    
    # Detailed body pose analysis if available
    if 'body_pose' in data and isinstance(data['body_pose'], np.ndarray):
        body_pose = data['body_pose']
        
        if body_pose.ndim >= 2:
            # Try to reshape to joints
            try:
                # Assuming body_pose is [batch, N*3] where N is number of joints
                batch_size = body_pose.shape[0]
                num_values = body_pose.shape[1]
                
                if num_values % 3 == 0:
                    num_joints = num_values // 3
                    
                    # Reshape to [batch, joints, 3]
                    reshaped_pose = body_pose.reshape(batch_size, num_joints, 3)
                    
                    # Plot magnitude of each joint rotation
                    joint_magnitudes = np.linalg.norm(reshaped_pose[0], axis=1)
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(num_joints), joint_magnitudes)
                    plt.title("Magnitude of Joint Rotations")
                    plt.xlabel("Joint Index")
                    plt.ylabel("Rotation Magnitude (radians)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "joint_rotation_magnitudes.png"))
                    plt.close()
                    
                    # Find joints with extreme rotations
                    extreme_joints = np.where(joint_magnitudes > 1.5)[0]
                    if len(extreme_joints) > 0:
                        print("\nJoints with extreme rotations (> 1.5 rad):")
                        for joint_idx in extreme_joints:
                            print(f"  Joint {joint_idx}: magnitude = {joint_magnitudes[joint_idx]:.4f}")
                            print(f"    Rotation: {reshaped_pose[0, joint_idx]}")
            except Exception as e:
                print(f"Error in body pose analysis: {e}")
    
    print("\nAnalysis completed. Visualizations saved to:", output_dir)

def main():
    parser = ArgumentParser(description="Analyze SMPL parameters in a PKL file")
    parser.add_argument("pkl_path", type=str, help="Path to the PKL file")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    check_pkl(args.pkl_path, args.output_dir)

if __name__ == "__main__":
    main()