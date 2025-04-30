import os
import torch
import numpy as np
import pickle
import trimesh
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

def create_t_pose():
    """Create a T-pose (all joints = 0)"""
    return np.zeros((1, 63), dtype=np.float32)  # For SMPL-X

def create_a_pose():
    """Create an A-pose (arms slightly raised)"""
    pose = np.zeros((1, 63), dtype=np.float32)
    
    # SMPL-X joint order might vary, these are approximate indices for shoulders
    # Left shoulder (raise arm)
    pose[0, 15] = 0.5  # ~30 degrees
    # Right shoulder (raise arm)
    pose[0, 18] = -0.5  # ~30 degrees
    
    return pose

def test_smpl_model(smpl_model_path, gender='neutral', pose_type='t', output_dir=None):
    """
    Test SMPL model with basic poses
    
    Args:
        smpl_model_path: Path to the SMPL model directory
        gender: Gender for the SMPL model
        pose_type: Type of pose ('t', 'a', or 'pkl')
        output_dir: Directory to save output files
    """
    # Setup paths
    if output_dir is None:
        output_dir = "smpl_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create SMPL model
    try:
        # Try SMPL-X first
        model = smplx.create(
            model_path=smpl_model_path,
            model_type='smplx',
            gender=gender,
            use_pca=False,
            batch_size=1,
            device=device
        )
        model_type = "SMPL-X"
        print(f"Initialized SMPL-X model with {gender} gender")
    except Exception as e:
        print(f"Error initializing SMPL-X: {e}")
        try:
            # Fall back to SMPL
            model = smplx.create(
                model_path=smpl_model_path,
                model_type='smpl',
                gender=gender,
                batch_size=1,
                device=device
            )
            model_type = "SMPL"
            print(f"Initialized SMPL model with {gender} gender")
        except Exception as e2:
            print(f"Error initializing SMPL: {e2}")
            return
    
    # Create pose based on selected type
    if pose_type == 't':
        print("Creating T-pose")
        body_pose = create_t_pose()
    elif pose_type == 'a':
        print("Creating A-pose")
        body_pose = create_a_pose()
    else:
        print("Unknown pose type. Using T-pose.")
        body_pose = create_t_pose()
    
    # Convert to tensor
    body_pose_tensor = torch.tensor(body_pose, dtype=torch.float32, device=device)
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(
                body_pose=body_pose_tensor,
                return_verts=True
            )
        
        # Get vertices and faces
        vertices = output.vertices[0].detach().cpu().numpy()
        faces = model.faces
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Save mesh
        output_mesh_path = os.path.join(output_dir, f"{model_type.lower()}_{gender}_{pose_type}_pose.obj")
        mesh.export(output_mesh_path)
        print(f"Saved mesh to {output_mesh_path}")
        
        # Create visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[::100, 0], vertices[::100, 1], vertices[::100, 2], c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{model_type} {gender} {pose_type}-pose")
        
        # Set equal aspect ratio
        max_range = np.array([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ]).max() / 2.0
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.savefig(os.path.join(output_dir, f"{model_type.lower()}_{gender}_{pose_type}_pose.png"))
        plt.close()
        
        print("Test completed successfully!")
        return output_mesh_path
        
    except Exception as e:
        print(f"Error in SMPL forward pass: {e}")
        return None

def main():
    parser = ArgumentParser(description="Test SMPL model with basic poses")
    parser.add_argument("--smpl_model_path", type=str, default="models",
                        help="Path to the SMPL model directory")
    parser.add_argument("--gender", type=str, default="neutral",
                        choices=["neutral", "male", "female"],
                        help="Gender for the SMPL model")
    parser.add_argument("--pose_type", type=str, default="t",
                        choices=["t", "a"],
                        help="Type of pose (t=T-pose, a=A-pose)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output files")
    
    args = parser.parse_args()
    test_smpl_model(
        args.smpl_model_path,
        args.gender,
        args.pose_type,
        args.output_dir
    )

if __name__ == "__main__":
    main()