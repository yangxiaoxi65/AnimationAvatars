import os
import pickle
import torch
import numpy as np
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import sys
import argparse

# Try to import human_body_prior for VPoser
try:
    from human_body_prior.tools.model_loader import load_vposer
    has_vposer = True
    print("VPoser module found!")
except ImportError:
    has_vposer = False
    print("VPoser module not found. Some functionality will be limited.")

def analyze_pkl(pkl_path, output_dir=None, vposer_path=None):
    """
    Analyze a SMPL/SMPL-X parameter PKL file and visualize the model
    
    Args:
        pkl_path: Path to the PKL file
        output_dir: Directory to save visualizations
        vposer_path: Path to VPoser model checkpoint (optional)
    """
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pkl_path), "pkl_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing PKL file: {pkl_path}")
    
    # Load PKL file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Print PKL contents
    print("\nPKL Contents:")
    print("=" * 50)
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            print(f"   min={np.min(value)}, max={np.max(value)}, mean={np.mean(value)}")
    
    # Check if we have VPoser body_pose
    has_vposer_pose = 'body_pose' in data and data['body_pose'].shape[1] == 32
    print(f"\nVPoser encoded body_pose detected: {has_vposer_pose}")
    
    # Try to visualize using direct parameters
    print("\nVisualizing with direct parameters...")
    try:
        # Initialize SMPL-X model
        model = smplx.create(model_path='models', model_type='smplx', gender='neutral')
        
        # Prepare parameters
        params = {}
        for key, value in data.items():
            if key in ['betas', 'body_pose', 'global_orient', 'transl', 'left_hand_pose', 
                       'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                params[key] = torch.tensor(value)
        
        # Forward pass
        output = model(**params)
        
        # Save direct visualization
        vertices = output.vertices[0].detach().numpy()
        faces = model.faces
        
        # Create mesh and save
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_path = os.path.join(output_dir, "direct_params.obj")
        mesh.export(mesh_path)
        print(f"Saved direct parameter mesh to: {mesh_path}")
        
        # Create 3D visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[::100, 0], vertices[::100, 1], vertices[::100, 2], c='r', marker='o')
        plt.title("Direct Parameter Visualization")
        plt.savefig(os.path.join(output_dir, "direct_params.png"))
        plt.close()
    except Exception as e:
        print(f"Error in direct visualization: {e}")
    
    # Try to visualize using VPoser if available
    if has_vposer_pose and has_vposer and vposer_path:
        print("\nVisualizing with VPoser decoding...")
        try:
            # Load VPoser model
            vposer, _ = load_vposer(vposer_path, vp_model='snapshot')
            
            # Decode VPoser parameters
            pose_embedding = torch.tensor(data['body_pose'], dtype=torch.float32)
            with torch.no_grad():
                body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)
            
            # Initialize SMPL-X model
            model = smplx.create(model_path='models', model_type='smplx', gender='neutral')
            
            # Prepare parameters
            params = {}
            for key, value in data.items():
                if key in ['betas', 'global_orient', 'transl', 'left_hand_pose', 
                           'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                    params[key] = torch.tensor(value)
            
            # Add decoded body_pose
            params['body_pose'] = body_pose
            
            # Forward pass
            output = model(**params)
            
            # Save visualization
            vertices = output.vertices[0].detach().numpy()
            faces = model.faces
            
            # Create mesh and save
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh_path = os.path.join(output_dir, "vposer_decoded.obj")
            mesh.export(mesh_path)
            print(f"Saved VPoser decoded mesh to: {mesh_path}")
            
            # Create 3D visualization
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(vertices[::100, 0], vertices[::100, 1], vertices[::100, 2], c='b', marker='o')
            plt.title("VPoser Decoded Visualization")
            plt.savefig(os.path.join(output_dir, "vposer_decoded.png"))
            plt.close()
        except Exception as e:
            print(f"Error in VPoser visualization: {e}")
    
    # Create T-pose for reference
    print("\nCreating T-pose reference...")
    try:
        # Initialize SMPL-X model
        model = smplx.create(model_path='models', model_type='smplx', gender='neutral')
        
        # Create zero pose
        params = {
            'betas': torch.zeros((1, 10)),
            'global_orient': torch.zeros((1, 3)),
            'body_pose': torch.zeros((1, 21 * 3))  # T-pose
        }
        
        # Forward pass
        output = model(**params)
        
        # Save visualization
        vertices = output.vertices[0].detach().numpy()
        faces = model.faces
        
        # Create mesh and save
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_path = os.path.join(output_dir, "tpose_reference.obj")
        mesh.export(mesh_path)
        print(f"Saved T-pose reference mesh to: {mesh_path}")
        
        # Create 3D visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[::100, 0], vertices[::100, 1], vertices[::100, 2], c='g', marker='o')
        plt.title("T-pose Reference")
        plt.savefig(os.path.join(output_dir, "tpose_reference.png"))
        plt.close()
        
        # Also create a T-pose PKL file for testing
        tpose_data = {}
        for key in data.keys():
            if key == 'body_pose':
                tpose_data[key] = np.zeros((1, 63), dtype=np.float32)  # 21*3=63
            elif key == 'global_orient':
                tpose_data[key] = np.zeros((1, 3), dtype=np.float32)
            else:
                tpose_data[key] = data[key]  # Keep other parameters the same
        
        tpose_pkl_path = os.path.join(output_dir, "tpose_test.pkl")
        with open(tpose_pkl_path, 'wb') as f:
            pickle.dump(tpose_data, f)
        print(f"Created T-pose test PKL file: {tpose_pkl_path}")
    except Exception as e:
        print(f"Error creating T-pose reference: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze SMPL/SMPL-X parameter PKL file")
    parser.add_argument("pkl_path", type=str, help="Path to the PKL file")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Directory to save visualizations")
    parser.add_argument("--vposer_path", type=str, default=None,
                       help="Path to VPoser model checkpoint")
    
    args = parser.parse_args()
    analyze_pkl(args.pkl_path, args.output_dir, args.vposer_path)

if __name__ == "__main__":
    main()