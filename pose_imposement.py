import torch
import numpy as np
import os
import pickle
import trimesh
import pyrender
from argparse import ArgumentParser

class PoseImposement:
    """
    PoseImposement class for applying poses to 3D models (Stage 4)
    """
    def __init__(self, mesh_path, pose_data_path):
        """
        Initialize the PoseImposement class
        
        Args:
            mesh_path: Path to the input mesh (.obj file)
            pose_data_path: Path to the pose data (.pkl file)
        """
        self.mesh_path = mesh_path
        self.pose_data_path = pose_data_path
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(mesh_path)), 
                                     "posed_meshes")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load mesh and pose data
        self.load_data()
    
    def load_data(self):
        """Load mesh and pose data from files"""
        print(f"Loading mesh from {self.mesh_path}")
        self.mesh = trimesh.load(self.mesh_path)
        
        print(f"Loading pose data from {self.pose_data_path}")
        with open(self.pose_data_path, 'rb') as f:
            self.pose_data = pickle.load(f)
        
        # Print pose data structure for debugging
        print(f"Pose data keys: {self.pose_data.keys() if isinstance(self.pose_data, dict) else 'Not a dictionary'}")
    
    def apply_pose(self, pose_idx=0):
        """
        Apply the specified pose to the mesh
        
        Args:
            pose_idx: Index of the pose to apply
        
        Returns:
            posed_mesh: The mesh with the pose applied
        """
        # Create a copy of the mesh for posing
        posed_mesh = self.mesh.copy()
        
        # Check if we have SMPL-X format data
        if isinstance(self.pose_data, dict) and 'global_orient' in self.pose_data and 'body_pose' in self.pose_data:
            print("Processing SMPL-X format data")
            # For SMPL-X, we'll apply a simple rotation based on global orientation
            if hasattr(self.pose_data['global_orient'], 'shape') and len(self.pose_data['global_orient'].shape) > 0:
                # Convert global_orient (in axis-angle format) to rotation matrix
                import numpy as np
                
                # Extract the global orientation
                global_orient = self.pose_data['global_orient']
                if isinstance(global_orient, torch.Tensor):
                    global_orient = global_orient.detach().cpu().numpy()
                    
                # Apply a rotation based on the global orientation
                angle = np.linalg.norm(global_orient[pose_idx])
                if angle > 0:
                    axis = global_orient[pose_idx] / angle
                    rotation = trimesh.transformations.rotation_matrix(
                        angle=angle,
                        direction=axis,
                        point=posed_mesh.centroid
                    )
                    posed_mesh.apply_transform(rotation)
                
                # Apply additional rotation to show different view
                extra_rotation = trimesh.transformations.rotation_matrix(
                    angle=np.pi/4,  # 45 degrees
                    direction=[0, 1, 0],  # Rotate around Y axis
                    point=posed_mesh.centroid
                )
                posed_mesh.apply_transform(extra_rotation)
            else:
                print("Global orientation data not available, applying default rotation")
                rotation = trimesh.transformations.rotation_matrix(
                    angle=np.pi/4,  # 45 degrees
                    direction=[0, 1, 0],  # Rotate around Y axis
                    point=posed_mesh.centroid
                )
                posed_mesh.apply_transform(rotation)
        else:
            print("Using default rotation as pose data format is not recognized")
            # Apply a default rotation
            rotation = trimesh.transformations.rotation_matrix(
                angle=np.pi/4,  # 45 degrees
                direction=[0, 1, 0],  # Rotate around Y axis
                point=posed_mesh.centroid
            )
            posed_mesh.apply_transform(rotation)
        
        return posed_mesh
    


    
    def save_posed_mesh(self, posed_mesh, output_filename):
        """
        Save the posed mesh to file
        
        Args:
            posed_mesh: The posed mesh to save
            output_filename: Output filename
        """
        output_path = os.path.join(self.output_dir, output_filename)
        posed_mesh.export(output_path)
        print(f"Saved posed mesh to {output_path}")
        return output_path
    
    def process_poses(self, num_poses=1):
        """
        Process multiple poses and save the results
        
        Args:
            num_poses: Number of poses to process
        
        Returns:
            output_paths: List of paths to the saved posed meshes
        """
        output_paths = []
        
        for i in range(num_poses):
            print(f"Processing pose {i+1}/{num_poses}")
            posed_mesh = self.apply_pose(pose_idx=i)
            
            # Generate output filename
            basename = os.path.basename(self.mesh_path)
            name, ext = os.path.splitext(basename)
            output_filename = f"{name}_posed_{i:03d}{ext}"
            
            # Save posed mesh
            output_path = self.save_posed_mesh(posed_mesh, output_filename)
            output_paths.append(output_path)
        
        return output_paths

def main():
    """Main function to run pose imposement"""
    parser = ArgumentParser(description="Apply poses to 3D models")
    parser.add_argument("--mesh_path", type=str, default="output/meshes/test_image/000.obj",
                        help="Path to the input mesh (.obj file)")
    parser.add_argument("--pose_data_path", type=str, default="output/results/test_image/000.pkl",
                        help="Path to the pose data (.pkl file)")
    parser.add_argument("--num_poses", type=int, default=1,
                        help="Number of poses to process")
    
    args = parser.parse_args()
    
    # Create PoseImposement instance
    pose_imposement = PoseImposement(
        mesh_path=args.mesh_path,
        pose_data_path=args.pose_data_path
    )
    
    # Process poses
    output_paths = pose_imposement.process_poses(num_poses=args.num_poses)
    
    print("Pose imposement completed.")
    return output_paths

if __name__ == "__main__":
    main()