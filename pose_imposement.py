import os
import torch
import numpy as np
import pickle
import trimesh
import smplx  # Direct import of smplx package
from argparse import ArgumentParser

class PoseImposement:
    """
    PoseImposement class for applying poses to 3D models (Stage 4)
    Uses SMPL/SMPL-X model to apply realistic poses to human meshes
    """
    def __init__(self, mesh_path, pose_data_path, smpl_model_path=None, gender='neutral'):
        """
        Initialize the PoseImposement class
        
        Args:
            mesh_path: Path to the input mesh (.obj file)
            pose_data_path: Path to the pose data (.pkl file)
            smpl_model_path: Path to the SMPL model directory
            gender: Gender for the SMPL model (neutral, male, female)
        """
        self.mesh_path = mesh_path
        self.pose_data_path = pose_data_path
        
        # Set default SMPL model path if not provided
        if smpl_model_path is None:
            self.smpl_model_path = 'models'  # Default model directory
        else:
            self.smpl_model_path = smpl_model_path
            
        self.gender = gender
        
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(mesh_path)), 
                                     "posed_meshes")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Initialize SMPL model
        self.initialize_smpl_model()
    
    def load_data(self):
        """Load mesh and pose data from files"""
        print(f"Loading mesh from {self.mesh_path}")
        self.mesh = trimesh.load(self.mesh_path)
        
        print(f"Loading pose data from {self.pose_data_path}")
        with open(self.pose_data_path, 'rb') as f:
            self.pose_data = pickle.load(f)
        
        # Print pose data structure for debugging
        print(f"Pose data keys: {self.pose_data.keys() if isinstance(self.pose_data, dict) else 'Not a dictionary'}")
    
    def initialize_smpl_model(self):
        """Initialize the SMPL model for pose application"""
        try:
            # Try to initialize SMPL-X model first
            print(f"Trying to load SMPL-X model from {self.smpl_model_path}")
            self.smpl_model = smplx.create(
                model_path=self.smpl_model_path,
                model_type='smplx',  # Explicitly specify model type
                gender=self.gender,
                use_pca=False,
                create_expression=True,
                create_jaw_pose=True,
                create_leye_pose=True,
                create_reye_pose=True
            )
            self.model_type = 'SMPLX'
            print(f"Initialized SMPL-X model with {self.gender} gender")
        except Exception as e:
            print(f"Error initializing SMPL-X model: {e}")
            try:
                # Fall back to SMPL model if SMPL-X fails
                self.smpl_model = smplx.create(
                    model_path=self.smpl_model_path,
                    model_type='smpl',
                    gender=self.gender
                )
                self.model_type = 'SMPL'
                print(f"Initialized SMPL model with {self.gender} gender")
            except Exception as e2:
                print(f"Error initializing SMPL model: {e2}")
                print("Using trimesh-based transformations as fallback.")
                self.model_type = 'NONE'
    

    def apply_pose(self, pose_idx=0):
        """
        Apply the specified pose to the mesh using SMPL model
        
        Args:
            pose_idx: Index of the pose to apply
        
        Returns:
            posed_mesh: The mesh with the pose applied
        """
        if self.model_type == 'NONE':
            # Fallback to basic rotation if SMPL model initialization failed
            return self.apply_basic_rotation(pose_idx)
        
        # Create torch tensors for SMPL parameters
        betas = torch.tensor(self.pose_data.get('betas', np.zeros((1, 10))), dtype=torch.float32)
        
        # Get global orient (rotation)
        if 'global_orient' in self.pose_data:
            global_orient = torch.tensor(self.pose_data['global_orient'][pose_idx:pose_idx+1], dtype=torch.float32)
        else:
            global_orient = torch.zeros((1, 3), dtype=torch.float32)
        
        # Get body pose parameters and handle possible size mismatch
        if 'body_pose' in self.pose_data:
            body_pose_data = self.pose_data['body_pose'][pose_idx:pose_idx+1]
            print(f"Original body_pose shape: {body_pose_data.shape}")
            
            # Check if our body_pose matches expected SMPL-X size (21 joints * 3)
            expected_size = 21 * 3  # SMPL-X expects 21 joints, each with 3 rotation values
            actual_size = body_pose_data.size
            
            if actual_size < expected_size:
                # If we have less data than needed, pad with zeros
                print(f"Padding body_pose from size {actual_size} to {expected_size}")
                padded_data = np.zeros((1, expected_size))
                padded_data[0, :actual_size] = body_pose_data.flatten()
                body_pose = torch.tensor(padded_data, dtype=torch.float32)
            elif actual_size > expected_size:
                # If we have more data than needed, truncate
                print(f"Truncating body_pose from size {actual_size} to {expected_size}")
                body_pose = torch.tensor(body_pose_data.flatten()[:expected_size].reshape(1, expected_size), dtype=torch.float32)
            else:
                body_pose = torch.tensor(body_pose_data, dtype=torch.float32)
        else:
            # Create zero body pose if not available
            print("No body_pose found, using zeros")
            if self.model_type == 'SMPLX':
                body_pose = torch.zeros((1, 21 * 3), dtype=torch.float32)
            else:
                body_pose = torch.zeros((1, 23 * 3), dtype=torch.float32)
        
        # Get translation
        if 'transl' in self.pose_data:
            transl = torch.tensor(self.pose_data['transl'][pose_idx:pose_idx+1], dtype=torch.float32)
        elif 'camera_translation' in self.pose_data:
            transl = torch.tensor(self.pose_data['camera_translation'][pose_idx:pose_idx+1], dtype=torch.float32)
        else:
            transl = torch.zeros((1, 3), dtype=torch.float32)
        
        # Additional parameters for SMPL-X model
        params = {}
        if self.model_type == 'SMPLX':
            # Handle left hand pose - expected size is 15 joints * 3 = 45
            if 'left_hand_pose' in self.pose_data:
                hand_data = self.pose_data['left_hand_pose'][pose_idx:pose_idx+1]
                expected_hand_size = 15 * 3
                actual_hand_size = hand_data.size
                
                print(f"Original left_hand_pose shape: {hand_data.shape}, size: {actual_hand_size}")
                
                if actual_hand_size < expected_hand_size:
                    print(f"Padding left_hand_pose from {actual_hand_size} to {expected_hand_size}")
                    padded_hand = np.zeros((1, expected_hand_size))
                    padded_hand[0, :actual_hand_size] = hand_data.flatten()
                    params['left_hand_pose'] = torch.tensor(padded_hand, dtype=torch.float32)
                elif actual_hand_size > expected_hand_size:
                    print(f"Truncating left_hand_pose from {actual_hand_size} to {expected_hand_size}")
                    params['left_hand_pose'] = torch.tensor(
                        hand_data.flatten()[:expected_hand_size].reshape(1, expected_hand_size), 
                        dtype=torch.float32)
                else:
                    params['left_hand_pose'] = torch.tensor(hand_data, dtype=torch.float32)
            
            # Handle right hand pose - expected size is 15 joints * 3 = 45
            if 'right_hand_pose' in self.pose_data:
                hand_data = self.pose_data['right_hand_pose'][pose_idx:pose_idx+1]
                expected_hand_size = 15 * 3
                actual_hand_size = hand_data.size
                
                print(f"Original right_hand_pose shape: {hand_data.shape}, size: {actual_hand_size}")
                
                if actual_hand_size < expected_hand_size:
                    print(f"Padding right_hand_pose from {actual_hand_size} to {expected_hand_size}")
                    padded_hand = np.zeros((1, expected_hand_size))
                    padded_hand[0, :actual_hand_size] = hand_data.flatten()
                    params['right_hand_pose'] = torch.tensor(padded_hand, dtype=torch.float32)
                elif actual_hand_size > expected_hand_size:
                    print(f"Truncating right_hand_pose from {actual_hand_size} to {expected_hand_size}")
                    params['right_hand_pose'] = torch.tensor(
                        hand_data.flatten()[:expected_hand_size].reshape(1, expected_hand_size), 
                        dtype=torch.float32)
                else:
                    params['right_hand_pose'] = torch.tensor(hand_data, dtype=torch.float32)
                
            # These parameters don't typically need resizing as they're already the right shape
            if 'jaw_pose' in self.pose_data:
                params['jaw_pose'] = torch.tensor(
                    self.pose_data['jaw_pose'][pose_idx:pose_idx+1], dtype=torch.float32)
                
            if 'leye_pose' in self.pose_data:
                params['leye_pose'] = torch.tensor(
                    self.pose_data['leye_pose'][pose_idx:pose_idx+1], dtype=torch.float32)
                
            if 'reye_pose' in self.pose_data:
                params['reye_pose'] = torch.tensor(
                    self.pose_data['reye_pose'][pose_idx:pose_idx+1], dtype=torch.float32)
                
            if 'expression' in self.pose_data:
                expr_data = self.pose_data['expression'][pose_idx:pose_idx+1]
                # SMPL-X typically expects 10 expression parameters
                expected_expr_size = 10
                actual_expr_size = expr_data.size
                
                print(f"Original expression shape: {expr_data.shape}, size: {actual_expr_size}")
                
                if actual_expr_size < expected_expr_size:
                    print(f"Padding expression from {actual_expr_size} to {expected_expr_size}")
                    padded_expr = np.zeros((1, expected_expr_size))
                    padded_expr[0, :actual_expr_size] = expr_data.flatten()
                    params['expression'] = torch.tensor(padded_expr, dtype=torch.float32)
                elif actual_expr_size > expected_expr_size:
                    print(f"Truncating expression from {actual_expr_size} to {expected_expr_size}")
                    params['expression'] = torch.tensor(
                        expr_data.flatten()[:expected_expr_size].reshape(1, expected_expr_size), 
                        dtype=torch.float32)
                else:
                    params['expression'] = torch.tensor(expr_data, dtype=torch.float32)
        
        # Forward pass through the model
        print(f"Applying {self.model_type} pose parameters to model...")
        try:
            with torch.no_grad():
                output = self.smpl_model.forward(
                    betas=betas,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    transl=transl,
                    **params
                )
            
            # Get vertices and create mesh
            vertices = output.vertices[0].detach().cpu().numpy()
            faces = self.smpl_model.faces
            
            posed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            return posed_mesh
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print("Falling back to basic rotation method")
            return self.apply_basic_rotation(pose_idx)
    
    
    def apply_basic_rotation(self, pose_idx=0):
        """
        Fallback method to apply simple rotation if SMPL model is not available
        
        Args:
            pose_idx: Index of the pose to apply
        
        Returns:
            posed_mesh: The mesh with a basic rotation applied
        """
        # Create a copy of the mesh for posing
        posed_mesh = self.mesh.copy()
        
        # Check if we have global orientation data
        if isinstance(self.pose_data, dict) and 'global_orient' in self.pose_data:
            print("Applying basic rotation using global_orient data")
            
            # Convert global_orient (in axis-angle format) to rotation matrix
            global_orient = self.pose_data['global_orient']
            if isinstance(global_orient, torch.Tensor):
                global_orient = global_orient.detach().cpu().numpy()
                
            # Apply rotation based on global orientation
            angle = np.linalg.norm(global_orient[pose_idx])
            if angle > 0:
                axis = global_orient[pose_idx] / angle
                rotation = trimesh.transformations.rotation_matrix(
                    angle=angle,
                    direction=axis,
                    point=posed_mesh.centroid
                )
                posed_mesh.apply_transform(rotation)
        else:
            print("No global_orient data found, applying default rotation")
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
        
        for i in range(min(num_poses, len(self.pose_data.get('global_orient', [1])))):
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
    parser = ArgumentParser(description="Apply poses to 3D models using SMPL/SMPL-X models")
    parser.add_argument("--mesh_path", type=str, default="output/meshes/test_image/000.obj",
                        help="Path to the input mesh (.obj file)")
    parser.add_argument("--pose_data_path", type=str, default="output/results/test_image/000.pkl",
                        help="Path to the pose data (.pkl file)")
    parser.add_argument("--smpl_model_path", type=str, default=None,
                        help="Path to the SMPL model directory")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"],
                        help="Gender for the SMPL model")
    parser.add_argument("--num_poses", type=int, default=1,
                        help="Number of poses to process")
    
    args = parser.parse_args()
    
    # Create PoseImposement instance
    pose_imposement = PoseImposement(
        mesh_path=args.mesh_path,
        pose_data_path=args.pose_data_path,
        smpl_model_path=args.smpl_model_path,
        gender=args.gender
    )
    
    # Process poses
    output_paths = pose_imposement.process_poses(num_poses=args.num_poses)
    
    print("Pose imposement completed.")
    return output_paths

if __name__ == "__main__":
    main()