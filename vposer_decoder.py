#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VPoser Decoder Module

This module provides functionality to decode VPoser latent representations (32-dim)
into SMPL-X compatible pose parameters (63-dim).
"""

import os
import torch
import numpy as np
import pickle
import trimesh
import smplx
import argparse
from typing import Dict, Tuple, Optional, Union, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VPoserDecoder:
    """
    A class for decoding VPoser latent codes to SMPL-X pose parameters
    and applying them to 3D human models.
    """
    
    def __init__(
        self, 
        vposer_path: str, 
        smplx_model_path: str, 
        gender: str = 'neutral',
        use_cuda: bool = True,
        optimize_pose: bool = True,
        debug_mode: bool = False
    ):
        """
        Initialize the VPoser decoder

        Args:
            vposer_path: Path to the VPoser model directory
            smplx_model_path: Path to the SMPL-X model directory
            gender: Gender for the SMPL-X model ('neutral', 'male', or 'female')
            use_cuda: Whether to use CUDA if available
            optimize_pose: Whether to optimize poses for naturalness
            debug_mode: Whether to enable debug mode with extra logging
        """
        self.vposer_path = vposer_path
        self.smplx_model_path = smplx_model_path
        self.gender = gender
        self.debug_mode = debug_mode
        self.optimize_pose = optimize_pose
        
        # Set device
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load VPoser model
        self.init_vposer()
        
        # Initialize SMPL-X model
        self.init_smplx_model()
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), 'output', 'posed_meshes')
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.debug_mode:
            self.debug_dir = os.path.join(self.output_dir, 'debug')
            os.makedirs(self.debug_dir, exist_ok=True)
    
    def init_vposer(self):
        """
        Initialize the VPoser model
        """
        try:
            # Try to import the VPoser module
            from human_body_prior.tools.model_loader import load_model
            from human_body_prior.models.vposer_model import VPoser
            
            logger.info(f"Loading VPoser model from {self.vposer_path}")
            self.vposer, _ = load_model(
                self.vposer_path, 
                model_code=VPoser,
                remove_words_in_model_weights='vp_model.',
                disable_grad=True
            )
            self.vposer = self.vposer.to(self.device)
            self.vposer.eval()  # Set to evaluation mode
            logger.info("VPoser model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import VPoser modules: {e}")
            try:
                # Alternative method using the body_model_vposer
                from human_body_prior.tools.model_loader import load_vposer
                
                logger.info(f"Attempting to load VPoser with alternate method")
                self.vposer, _ = load_vposer(self.vposer_path, vp_model='snapshot')
                self.vposer = self.vposer.to(self.device)
                self.vposer.eval()
                logger.info("VPoser model loaded using alternate method")
                
            except Exception as e2:
                logger.error(f"Failed to load VPoser with either method: {e2}")
                raise RuntimeError("Could not initialize VPoser model")
    
    def init_smplx_model(self):
        """
        Initialize the SMPL-X model
        """
        try:
            # Get the path to the SMPL-X model based on gender
            model_file = os.path.join(self.smplx_model_path, f'SMPLX_{self.gender.upper()}.npz')
            if not os.path.exists(model_file):
                model_file = os.path.join(self.smplx_model_path, f'SMPLX_{self.gender.upper()}.pkl')
            
            if not os.path.exists(model_file):
                logger.warning(f"Could not find specific gender model: {model_file}")
                # Try a more generic approach
                model_file = self.smplx_model_path
            
            logger.info(f"Loading SMPL-X model from {model_file}")
            self.body_model = smplx.create(
                model_file,
                model_type='smplx',
                gender=self.gender,
                use_pca=False,
                num_pca_comps=12,
                batch_size=1,
                device=self.device
            )
            logger.info("SMPL-X model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SMPL-X model: {e}")
            raise RuntimeError("Could not initialize SMPL-X model")
    
    def decode_vposer(self, latent_code: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Decode VPoser latent code to SMPL-X pose parameters

        Args:
            latent_code: VPoser latent code (32-dim)
            
        Returns:
            Decoded pose parameters (63-dim)
        """
        if self.debug_mode:
            logger.info(f"Decoding VPoser latent code with shape: {latent_code.shape}")
        
        # Convert to tensor if necessary
        if isinstance(latent_code, np.ndarray):
            latent_code = torch.tensor(latent_code, dtype=torch.float32, device=self.device)
        else:
            latent_code = latent_code.to(device=self.device, dtype=torch.float32)
        
        # Make sure input is properly shaped
        if latent_code.dim() == 1:
            latent_code = latent_code.unsqueeze(0)
            
        # Ensure latent code has the right dimensions
        assert latent_code.shape[1] == 32, f"Expected latent code dimension to be 32, got {latent_code.shape[1]}"
            
        with torch.no_grad():
            try:
                # Try decoding with standard method first
                pose_body = self.vposer.decode(latent_code, output_type='aa')['pose_body']
                
                # Reshape to 63-dim if needed
                if pose_body.shape[-1] != 63:
                    pose_body = pose_body.reshape(-1, 63)
                    
                if self.debug_mode:
                    logger.info(f"Decoded pose body shape: {pose_body.shape}")
                
                return pose_body
                
            except Exception as e:
                logger.error(f"Error in standard decoding: {e}")
                try:
                    # Alternative method
                    pose_body = self.vposer.decode(latent_code)['pose_body'].contiguous().view(-1, 63)
                    return pose_body
                except Exception as e2:
                    logger.error(f"Both decoding methods failed: {e2}")
                    raise RuntimeError("Failed to decode VPoser latent code")
    
    def optimize_body_pose(
        self, 
        body_pose: torch.Tensor, 
        global_orient: Optional[torch.Tensor] = None,
        betas: Optional[torch.Tensor] = None,
        iterations: int = 50
    ) -> torch.Tensor:
        """
        Optimize body pose to reduce artifacts and ensure natural poses

        Args:
            body_pose: Initial body pose (63-dim)
            global_orient: Global orientation (3-dim)
            betas: Shape parameters (10-dim)
            iterations: Number of optimization iterations
            
        Returns:
            Optimized body pose (63-dim)
        """
        if not self.optimize_pose:
            return body_pose
            
        logger.info("Optimizing body pose...")
        
        # Clone inputs to avoid modifying the originals
        body_pose = body_pose.clone().detach().requires_grad_(True)
        
        if global_orient is not None:
            global_orient = global_orient.clone().detach().requires_grad_(True)
            params = [body_pose, global_orient]
        else:
            params = [body_pose]
            
        # Setup optimizer
        optimizer = torch.optim.Adam(params, lr=0.01)
        
        # Define joint limits
        joint_limits_min = torch.ones_like(body_pose) * -1.7  # ~ -100 degrees
        joint_limits_max = torch.ones_like(body_pose) * 1.7   # ~ 100 degrees
        
        # Reference to original values
        orig_body_pose = body_pose.clone().detach()
        
        # Optimization loop
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass through SMPL-X model
            model_output = self.body_model(
                body_pose=body_pose,
                global_orient=global_orient if global_orient is not None else None,
                betas=betas,
                return_verts=True
            )
            
            # Define losses
            
            # 1. Pose regularization - penalize extreme poses
            pose_loss = torch.mean(body_pose ** 2)
            
            # 2. Joint limits - enforce anatomical constraints
            joint_limit_loss = torch.sum(
                torch.clamp(joint_limits_min - body_pose, min=0) ** 2 +
                torch.clamp(body_pose - joint_limits_max, min=0) ** 2
            )
            
            # 3. Original pose guidance - stay close to original pose
            orig_pose_loss = torch.mean((body_pose - orig_body_pose) ** 2)
            
            # Total loss
            loss = 0.1 * pose_loss + 5.0 * joint_limit_loss + 1.0 * orig_pose_loss
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Apply hard constraints after optimization step
            with torch.no_grad():
                body_pose.data = torch.clamp(body_pose.data, joint_limits_min, joint_limits_max)
                
            if self.debug_mode and i % 10 == 0:
                logger.info(f"Optimization iteration {i}, loss: {loss.item():.4f}")
        
        logger.info("Pose optimization completed")
        return body_pose.detach()
    
    def process_pose_data(self, pkl_path: str, pose_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Process pose data from a pickle file

        Args:
            pkl_path: Path to the pickle file containing pose data
            pose_idx: Index of the pose to process
            
        Returns:
            Dictionary of processed SMPL-X parameters
        """
        logger.info(f"Processing pose data from {pkl_path}")
        
        # Load pickle file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Print data keys for debugging
        if self.debug_mode:
            logger.info(f"Data keys: {list(data.keys())}")
            
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"{key} shape: {value.shape}")
        
        # Extract and process parameters
        params = {}
        
        # Handle body pose (VPoser encoded)
        if 'body_pose' in data and data['body_pose'].shape[1] == 32:
            logger.info("Found VPoser encoded body_pose (32-dim), decoding...")
            vposer_code = data['body_pose'][pose_idx:pose_idx+1]
            body_pose = self.decode_vposer(vposer_code)
            
            # Optimize pose if enabled
            if self.optimize_pose:
                # Handle global orientation
                global_orient = None
                if 'global_orient' in data:
                    global_orient = torch.tensor(
                        data['global_orient'][pose_idx:pose_idx+1], 
                        dtype=torch.float32, 
                        device=self.device
                    )
                
                # Handle betas (shape parameters)
                betas = None
                if 'betas' in data:
                    betas = torch.tensor(
                        data['betas'][pose_idx:pose_idx+1], 
                        dtype=torch.float32, 
                        device=self.device
                    )
                
                body_pose = self.optimize_body_pose(body_pose, global_orient, betas)
                
            params['body_pose'] = body_pose
            
        else:
            logger.error("Expected 32-dim VPoser encoding not found in data")
            return None
            
        # Handle global orientation
        if 'global_orient' in data:
            params['global_orient'] = torch.tensor(
                data['global_orient'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        
        # Handle shape parameters (betas)
        if 'betas' in data:
            params['betas'] = torch.tensor(
                data['betas'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        
        # Handle translation
        if 'transl' in data:
            params['transl'] = torch.tensor(
                data['transl'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        elif 'camera_translation' in data:
            params['transl'] = torch.tensor(
                data['camera_translation'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        
        # Handle expression, if available
        if 'expression' in data:
            params['expression'] = torch.tensor(
                data['expression'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        
        # Handle jaw pose, if available
        if 'jaw_pose' in data:
            params['jaw_pose'] = torch.tensor(
                data['jaw_pose'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        
        # Handle eye poses, if available
        if 'leye_pose' in data:
            params['leye_pose'] = torch.tensor(
                data['leye_pose'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        
        if 'reye_pose' in data:
            params['reye_pose'] = torch.tensor(
                data['reye_pose'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        
        # Handle hand poses, if available
        if 'left_hand_pose' in data:
            params['left_hand_pose'] = torch.tensor(
                data['left_hand_pose'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
        
        if 'right_hand_pose' in data:
            params['right_hand_pose'] = torch.tensor(
                data['right_hand_pose'][pose_idx:pose_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
            
        return params
    
    def generate_mesh(self, params: Dict[str, torch.Tensor]) -> trimesh.Trimesh:
        """
        Generate a 3D mesh using the provided parameters

        Args:
            params: Dictionary of SMPL-X parameters
            
        Returns:
            Trimesh mesh object
        """
        logger.info("Generating 3D mesh from pose parameters...")
        
        with torch.no_grad():
            output = self.body_model(**params)
            
            # Get vertices and faces
            vertices = output.vertices[0].detach().cpu().numpy()
            faces = self.body_model.faces
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            logger.info(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
            return mesh
    
    def save_mesh(self, mesh: trimesh.Trimesh, output_path: str) -> str:
        """
        Save mesh to file

        Args:
            mesh: Trimesh object
            output_path: Path to save the mesh
            
        Returns:
            Path to the saved mesh
        """
        logger.info(f"Saving mesh to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        
        return output_path
    
    def process_pkl_file(self, pkl_path: str, output_path: Optional[str] = None, pose_idx: int = 0) -> str:
        """
        Process a pickle file containing pose data and generate a mesh

        Args:
            pkl_path: Path to the pickle file
            output_path: Path to save the output mesh (optional)
            pose_idx: Index of the pose to process
            
        Returns:
            Path to the saved mesh
        """
        # Generate default output path if not provided
        if output_path is None:
            basename = os.path.basename(pkl_path).replace('.pkl', '')
            output_path = os.path.join(self.output_dir, f"{basename}_posed_{pose_idx:03d}.obj")
        
        # Process pose data
        params = self.process_pose_data(pkl_path, pose_idx)
        if params is None:
            raise ValueError("Failed to process pose data")
        
        # Generate mesh
        mesh = self.generate_mesh(params)
        
        # Save mesh
        output_path = self.save_mesh(mesh, output_path)
        
        return output_path
    
    def process_batch(self, pkl_dir: str, output_dir: Optional[str] = None, max_poses: int = 1) -> List[str]:
        """
        Process multiple pickle files in a directory

        Args:
            pkl_dir: Directory containing pickle files
            output_dir: Directory to save output meshes (optional)
            max_poses: Maximum number of poses to process per file
            
        Returns:
            List of paths to the saved meshes
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all pickle files in the directory
        pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
        
        output_paths = []
        for pkl_file in pkl_files:
            pkl_path = os.path.join(pkl_dir, pkl_file)
            
            for i in range(max_poses):
                try:
                    basename = os.path.basename(pkl_file).replace('.pkl', '')
                    output_path = os.path.join(output_dir, f"{basename}_posed_{i:03d}.obj")
                    result_path = self.process_pkl_file(pkl_path, output_path, i)
                    output_paths.append(result_path)
                except Exception as e:
                    logger.error(f"Error processing pose {i} from {pkl_file}: {e}")
                    break
        
        return output_paths

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="VPoser decoder for SMPL-X models")
    
    parser.add_argument("--pkl_path", type=str, required=True,
                      help="Path to the pickle file containing pose data")
    parser.add_argument("--vposer_path", type=str, default="vposer_v1_0",
                      help="Path to the VPoser model directory")
    parser.add_argument("--smplx_path", type=str, default="models/smplx",
                      help="Path to the SMPL-X model directory")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"],
                      help="Gender for the SMPL-X model")
    parser.add_argument("--output_path", type=str, default=None,
                      help="Path to save the output mesh (optional)")
    parser.add_argument("--pose_idx", type=int, default=0,
                      help="Index of the pose to process")
    parser.add_argument("--optimize", action="store_true", default=True,
                      help="Optimize poses for naturalness")
    parser.add_argument("--no-optimize", action="store_false", dest="optimize",
                      help="Disable pose optimization")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with extra logging")
    parser.add_argument("--use_cpu", action="store_true",
                      help="Force CPU usage even if CUDA is available")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Initialize decoder
    decoder = VPoserDecoder(
        vposer_path=args.vposer_path,
        smplx_model_path=args.smplx_path,
        gender=args.gender,
        use_cuda=not args.use_cpu,
        optimize_pose=args.optimize,
        debug_mode=args.debug
    )
    
    # Process pose data
    output_path = decoder.process_pkl_file(
        pkl_path=args.pkl_path,
        output_path=args.output_path,
        pose_idx=args.pose_idx
    )
    
    logger.info(f"Successfully processed pose data and saved mesh to {output_path}")
    
if __name__ == "__main__":
    main()