import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import trimesh
import smplx  # Direct import of smplx package
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import copy
from mpl_toolkits.mplot3d import Axes3D

# Try to import VPoser
try:
    from human_body_prior.tools.model_loader import load_vposer
    has_vposer = True
    print("VPoser module found!")
except ImportError:
    has_vposer = False
    print("VPoser module not found. Using fallback methods for pose handling.")

class PoseImposement:
    """
    PoseImposement class for applying poses to 3D models (Stage 4)
    Uses SMPL/SMPL-X model to apply realistic poses to human meshes
    Implements support for VPoser-encoded parameters and fallback solutions
    """
    def __init__(self, mesh_path, pose_data_path, smpl_model_path=None, gender='neutral', 
                 optimize_pose=True, use_vposer=True, vposer_path=None, 
                 use_tpose_fallback=False, debug_mode=False):
        """
        Initialize the PoseImposement class
        
        Args:
            mesh_path: Path to the input mesh (.obj file)
            pose_data_path: Path to the pose data (.pkl file)
            smpl_model_path: Path to the SMPL model directory
            gender: Gender for the SMPL model (neutral, male, female)
            optimize_pose: Whether to optimize the pose
            use_vposer: Whether to use VPoser for decoding pose parameters
            vposer_path: Path to VPoser model checkpoint
            use_tpose_fallback: Use T-pose if other methods fail
            debug_mode: Whether to enable debug visualization
        """
        self.mesh_path = mesh_path
        self.pose_data_path = pose_data_path
        self.optimize_pose = optimize_pose
        self.use_vposer = use_vposer and has_vposer
        self.vposer_path = vposer_path
        self.use_tpose_fallback = use_tpose_fallback
        self.debug_mode = debug_mode
        
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
        
        # Create debug directory if needed
        if self.debug_mode:
            self.debug_dir = os.path.join(self.output_dir, "debug")
            os.makedirs(self.debug_dir, exist_ok=True)
        
        # Device setup - try to use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize VPoser if available
        self.vposer = None
        if self.use_vposer:
            self.initialize_vposer()
        
        # Load data
        self.load_data()
        
        # Initialize SMPL model
        self.initialize_smpl_model()
        
        # Create reference T-pose
        self.create_reference_pose()
        
        # Print data and model information for debugging
        self.print_info()
    
    def initialize_vposer(self):
        """Initialize VPoser model for pose decoding"""
        if not has_vposer:
            print("VPoser module not found. Cannot initialize VPoser.")
            return False
        
        if self.vposer_path is None:
            print("VPoser path not provided. Cannot initialize VPoser.")
            return False
        
        try:
            print(f"Initializing VPoser from {self.vposer_path}")
            self.vposer, _ = load_vposer(self.vposer_path, vp_model='snapshot')
            self.vposer = self.vposer.to(device=self.device)
            self.vposer.eval()  # Set to evaluation mode
            print("VPoser initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing VPoser: {e}")
            self.vposer = None
            return False
    
    def print_info(self):
        """Print information about the loaded data and model for debugging"""
        if not self.debug_mode:
            return
            
        print("\n" + "=" * 50)
        print("DATA AND MODEL INFORMATION")
        print("=" * 50)
        
        if 'betas' in self.pose_data:
            print(f"Betas shape: {self.pose_data['betas'].shape}")
        
        if 'body_pose' in self.pose_data:
            print(f"Body pose shape: {self.pose_data['body_pose'].shape}")
            body_pose = self.pose_data['body_pose']
            print(f"Body pose stats: min={np.min(body_pose)}, max={np.max(body_pose)}, mean={np.mean(body_pose)}")
            
            # Detect if this is likely VPoser encoded
            if body_pose.shape[1] == 32:
                print("Detected VPoser encoded body_pose (32 dimensions)")
            elif body_pose.shape[1] % 3 == 0:
                print(f"Detected standard body_pose ({body_pose.shape[1]} dimensions = {body_pose.shape[1]//3} joints)")
            else:
                print(f"WARNING: Unusual body_pose dimension: {body_pose.shape[1]}")
        
        if 'global_orient' in self.pose_data:
            print(f"Global orient shape: {self.pose_data['global_orient'].shape}")
            global_orient = self.pose_data['global_orient']
            print(f"Global orient stats: min={np.min(global_orient)}, max={np.max(global_orient)}, mean={np.mean(global_orient)}")
        
        print(f"Model type: {self.model_type}")
        print(f"Using VPoser: {self.vposer is not None}")
        print("=" * 50 + "\n")
    
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
            # 指定正确的模型文件和路径
            print(f"Trying to load SMPL-X model from {self.smpl_model_path}")
            
            # 关键修改：将model_path设置为基础目录，而不是smplx目录
            base_dir = os.path.dirname(self.smpl_model_path)
            
            self.smpl_model = smplx.create(
                model_path=base_dir,  # 使用父目录
                model_type='smplx',
                model_folder=os.path.basename(self.smpl_model_path),  # 使用smplx作为子文件夹名
                gender=self.gender,
                use_pca=False,
                create_expression=True,
                create_jaw_pose=True,
                create_leye_pose=True,
                create_reye_pose=True,
                batch_size=1,
                device=self.device
            )
            self.model_type = 'SMPLX'
            print(f"Initialized SMPL-X model with {self.gender} gender")
        except Exception as e:
            print(f"Error initializing SMPL-X model: {e}")
            try:
                # 同样修改SMPL模型的加载方式
                base_dir = os.path.dirname(self.smpl_model_path)
                
                self.smpl_model = smplx.create(
                    model_path=base_dir,
                    model_type='smpl',
                    model_folder=os.path.basename(self.smpl_model_path),
                    gender=self.gender,
                    batch_size=1,
                    device=self.device
                )
                self.model_type = 'SMPL'
                print(f"Initialized SMPL model with {self.gender} gender")
            except Exception as e2:
                print(f"Error initializing SMPL model: {e2}")
                print("Using trimesh-based transformations as fallback.")
                self.model_type = 'NONE'


    def create_reference_pose(self):
        """Create reference T-pose for fallback and visualization"""
        # Create zero pose for SMPL (T-pose)
        if self.model_type == 'SMPLX':
            self.ref_body_pose = torch.zeros((1, 21 * 3), dtype=torch.float32, device=self.device)
            self.ref_global_orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        else:  # SMPL
            self.ref_body_pose = torch.zeros((1, 23 * 3), dtype=torch.float32, device=self.device)
            self.ref_global_orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        
        # Generate reference pose visualization for debugging
        if self.debug_mode:
            try:
                with torch.no_grad():
                    output = self.smpl_model(
                        body_pose=self.ref_body_pose,
                        global_orient=self.ref_global_orient,
                        return_verts=True
                    )
                    
                    # Save visualization
                    vertices = output.vertices[0].detach().cpu().numpy()
                    
                    # Plot reference pose
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(vertices[::100, 0], vertices[::100, 1], vertices[::100, 2], 
                               c='b', marker='o', alpha=0.5)
                    plt.title("Reference T-Pose")
                    plt.savefig(os.path.join(self.debug_dir, "reference_tpose.png"))
                    plt.close()
                    
                    print("Created reference T-pose for fallback")
            except Exception as e:
                print(f"Error creating reference pose visualization: {e}")
        
    def decode_vposer(self, pose_embedding):
        """
        Decode VPoser parameters to body pose
        
        Args:
            pose_embedding: VPoser latent code (32 dimensions)
            
        Returns:
            Decoded body pose parameters
        """
        if self.vposer is None:
            print("VPoser not initialized. Cannot decode parameters.")
            return None
        
        try:
            with torch.no_grad():
                # 确保输入是PyTorch张量
                pose_embedding_tensor = torch.tensor(pose_embedding, dtype=torch.float32, device=self.device)
                
                # 尝试直接解码，但不要调用output_type='aa'，避免使用torchgeometry
                pose_body = None
                try:
                    # 方法1：直接从VPoser获取矩阵旋转表示
                    result = self.vposer.decode(pose_embedding_tensor)  # 默认返回矩阵旋转表示
                    
                    # 如果结果是字典，查找pose_body或pose_matrot
                    if isinstance(result, dict):
                        if 'pose_body' in result:
                            pose_matrot = result['pose_body']
                        elif 'pose_matrot' in result:
                            pose_matrot = result['pose_matrot']
                    else:
                        pose_matrot = result
                    
                    # 手动将矩阵旋转表示转换为轴角表示
                    if pose_matrot is not None:
                        # 重塑为合适的形状，假设输出是矩阵旋转
                        if pose_matrot.shape[-1] == 9:
                            pose_matrot = pose_matrot.reshape(-1, 21, 3, 3)  # 假设是21个关节
                        elif pose_matrot.shape[-1] != 3 and pose_matrot.dim() > 2:
                            pose_matrot = pose_matrot.reshape(-1, 21, 3, 3)  # 调整形状
                        
                        # 初始化轴角表示
                        pose_body = torch.zeros((1, 63), dtype=torch.float32, device=self.device)
                        
                        # 手动将每个关节的旋转矩阵转换为轴角
                        for j in range(pose_matrot.shape[1]):  # 遍历每个关节
                            # 使用Rodrigues公式或其他方法将旋转矩阵转换为轴角
                            # 这里简化处理，使用OpenCV或SciPy
                            try:
                                import cv2
                                rot_mat = pose_matrot[0, j].cpu().numpy()
                                # OpenCV的Rodrigues函数可以将3x3旋转矩阵转换为轴角表示
                                ax_angle, _ = cv2.Rodrigues(rot_mat)
                                ax_angle = ax_angle.flatten()
                                # 填入对应位置
                                pose_body[0, j*3:(j+1)*3] = torch.from_numpy(ax_angle).to(self.device)
                            except ImportError:
                                try:
                                    from scipy.spatial.transform import Rotation
                                    rot_mat = pose_matrot[0, j].cpu().numpy()
                                    r = Rotation.from_matrix(rot_mat)
                                    ax_angle = r.as_rotvec()
                                    pose_body[0, j*3:(j+1)*3] = torch.from_numpy(ax_angle).to(self.device)
                                except ImportError:
                                    print("Neither OpenCV nor SciPy available for rotation conversion")
                                    # 使用近似值
                                    pose_body[0, j*3:(j+1)*3] = torch.zeros(3, dtype=torch.float32, device=self.device)
                
                except Exception as e:
                    print(f"Error in matrix rotation conversion: {e}")
                    pose_body = None
                
                # 如果以上方法失败，尝试其他方法
                if pose_body is None:
                    print("Trying alternative decoding method...")
                    try:
                        # 方法2：直接构造一个从32维到63维的简单映射
                        # 这只是一个近似方法，结果可能不如直接解码准确
                        # 从32维扩展到63维（21个关节*3）
                        pose_body = torch.zeros((1, 63), dtype=torch.float32, device=self.device)
                        
                        # 映射前21个维度（如果有）到主要关节
                        min_dim = min(pose_embedding_tensor.shape[1], 21)
                        for i in range(min_dim):
                            # 为每个主关节设置近似轴角值
                            # 这是一个简化的线性映射
                            pose_body[0, i*3] = pose_embedding_tensor[0, i] * 0.2  # x轴
                            if i+1 < pose_embedding_tensor.shape[1]:
                                pose_body[0, i*3+1] = pose_embedding_tensor[0, i+1] * 0.2  # y轴
                            if i+2 < pose_embedding_tensor.shape[1]:
                                pose_body[0, i*3+2] = pose_embedding_tensor[0, i+2] * 0.2  # z轴
                        
                        print("Generated approximate pose parameters through direct mapping")
                    except Exception as e:
                        print(f"Error in alternative decoding: {e}")
                        # 最后的备选方案：使用零姿势
                        pose_body = torch.zeros((1, 63), dtype=torch.float32, device=self.device)
                        print("Using zero pose as fallback")
                
                # 返回最终结果
                print(f"Final body pose shape: {pose_body.shape}")
                return pose_body
                
        except Exception as e:
            print(f"Unexpected error in decode_vposer: {e}")
            return torch.zeros((1, 63), dtype=torch.float32, device=self.device)
        




    def optimize_body_pose(self, global_orient, body_pose, betas=None, transl=None, iterations=50):
        """
        Optimize body pose to reduce artifacts
        
        Args:
            global_orient: Initial global orientation
            body_pose: Initial body pose
            betas: Shape parameters
            transl: Translation vector
            iterations: Number of optimization iterations
                
        Returns:
            Optimized global_orient and body_pose
        """
        # Skip optimization if disabled
        if not self.optimize_pose:
            return global_orient, body_pose
                
        # Convert inputs to PyTorch tensors and move to device
        if not isinstance(global_orient, torch.Tensor):
            global_orient = torch.tensor(global_orient, dtype=torch.float32, device=self.device)
        if not isinstance(body_pose, torch.Tensor):
            body_pose = torch.tensor(body_pose, dtype=torch.float32, device=self.device)
                
        # Clone the tensors to avoid modifying the originals
        global_orient = global_orient.clone().detach().requires_grad_(True)
        body_pose = body_pose.clone().detach().requires_grad_(True)
            
        # Optional betas and transl handling
        if betas is not None and not isinstance(betas, torch.Tensor):
            betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        if transl is not None and not isinstance(transl, torch.Tensor):
            transl = torch.tensor(transl, dtype=torch.float32, device=self.device)
                
        # Setup optimizer with parameters to optimize
        params = [global_orient, body_pose]
        optimizer = torch.optim.Adam(params, lr=0.01)
            
        # Reference to original values
        orig_global_orient = global_orient.clone().detach()
        orig_body_pose = body_pose.clone().detach()
            
        # Define joint limits for natural human pose
        joint_limits_min = torch.ones_like(body_pose) * -1.7  # ~ -100 degrees
        joint_limits_max = torch.ones_like(body_pose) * 1.7   # ~ 100 degrees
            
        # Simple optimization loop
        for i in range(iterations):
            optimizer.zero_grad()
                
            # Forward pass through SMPL model
            output = self.smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                return_verts=True
            )
                
            # Define losses
                
            # 1. Pose regularization
            pose_loss = torch.mean(body_pose ** 2)
                
            # 2. Joint limits
            joint_limit_loss = torch.sum(
                torch.clamp(joint_limits_min - body_pose, min=0) ** 2 +
                torch.clamp(body_pose - joint_limits_max, min=0) ** 2
            )
                
            # 3. Original pose guidance
            orig_pose_loss = torch.mean((global_orient - orig_global_orient) ** 2) + \
                            torch.mean((body_pose - orig_body_pose) ** 2)
                
            # 4. Guidance towards T-pose (with small weight)
            tpose_loss = 0.01 * torch.mean((body_pose - self.ref_body_pose) ** 2)
                
            # Total loss
            loss = 0.1 * pose_loss + 5.0 * joint_limit_loss + 1.0 * orig_pose_loss + tpose_loss
                
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
                
            # Apply hard constraints after optimization step
            with torch.no_grad():
                # Enforce joint limits by clipping values
                body_pose.data = torch.clamp(body_pose.data, joint_limits_min, joint_limits_max)
                
            if self.debug_mode and i % 10 == 0:
                print(f"Optimization iteration {i}, loss: {loss.item():.4f}")
            
        # Return optimized parameters (detached from computation graph)
        return global_orient.detach(), body_pose.detach()


    def convert_pose_params(self, pose_idx=0, align_position=False):
        """
        Convert and validate pose parameters from the pose data
        Handles VPoser encoded parameters if detected
        
        Args:
            pose_idx: Index of the pose to process
            align_position: Whether to align model position
            
        Returns:
            Dictionary of validated pose parameters
        """
        # Prepare parameters dict
        params = {}
        
        # Extract betas (shape parameters)
        if 'betas' in self.pose_data:
            betas = torch.tensor(self.pose_data['betas'], dtype=torch.float32, device=self.device)
            params['betas'] = betas
        else:
            params['betas'] = torch.zeros((1, 10), dtype=torch.float32, device=self.device)
        
        # Handle global orientation
        if 'global_orient' in self.pose_data:
            global_orient = torch.tensor(self.pose_data['global_orient'][pose_idx:pose_idx+1], 
                                        dtype=torch.float32, device=self.device)
        else:
            # Try to extract from thetas if available
            if 'thetas' in self.pose_data:
                global_orient = torch.tensor(self.pose_data['thetas'][pose_idx:pose_idx+1, :3], 
                                        dtype=torch.float32, device=self.device)
            else:
                global_orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
                
        params['global_orient'] = global_orient
        
        # Handle body pose - check for VPoser encoded parameters
        if 'body_pose' in self.pose_data:
            body_pose_data = self.pose_data['body_pose'][pose_idx:pose_idx+1]
            
            # Check if this is likely VPoser encoded (32 dimensions)
            if body_pose_data.shape[1] == 32 and self.vposer is not None:
                print("Detected VPoser encoded body_pose. Decoding...")
                decoded_body_pose = self.decode_vposer(body_pose_data)
                if decoded_body_pose is not None:
                    body_pose = decoded_body_pose
                    print(f"Successfully decoded VPoser parameters to shape {body_pose.shape}")
                else:
                    print("VPoser decoding failed. Using fallback method.")
                    # Fallback to standard conversion
                    body_pose = self.convert_standard_body_pose(body_pose_data)
            else:
                # Standard conversion
                body_pose = self.convert_standard_body_pose(body_pose_data)
        elif 'thetas' in self.pose_data and self.pose_data['thetas'].shape[1] > 3:
            body_pose_data = self.pose_data['thetas'][pose_idx:pose_idx+1, 3:]
            body_pose = self.convert_standard_body_pose(body_pose_data)
        elif self.use_tpose_fallback:
            print("No body pose data found. Using T-pose.")
            body_pose = self.ref_body_pose  # Use reference T-pose
        else:
            print("WARNING: No body pose data found and T-pose fallback disabled.")
            body_pose = torch.zeros((1, 21 * 3), dtype=torch.float32, device=self.device)
        
        # Optional pose optimization
        if self.optimize_pose:
            print("Optimizing pose parameters...")
            global_orient, body_pose = self.optimize_body_pose(
                global_orient, 
                body_pose, 
                betas=params['betas']
            )
            
        params['body_pose'] = body_pose
        
        # Handle translation
        if 'transl' in self.pose_data:
            transl = torch.tensor(self.pose_data['transl'][pose_idx:pose_idx+1], 
                                dtype=torch.float32, device=self.device)
        elif 'camera_translation' in self.pose_data:
            # Use camera translation data
            camera_trans = self.pose_data['camera_translation'][pose_idx:pose_idx+1].copy()
            transl = torch.tensor(camera_trans, dtype=torch.float32, device=self.device)
        else:
            transl = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
            
        params['transl'] = transl
        
        # Handle additional SMPL-X specific parameters
        if self.model_type == 'SMPLX':
            # Handle left and right hand poses
            for hand_type in ['left_hand_pose', 'right_hand_pose']:
                if hand_type in self.pose_data:
                    hand_data = self.pose_data[hand_type][pose_idx:pose_idx+1]
                    expected_hand_size = 15 * 3  # 15 joints per hand
                    actual_hand_size = hand_data.size
                    
                    if actual_hand_size < expected_hand_size:
                        print(f"Padding {hand_type} from {actual_hand_size} to {expected_hand_size}")
                        padded_hand = np.zeros((1, expected_hand_size), dtype=np.float32)
                        padded_hand[0, :actual_hand_size] = hand_data.flatten()
                        params[hand_type] = torch.tensor(padded_hand, dtype=torch.float32, device=self.device)
                    else:
                        params[hand_type] = torch.tensor(hand_data, dtype=torch.float32, device=self.device)
            
            # Handle other SMPL-X parameters
            for param_name in ['jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                if param_name in self.pose_data:
                    param_data = self.pose_data[param_name][pose_idx:pose_idx+1]
                    params[param_name] = torch.tensor(param_data, dtype=torch.float32, device=self.device)
        
        # Apply coordinate system transformation with position alignment option
        params = self.transform_coordinate_system(params, align_position=align_position)
        
        return params


    def convert_standard_body_pose(self, body_pose_data):
        """
        Convert standard body pose data to the required format
        
        Args:
            body_pose_data: Original body pose data
            
        Returns:
            Converted body pose tensor
        """
        print(f"Original body_pose shape: {body_pose_data.shape}")
        
        # Determine required body pose size based on model type
        if self.model_type == 'SMPLX':
            expected_size = 21 * 3  # SMPL-X (21 joints * 3 per joint)
        else:  # SMPL
            expected_size = 23 * 3  # SMPL (23 joints * 3 per joint)
            
        actual_size = body_pose_data.size
        
        # Handle size mismatch
        if actual_size < expected_size:
            print(f"Resizing body_pose from {actual_size} to {expected_size}")
            
            # More intelligent handling
            try:
                # Check if size is divisible by 3 (axis-angle format)
                if actual_size % 3 == 0:
                    num_joints = actual_size // 3
                    
                    # Reshape to [batch, joints, 3]
                    joints_data = body_pose_data.reshape(1, num_joints, 3)
                    
                    # Create target array
                    target_joints = np.zeros((1, expected_size // 3, 3), dtype=np.float32)
                    
                    # Copy available joint data
                    target_joints[:, :num_joints, :] = joints_data
                    
                    # Reshape back to flat array
                    resized_data = target_joints.reshape(1, expected_size)
                else:
                    # If not divisible by 3, try to make the most of it
                    print(f"WARNING: body_pose size ({actual_size}) not divisible by 3")
                    usable_size = (actual_size // 3) * 3  # Use the largest multiple of 3
                    
                    resized_data = np.zeros((1, expected_size), dtype=np.float32)
                    resized_data[0, :usable_size] = body_pose_data.flatten()[:usable_size]
            except Exception as e:
                print(f"Error in intelligent reshape: {e}")
                # Fallback to simple padding
                resized_data = np.zeros((1, expected_size), dtype=np.float32)
                resized_data[0, :actual_size] = body_pose_data.flatten()
                
            body_pose = torch.tensor(resized_data, dtype=torch.float32, device=self.device)
            
        elif actual_size > expected_size:
            print(f"Truncating body_pose from {actual_size} to {expected_size}")
            truncated_data = body_pose_data.flatten()[:expected_size].reshape(1, expected_size)
            body_pose = torch.tensor(truncated_data, dtype=torch.float32, device=self.device)
        else:
            body_pose = torch.tensor(body_pose_data, dtype=torch.float32, device=self.device)
        
        return body_pose


    def apply_pose(self, pose_idx=0, align_position=False):
        """
        Apply the specified pose to the mesh using SMPL model
        
        Args:
            pose_idx: Index of the pose to apply
            align_position: Whether to align model position
        
        Returns:
            posed_mesh: The mesh with the pose applied
        """

        if self.model_type == 'NONE':
            print("SMPL model not available, returning original mesh")
            return self.mesh.copy() 
        
        # Convert and validate pose parameters
        params = self.convert_pose_params(pose_idx, align_position=align_position)
        
        # Print key parameters for debugging
        if self.debug_mode:
            if 'body_pose' in params:
                print(f"Body pose norm: {torch.norm(params['body_pose']).item()}")
                print(f"Body pose non-zero elements: {torch.count_nonzero(params['body_pose']).item()}/{params['body_pose'].numel()}")
            if 'global_orient' in params:
                print(f"Global orient: {params['global_orient'].detach().cpu().numpy()}")
        
        # Forward pass through the model
        print(f"Applying {self.model_type} pose parameters to model...")
        try:
            with torch.no_grad():
                output = self.smpl_model(
                    **params,
                    return_verts=True
                )
            
            # Create debug visualizations if enabled
            if self.debug_mode:
                # Save SMPL vertices for debugging
                vertices = output.vertices[0].detach().cpu().numpy()
                
                # Create visualization of the SMPL model
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                # Plot a subset of vertices for clarity
                ax.scatter(vertices[::100, 0], vertices[::100, 1], vertices[::100, 2], 
                        c='r', marker='o', alpha=0.5)
                plt.title(f"SMPL Model Output - Pose {pose_idx}")
                plt.savefig(os.path.join(self.debug_dir, f"smpl_vertices_{pose_idx}.png"))
                plt.close()
                
                # Create skeleton visualization
                try:
                    self.visualize_skeleton(output, os.path.join(self.debug_dir, f"skeleton_{pose_idx}.png"))
                except Exception as e:
                    print(f"Error creating skeleton visualization: {e}")
            
            # Get vertices and create mesh
            vertices = output.vertices[0].detach().cpu().numpy()
            faces = self.smpl_model.faces
            
            posed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            return posed_mesh
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print("Unable to apply pose, returning original mesh")
            return self.mesh.copy()
        


    def visualize_skeleton(self, model_output, output_path):
        """
        Visualize the skeleton of the model
        
        Args:
            model_output: Output from SMPL model
            output_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        # Get joints
        if hasattr(model_output, 'joints') and model_output.joints is not None:
            joint_positions = model_output.joints[0].detach().cpu().numpy()
        else:
            # Use vertices as fallback
            vertices = model_output.vertices[0].detach().cpu().numpy()
            # Select subset of vertices as joint approximations
            joint_indices = [0, 1, 4, 7, 10, 11, 13, 16, 18, 14, 17, 19, 2, 5, 3, 6]
            joint_positions = vertices[joint_indices]
        
        # Define connections for visualization
        connections = [
            (0, 1), (0, 4), (0, 7),  # Pelvis connections
            (7, 10), (10, 11),       # Spine to head
            (10, 13), (13, 16), (16, 18),  # Left arm
            (10, 14), (14, 17), (17, 19),  # Right arm
            (1, 2), (2, 3),           # Left leg
            (4, 5), (5, 6)            # Right leg
        ]
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot joints
        ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], 
                c='r', marker='o', s=50)
        
        # Plot connections
        for connection in connections:
            start, end = connection
            if start < len(joint_positions) and end < len(joint_positions):
                ax.plot([joint_positions[start, 0], joint_positions[end, 0]],
                        [joint_positions[start, 1], joint_positions[end, 1]],
                        [joint_positions[start, 2], joint_positions[end, 2]], 'b-', linewidth=2)
        
        # Set axis limits
        max_range = np.max(joint_positions.max(axis=0) - joint_positions.min(axis=0))
        mid_x = (joint_positions[:, 0].max() + joint_positions[:, 0].min()) * 0.5
        mid_y = (joint_positions[:, 1].max() + joint_positions[:, 1].min()) * 0.5
        mid_z = (joint_positions[:, 2].max() + joint_positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Set title and labels
        ax.set_title('Skeleton Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved skeleton visualization to {output_path}")


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
    
    def process_poses(self, num_poses=1, align_position=False):
        """
        Process multiple poses and save the results
        
        Args:
            num_poses: Number of poses to process
            align_position: Whether to align model position
            
        Returns:
            output_paths: List of paths to the saved posed meshes
        """
        output_paths = []
        
        # Determine how many poses we can process
        if 'global_orient' in self.pose_data:
            max_poses = len(self.pose_data['global_orient'])
        elif 'thetas' in self.pose_data:
            max_poses = len(self.pose_data['thetas'])
        else:
            max_poses = 1
            
        num_poses = min(num_poses, max_poses)
        
        for i in range(num_poses):
            print(f"Processing pose {i+1}/{num_poses}")
            posed_mesh = self.apply_pose(pose_idx=i, align_position=align_position)
            
            # Generate output filename
            basename = os.path.basename(self.mesh_path)
            name, ext = os.path.splitext(basename)
            output_filename = f"{name}_posed_{i:03d}{ext}"
            
            # Save posed mesh
            output_path = self.save_posed_mesh(posed_mesh, output_filename)
            output_paths.append(output_path)
        
        return output_paths




    def transform_coordinate_system(self, params, align_position=False):
        """
        Transform parameters from source coordinate system to target coordinate system
        using automatically detected transformation
        
        Args:
            params: Dictionary of SMPL-X parameters
            align_position: Whether to intelligently align model position
            
        Returns:
            Transformed parameters dictionary
        """
        # Make a deep copy to avoid modifying the original
        transformed_params = copy.deepcopy(params)
        
        # Automatically detect coordinate transformation
        R_transform = self.auto_detect_coordinate_system(params)
        
        # Log the transformation matrix
        if self.debug_mode:
            print(f"Coordinate transformation matrix:\n{R_transform.detach().cpu().numpy()[0]}")
        
        # Transform global orientation
        if 'global_orient' in transformed_params:
            # Convert global_orient from axis-angle to rotation matrix
            global_orient_mat = self.batch_rodrigues(transformed_params['global_orient'].view(-1, 3)).view(-1, 3, 3)
            
            # Apply coordinate transformation: R_new = R_transform * R_old
            transformed_global_orient_mat = torch.bmm(R_transform, global_orient_mat)
            
            # Convert back to axis-angle representation
            transformed_global_orient = self.rotation_matrix_to_angle_axis(transformed_global_orient_mat)
            transformed_params['global_orient'] = transformed_global_orient.view(params['global_orient'].shape)
            
            print(f"Transformed global orientation from {params['global_orient'].detach().cpu().numpy()} to {transformed_params['global_orient'].detach().cpu().numpy()}")
        
        # Transform translation parameter
        if 'transl' in transformed_params:
            # Apply the same transformation to the translation vector
            transl = transformed_params['transl'].view(-1, 3, 1)
            transformed_transl = torch.bmm(R_transform, transl).view(-1, 3)
            transformed_params['transl'] = transformed_transl
            
            print(f"Transformed translation from {params['transl'].detach().cpu().numpy()} to {transformed_params['transl'].detach().cpu().numpy()}")
        

        # Apply position alignment if requested and model is available
        if align_position and self.model_type != 'NONE':
            transformed_params = self.align_model_positions(transformed_params)
        
        return transformed_params



    def batch_rodrigues(self, rot_vecs):
        """
        Convert axis-angle representation to rotation matrix
        
        Args:
            rot_vecs: Axis-angle representation (Nx3)
            
        Returns:
            Rotation matrices (Nx3x3)
        """
        batch_size = rot_vecs.shape[0]
        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle
        
        cos = torch.cos(angle).unsqueeze(-1)
        sin = torch.sin(angle).unsqueeze(-1)
        
        # Rodrigues formula
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        zeros = torch.zeros((batch_size, 1), dtype=rot_vecs.dtype, device=rot_vecs.device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(batch_size, 3, 3)
        
        ident = torch.eye(3, dtype=rot_vecs.dtype, device=rot_vecs.device).unsqueeze(0)
        rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
        
        return rot_mat

    def rotation_matrix_to_angle_axis(self, rotation_matrix):
        """
        Convert rotation matrix to axis-angle representation
        
        Args:
            rotation_matrix: Rotation matrices (Nx3x3)
            
        Returns:
            Axis-angle representation (Nx3)
        """
        batch_size = rotation_matrix.shape[0]
        
        # Get rotation angle using arccos(0.5*(trace(R)-1))
        cos_angle = (rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2] - 1) * 0.5
        cos_angle = torch.clamp(cos_angle, -1, 1)
        angle = torch.acos(cos_angle)
        
        # Get rotation axis
        sin_angle = torch.sin(angle)
        axis = torch.zeros((batch_size, 3), dtype=rotation_matrix.dtype, device=rotation_matrix.device)
        
        # Handle special cases
        # Case 1: angle is close to 0 (identity rotation)
        mask_zero = angle < 1e-5
        
        # Case 2: angle is close to π (180 degrees)
        mask_pi = (angle > (torch.pi - 1e-5))
        
        # Regular case: get axis from skew-symmetric part
        mask_regular = ~(mask_zero | mask_pi)
        
        if torch.any(mask_regular):
            s = 1.0 / (2.0 * sin_angle[mask_regular])
            axis[mask_regular, 0] = s * (rotation_matrix[mask_regular, 2, 1] - rotation_matrix[mask_regular, 1, 2])
            axis[mask_regular, 1] = s * (rotation_matrix[mask_regular, 0, 2] - rotation_matrix[mask_regular, 2, 0])
            axis[mask_regular, 2] = s * (rotation_matrix[mask_regular, 1, 0] - rotation_matrix[mask_regular, 0, 1])
        
        # Handle special case: 180 degree rotation
        if torch.any(mask_pi):
            # For 180 degree rotation, find eigenvector with eigenvalue 1
            for i in torch.where(mask_pi)[0]:
                # Alternative approach - use diagonal elements to determine axis
                diag = rotation_matrix[i].diag()
                axis[i, 0] = torch.sqrt((diag[0] + 1) / 2) if diag[0] > -0.9 else 0
                axis[i, 1] = torch.sqrt((diag[1] + 1) / 2) if diag[1] > -0.9 else 0
                axis[i, 2] = torch.sqrt((diag[2] + 1) / 2) if diag[2] > -0.9 else 0
                
                # Fix signs
                if rotation_matrix[i, 0, 1] + rotation_matrix[i, 1, 0] < 0:
                    axis[i, 0] = -axis[i, 0]
                if rotation_matrix[i, 1, 2] + rotation_matrix[i, 2, 1] < 0:
                    axis[i, 1] = -axis[i, 1]
                if rotation_matrix[i, 2, 0] + rotation_matrix[i, 0, 2] < 0:
                    axis[i, 2] = -axis[i, 2]
        
        # Return axis-angle representation
        return axis * angle.unsqueeze(-1)
    

    def align_model_positions(self, params):
        """
        Intelligently align model positions based on reference T-pose
        
        Args:
            params: Dictionary of SMPL-X parameters
            
        Returns:
            Params with aligned position
        """
        # Deep copy to avoid modifying original
        aligned_params = copy.deepcopy(params)
        
        # 1. Generate a reference mesh (T-pose) in SMPL-X coordinates
        with torch.no_grad():
            t_pose_params = {
                'body_pose': torch.zeros((1, 63), dtype=torch.float32, device=self.device),
                'global_orient': torch.zeros((1, 3), dtype=torch.float32, device=self.device),
                'betas': params['betas'] if 'betas' in params else torch.zeros((1, 10), dtype=torch.float32, device=self.device)
            }
            t_pose_output = self.smpl_model(**t_pose_params)
            t_pose_joints = t_pose_output.joints[0].detach().cpu().numpy()
        
        # 2. Generate the posed mesh with PKL translations
        with torch.no_grad():
            posed_output = self.smpl_model(**params)
            posed_joints = posed_output.joints[0].detach().cpu().numpy()
        
        # 3. Calculate root joint (pelvis) offset
        t_pose_pelvis = t_pose_joints[0]  # assuming joint 0 is pelvis
        posed_pelvis = posed_joints[0]
        
        # Add debug information
        if self.debug_mode:
            print(f"T-pose pelvis position: {t_pose_pelvis}")
            print(f"Posed pelvis position: {posed_pelvis}")
        
        # 4. Calculate required offset for all axes
        offset = t_pose_pelvis - posed_pelvis
        
        if self.debug_mode:
            print(f"Calculated alignment offset: X={offset[0]:.4f}, Y={offset[1]:.4f}, Z={offset[2]:.4f}")
        
        # 5. Apply offset to all three translation axes
        if 'transl' in aligned_params:
            # Store original translation for comparison
            if self.debug_mode:
                orig_transl = aligned_params['transl'].clone().detach().cpu().numpy()
                print(f"Original translation: {orig_transl}")
            
            # Apply computed offset to all three axes
            offset_tensor = torch.tensor(offset, dtype=torch.float32, device=self.device).unsqueeze(0)
            aligned_params['transl'] = aligned_params['transl'] + offset_tensor
            
            if self.debug_mode:
                new_transl = aligned_params['transl'].detach().cpu().numpy()
                print(f"New translation: {new_transl}")
                print(f"Position change: X={new_transl[0,0]-orig_transl[0,0]:.4f}, Y={new_transl[0,1]-orig_transl[0,1]:.4f}, Z={new_transl[0,2]-orig_transl[0,2]:.4f}")
            
            print(f"Applied intelligent position alignment to all axes. Offset: {offset}")
        
        return aligned_params
    
    
    
    
    
    def auto_detect_coordinate_system(self, params):
        """
        Automatically detect coordinate system transformation based on reference pose
        
        Args:
            params: Dictionary of SMPL-X parameters
            
        Returns:
            R_transform: Rotation matrix for coordinate transformation
        """
        print("Auto-detecting coordinate system transformation...")
        
        # Generate T-pose joint positions in SMPL-X coordinate system
        with torch.no_grad():
            t_pose_params = {
                'body_pose': torch.zeros((1, 63), dtype=torch.float32, device=self.device),
                'global_orient': torch.zeros((1, 3), dtype=torch.float32, device=self.device),
                'betas': params['betas'] if 'betas' in params else torch.zeros((1, 10), dtype=torch.float32, device=self.device)
            }
            t_pose_output = self.smpl_model(**t_pose_params)
            t_pose_joints = t_pose_output.joints[0].detach().cpu().numpy()
        
        # Generate test pose with original parameters to check alignment
        with torch.no_grad():
            test_params = copy.deepcopy(params)
            # Use only global_orient to check main axis alignment
            if 'body_pose' in test_params:
                test_params['body_pose'] = torch.zeros_like(test_params['body_pose'])
                
            test_output = self.smpl_model(**test_params)
            test_joints = test_output.joints[0].detach().cpu().numpy()
        
        # Calculate spine direction (from pelvis to neck)
        pelvis_idx = 0
        neck_idx = 12  # This might need adjustment based on SMPL-X joint hierarchy
        
        t_pose_spine = t_pose_joints[neck_idx] - t_pose_joints[pelvis_idx]
        test_spine = test_joints[neck_idx] - test_joints[pelvis_idx]
        
        # Normalize vectors
        t_pose_spine = t_pose_spine / np.linalg.norm(t_pose_spine)
        test_spine = test_spine / np.linalg.norm(test_spine)
        
        # Compute dot product to check alignment
        dot_product = np.dot(t_pose_spine, test_spine)
        
        # If main axes are already aligned (within some tolerance)
        if dot_product > 0.9:
            print("Coordinate systems are already well-aligned")
            return torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # If main axes are approximately anti-parallel (need 180 rotation)
        if dot_product < -0.9:
            print("Detected 180-degree rotation between coordinate systems")
            # Determine which axes need to be flipped
            if abs(test_spine[1]) > abs(test_spine[2]):
                # If Y component is dominant
                print("Applying Y-Z axes flip")
                R = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0]
                ])
            else:
                # If Z component is dominant
                print("Applying X-Y axes flip")
                R = np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ])
                
            return torch.tensor(R, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # For more complex misalignments, compute rotation matrix
        # Calculate rotation axis and angle using cross product and dot product
        rotation_axis = np.cross(test_spine, t_pose_spine)
        
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.clip(dot_product, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Create rotation matrix (Rodrigues formula)
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            
            print(f"Computed rotation matrix for coordinate transformation")
            print(f"Rotation angle: {np.degrees(angle):.2f} degrees")
            
            return torch.tensor(R, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            # Default transformation if calculation fails
            print("Warning: Could not compute rotation. Using Y-Z flip as fallback")
            R = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])
            return torch.tensor(R, dtype=torch.float32, device=self.device).unsqueeze(0)

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
    parser.add_argument("--optimize_pose", action="store_true",
                        help="Enable pose optimization to reduce artifacts")
    parser.add_argument("--use_vposer", action="store_true", default=True,
                        help="Use VPoser for pose decoding if available")
    parser.add_argument("--vposer_path", type=str, default=None,
                        help="Path to VPoser model checkpoint")
    parser.add_argument("--use_tpose_fallback", action="store_true", default=True,
                        help="Use T-pose if other methods fail")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with visualizations")
    parser.add_argument("--align_position", action="store_true",
                        help="Intelligently align model position with reference pose")
    
    args = parser.parse_args()
    
    # Create PoseImposement instance
    pose_imposement = PoseImposement(
        mesh_path=args.mesh_path,
        pose_data_path=args.pose_data_path,
        smpl_model_path=args.smpl_model_path,
        gender=args.gender,
        optimize_pose=args.optimize_pose,
        use_vposer=args.use_vposer,
        vposer_path=args.vposer_path,
        use_tpose_fallback=args.use_tpose_fallback,
        debug_mode=args.debug
    )
    
    # Process poses
    output_paths = pose_imposement.process_poses(num_poses=args.num_poses, align_position=args.align_position)
    
    print("Pose imposement completed.")
    return output_paths


if __name__ == "__main__":
    main()