import os
import subprocess
from argparse import ArgumentParser

def run_pipeline(mesh_path, pose_data_path, smpl_model_path=None, gender='neutral', 
                 num_poses=1, render_360=False, num_views=8, use_matplotlib=True):
    """
    Run the complete pipeline for stages 4 and 5
    
    Args:
        mesh_path: Path to the input mesh (.obj file)
        pose_data_path: Path to the pose data (.pkl file)
        smpl_model_path: Path to the SMPL model directory
        gender: Gender for the SMPL model
        num_poses: Number of poses to process
        render_360: Whether to render 360 degree views
        num_views: Number of views for 360 rendering
        use_matplotlib: Use matplotlib for rendering instead of pyrender
    """
    print("=" * 50)
    print("Running Stage 4: Pose Imposement")
    print("=" * 50)
    
    # Run pose imposement
    command = [
        "python", "pose_imposement.py",
        "--mesh_path", mesh_path,
        "--pose_data_path", pose_data_path,
        "--num_poses", str(num_poses)
    ]
    
    # Add SMPL model path if provided
    if smpl_model_path:
        command.extend(["--smpl_model_path", smpl_model_path])
        
    # Add gender if provided
    command.extend(["--gender", gender])
    
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("Error in Stage 4:")
        print(result.stderr)
        return
    
    # Get the output path from pose imposement
    mesh_dir = os.path.dirname(mesh_path)
    base_dir = os.path.dirname(os.path.dirname(mesh_path))
    posed_meshes_dir = os.path.join(base_dir, "posed_meshes")
    
    # Find the posed mesh files
    posed_mesh_files = []
    for i in range(num_poses):
        basename = os.path.basename(mesh_path)
        name, ext = os.path.splitext(basename)
        posed_filename = f"{name}_posed_{i:03d}{ext}"
        posed_path = os.path.join(posed_meshes_dir, posed_filename)
        
        if os.path.exists(posed_path):
            posed_mesh_files.append(posed_path)
    
    print(f"Found {len(posed_mesh_files)} posed mesh files")
    
    # Render each posed mesh
    for posed_mesh_path in posed_mesh_files:
        print("\n" + "=" * 50)
        print(f"Running Stage 5: Rendering for {os.path.basename(posed_mesh_path)}")
        print("=" * 50)
        
        # Run rendering
        command = [
            "python", "render_avatar.py",
            "--mesh_path", posed_mesh_path
        ]
        
        if render_360:
            command.extend(["--render_360", "--num_views", str(num_views)])
        
        if use_matplotlib:
            command.append("--use_matplotlib")
        
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode != 0:
            print("Error in Stage 5:")
            print(result.stderr)

def main():
    """Main function to run the pipeline"""
    parser = ArgumentParser(description="Run the complete pipeline for stages 4 and 5")
    parser.add_argument("--mesh_path", type=str, default="output/meshes/test_image/000.obj",
                        help="Path to the input mesh (.obj file)")
    parser.add_argument("--pose_data_path", type=str, default="output/results/test_image/000.pkl",
                        help="Path to the pose data (.pkl file)")
    parser.add_argument("--smpl_model_path", type=str, default=None,
                        help="Path to the SMPL model directory")
    parser.add_argument("--gender", type=str, default="neutral", 
                        choices=["neutral", "male", "female"],
                        help="Gender for the SMPL model")
    parser.add_argument("--num_poses", type=int, default=1,
                        help="Number of poses to process")
    parser.add_argument("--render_360", action="store_true",
                        help="Render 360 degree views")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views for 360 rendering")
    parser.add_argument("--use_matplotlib", action="store_true", default=True,
                        help="Use matplotlib for rendering instead of pyrender")
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        mesh_path=args.mesh_path,
        pose_data_path=args.pose_data_path,
        smpl_model_path=args.smpl_model_path,
        gender=args.gender,
        num_poses=args.num_poses,
        render_360=args.render_360,
        num_views=args.num_views,
        use_matplotlib=args.use_matplotlib
    )
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()