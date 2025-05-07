import os
import subprocess
from argparse import ArgumentParser

def run_pipeline(mesh_path, pose_data_path, smpl_model_path=None, gender='neutral', 
                 num_poses=1, render_360=False, num_views=8, use_matplotlib=True,
                 optimize_pose=True, use_vposer=True, vposer_path=None, 
                 use_tpose_fallback=True, debug=False, data_dir=None, max_images=1,
                 texture_path=None):
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
        optimize_pose: Enable pose optimization to reduce artifacts
        use_vposer: Use VPoser for pose decoding if available
        vposer_path: Path to VPoser model checkpoint
        use_tpose_fallback: Use T-pose if other methods fail
        debug: Enable debug mode
        data_dir: Directory containing processed data for texture mapping
        max_images: Maximum number of images to process for texture mapping
        texture_path: Path to predefined texture image
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
        
    # Add gender
    command.extend(["--gender", gender])
    
    # Add pose optimization flag if enabled
    if optimize_pose:
        command.append("--optimize_pose")
    
    # Add VPoser options if enabled
    if use_vposer:
        command.append("--use_vposer")
        if vposer_path:
            command.extend(["--vposer_path", vposer_path])
    
    # Add T-pose fallback flag if enabled
    if use_tpose_fallback:
        command.append("--use_tpose_fallback")
    
    # Add debug flag if enabled
    if debug:
        command.append("--debug")
    
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
    
    # If no posed mesh files were found, exit
    if len(posed_mesh_files) == 0:
        print("No posed meshes found. Exiting.")
        return
    
    # Run texture mapping if data_dir is provided or texture_path is provided
    textured_mesh_files = []
    if data_dir or texture_path:
        print("\n" + "=" * 50)
        print("Running Stage 4.5: Texture Mapping")
        print("=" * 50)
        
        # Check if texture_mapping.py exists
        if not os.path.exists("texture_mapping.py"):
            print("texture_mapping.py not found. Skipping texture mapping.")
            textured_mesh_files = posed_mesh_files
        else:
            # Process each posed mesh
            for posed_mesh_path in posed_mesh_files:
                print(f"Applying texture to {os.path.basename(posed_mesh_path)}")
                
                # Run texture mapping
                command = [
                    "python", "texture_mapping.py",
                    "--mesh_path", posed_mesh_path
                ]
                
                # Add data_dir if provided
                if data_dir:
                    command.extend(["--data_dir", data_dir])
                
                # Add max_images
                command.extend(["--max_images", str(max_images)])
                
                # Add texture_path if provided
                if texture_path:
                    command.extend(["--texture_path", texture_path])
                
                result = subprocess.run(command, capture_output=True, text=True)
                print(result.stdout)
                
                if result.returncode != 0:
                    print("Error in Texture Mapping:")
                    print(result.stderr)
                    textured_mesh_files.append(posed_mesh_path)  # Use original mesh as fallback
                else:
                    # Check if textured mesh was created
                    textured_meshes_dir = os.path.join(base_dir, "textured_meshes")
                    basename = os.path.basename(posed_mesh_path)
                    name, ext = os.path.splitext(basename)
                    textured_filename = f"{name}_textured{ext}"
                    textured_path = os.path.join(textured_meshes_dir, textured_filename)
                    
                    if os.path.exists(textured_path):
                        textured_mesh_files.append(textured_path)
                        print(f"Successfully created textured mesh: {textured_path}")
                    else:
                        textured_mesh_files.append(posed_mesh_path)  # Use original mesh as fallback
                        print(f"Textured mesh not found, using original: {posed_mesh_path}")
    else:
        # If no data_dir or texture_path provided, use posed meshes without texturing
        textured_mesh_files = posed_mesh_files
        print("No data directory or texture path provided. Skipping texture mapping.")
    
    # Render each mesh (textured if available, otherwise posed)
    meshes_to_render = textured_mesh_files if textured_mesh_files else posed_mesh_files
    
    for mesh_to_render in meshes_to_render:
        print("\n" + "=" * 50)
        print(f"Running Stage 5: Rendering for {os.path.basename(mesh_to_render)}")
        print("=" * 50)
        
        # Run rendering
        command = [
            "python", "render_avatar.py",
            "--mesh_path", mesh_to_render
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
    parser.add_argument("--optimize_pose", action="store_true", default=True,
                        help="Enable pose optimization to reduce artifacts")
    parser.add_argument("--use_vposer", action="store_true", default=True,
                        help="Use VPoser for pose decoding if available")
    parser.add_argument("--vposer_path", type=str, default=None,
                        help="Path to VPoser model checkpoint")
    parser.add_argument("--use_tpose_fallback", action="store_true", default=True,
                        help="Use T-pose if other methods fail")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing processed data for texture mapping")
    parser.add_argument("--max_images", type=int, default=1,
                        help="Maximum number of images to process for texture mapping")
    parser.add_argument("--texture_path", type=str, default=None,
                        help="Path to predefined texture image")
    
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
        use_matplotlib=args.use_matplotlib,
        optimize_pose=args.optimize_pose,
        use_vposer=args.use_vposer,
        vposer_path=args.vposer_path,
        use_tpose_fallback=args.use_tpose_fallback,
        debug=args.debug,
        data_dir=args.data_dir,
        max_images=args.max_images,
        texture_path=args.texture_path
    )
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()