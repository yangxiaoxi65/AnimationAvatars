import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser

PYRENDER_AVAILABLE = False

try:
    import pyrender
    PYRENDER_AVAILABLE = True
    print("Pyrender available, using it for high-quality rendering")
except ImportError:
    print("Pyrender not available, using matplotlib for rendering")

class AvatarRenderer:
    """
    AvatarRenderer class for visualizing 3D human models
    Uses matplotlib for rendering
    """
    def __init__(self, mesh_path, output_dir=None):
        """
        Initialize the AvatarRenderer
        
        Args:
            mesh_path: Path to the input mesh (.obj file)
            output_dir: Directory to save output files
        """
        self.mesh_path = mesh_path
        
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(mesh_path)), 
                                           "renders")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set default rendering parameters
        self.viewport_width = 640
        self.viewport_height = 640
        self.background_color = np.array([1.0, 1.0, 1.0, 0.0])  # Transparent white
        
        # Load mesh
        self.load_mesh()
    
    def load_mesh(self):
        """Load mesh from file"""
        print(f"Loading mesh from {self.mesh_path}")
        self.trimesh_mesh = trimesh.load(self.mesh_path)
        
        # Create pyrender mesh if available
        global PYRENDER_AVAILABLE
        if PYRENDER_AVAILABLE:
            try:
                self.mesh = pyrender.Mesh.from_trimesh(self.trimesh_mesh, smooth=True)
            except Exception as e:
                print(f"Error creating pyrender mesh: {e}")
                PYRENDER_AVAILABLE = False
    
    def render_single_view(self, output_filename=None, use_matplotlib=True):
        """
        Render a single view of the mesh
        
        Args:
            output_filename: Optional output filename
            use_matplotlib: Whether to use matplotlib for rendering
            
        Returns:
            output_path: Path to the saved image
        """
        global PYRENDER_AVAILABLE
        if PYRENDER_AVAILABLE and not use_matplotlib:
            return self.render_with_pyrender(output_filename)
        else:
            return self.render_with_matplotlib(output_filename)
    
    def render_with_pyrender(self, output_filename=None):
        """
        Render using pyrender
        
        Args:
            output_filename: Optional output filename
            
        Returns:
            output_path: Path to the saved image
        """
        try:
            # Setup scene
            scene = pyrender.Scene(bg_color=self.background_color, ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
            
            # Add the mesh to the scene
            scene.add(self.mesh)
            
            # Default camera pose (front view)
            camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.5],  # 2.5 units away from origin
                [0.0, 0.0, 0.0, 1.0],
            ])
            
            # Add camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            scene.add(camera, pose=camera_pose)
            
            # Add lighting
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            scene.add(light, pose=camera_pose)
            
            # Create renderer
            r = pyrender.OffscreenRenderer(viewport_width=self.viewport_width, 
                                          viewport_height=self.viewport_height)
            
            # Render the scene
            color, depth = r.render(scene)
            
            # Create image from rendered data
            img = Image.fromarray(color)
            
            # Generate output filename if not provided
            if output_filename is None:
                basename = os.path.basename(self.mesh_path)
                name, _ = os.path.splitext(basename)
                output_filename = f"{name}_rendered.png"
            
            # Save image
            output_path = os.path.join(self.output_dir, output_filename)
            img.save(output_path)
            print(f"Saved rendered image to {output_path}")
            
            # Clean up
            r.delete()
            
            return output_path
        except Exception as e:
            print(f"Error with pyrender: {e}")
            print("Falling back to matplotlib rendering...")
            return self.render_with_matplotlib(output_filename)
    
    def render_360(self, num_views=8, output_prefix=None, use_matplotlib=True):
        """
        Render multiple views of the mesh at different angles
        
        Args:
            num_views: Number of views to render
            output_prefix: Prefix for output filenames
            use_matplotlib: Whether to use matplotlib for rendering
            
        Returns:
            output_paths: List of paths to the saved images
        """
        # Create output directory for 360 renders
        output_dir_360 = os.path.join(self.output_dir, "360")
        os.makedirs(output_dir_360, exist_ok=True)
        
        # Generate output prefix if not provided
        if output_prefix is None:
            basename = os.path.basename(self.mesh_path)
            name, _ = os.path.splitext(basename)
            output_prefix = name
        
        global PYRENDER_AVAILABLE
        if PYRENDER_AVAILABLE and not use_matplotlib:
            return self.render_360_with_pyrender(num_views, output_prefix, output_dir_360)
        else:
            return self.render_360_with_matplotlib(num_views, output_prefix, output_dir_360)
    
    def render_360_with_pyrender(self, num_views=8, output_prefix=None, output_dir=None):
        """
        Render 360 degree views using pyrender
        
        Args:
            num_views: Number of views to render
            output_prefix: Prefix for output filenames
            output_dir: Directory to save output files
            
        Returns:
            output_paths: List of paths to the saved images
        """
        try:
            # Create renderer
            r = pyrender.OffscreenRenderer(viewport_width=self.viewport_width, 
                                          viewport_height=self.viewport_height)
            
            output_paths = []
            
            # Render from different angles
            for i in range(num_views):
                # Calculate angle in radians
                angle = 2 * np.pi * i / num_views
                
                # Calculate camera position
                radius = 2.5  # Distance from model
                cam_x = radius * np.sin(angle)
                cam_z = radius * np.cos(angle)
                
                # Create camera pose matrix (looking at origin)
                camera_pose = np.array([
                    [np.cos(angle), 0, np.sin(angle), cam_x],
                    [0, 1, 0, 0],
                    [-np.sin(angle), 0, np.cos(angle), cam_z],
                    [0, 0, 0, 1]
                ])
                
                # Setup scene
                scene = pyrender.Scene(bg_color=self.background_color, ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
                
                # Add the mesh to the scene
                scene.add(self.mesh)
                
                # Add camera
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                scene.add(camera, pose=camera_pose)
                
                # Add lighting
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
                scene.add(light, pose=camera_pose)
                
                # Render the scene
                color, depth = r.render(scene)
                
                # Create image from rendered data
                img = Image.fromarray(color)
                
                # Generate output filename
                angle_degrees = int(angle * 180 / np.pi)
                output_filename = f"{output_prefix}_view_{angle_degrees:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save image
                img.save(output_path)
                output_paths.append(output_path)
                
                print(f"Rendered view at {angle_degrees}°")
            
            # Clean up
            r.delete()
            
            print(f"Rendered {num_views} views to {output_dir}")
            return output_paths
        except Exception as e:
            print(f"Error with pyrender: {e}")
            print("Falling back to matplotlib rendering...")
            return self.render_360_with_matplotlib(num_views, output_prefix, output_dir)
    
    def render_360_with_matplotlib(self, num_views=8, output_prefix=None, output_dir=None):
        """
        Render 360 degree views using matplotlib
        
        Args:
            num_views: Number of views to render
            output_prefix: Prefix for output filenames
            output_dir: Directory to save output files
            
        Returns:
            output_paths: List of paths to the saved images
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "360")
            os.makedirs(output_dir, exist_ok=True)
        
        if output_prefix is None:
            basename = os.path.basename(self.mesh_path)
            name, _ = os.path.splitext(basename)
            output_prefix = name
        
        output_paths = []
        
        # Get mesh data
        vertices = self.trimesh_mesh.vertices
        faces = self.trimesh_mesh.faces
        
        # Render from different angles
        for i in range(num_views):
            # Calculate angle
            angle = 360 * i / num_views
            
            # Create a new figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the mesh
            mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                triangles=faces, 
                                color='gray',
                                alpha=0.9,
                                edgecolor=None,
                                shade=True)
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Remove axis
            ax.set_axis_off()
            
            # Set the view angle
            ax.view_init(elev=30, azim=angle)
            
            # Tight layout
            plt.tight_layout()
            
            # Generate output filename
            output_filename = f"{output_prefix}_view_{int(angle):03d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save figure
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            
            output_paths.append(output_path)
            print(f"Rendered view at {int(angle)}°")
        
        print(f"Rendered {num_views} views to {output_dir}")
        return output_paths
    
    def render_with_matplotlib(self, output_filename=None):
        """
        Render with matplotlib
        
        Args:
            output_filename: Optional output filename
            
        Returns:
            output_path: Path to the saved image
        """
        # Create a new figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get mesh data
        vertices = self.trimesh_mesh.vertices
        faces = self.trimesh_mesh.faces
        
        # Plot the mesh with enhanced visuals
        mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                            triangles=faces, 
                            color='gray',
                            alpha=0.9,
                            edgecolor=None,
                            shade=True)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Remove axis
        ax.set_axis_off()
        
        # Set the view angle
        ax.view_init(elev=30, azim=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Generate output filename if not provided
        if output_filename is None:
            basename = os.path.basename(self.mesh_path)
            name, _ = os.path.splitext(basename)
            output_filename = f"{name}_rendered.png"
        
        # Save figure
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        print(f"Saved matplotlib rendered image to {output_path}")
        return output_path

def main():
    """Main function to run rendering"""
    parser = ArgumentParser(description="Render 3D human avatar models")
    parser.add_argument("--mesh_path", type=str, required=True,
                        help="Path to the input mesh (.obj file)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save rendered images")
    parser.add_argument("--render_360", action="store_true",
                        help="Render 360 degree views")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views for 360 rendering")
    parser.add_argument("--use_matplotlib", action="store_true", default=True,
                        help="Use matplotlib for rendering instead of pyrender")
    
    args = parser.parse_args()
    
    # Create renderer
    renderer = AvatarRenderer(
        mesh_path=args.mesh_path,
        output_dir=args.output_dir
    )
    
    if args.render_360:
        # Render 360 degree views
        output_paths = renderer.render_360(
            num_views=args.num_views,
            use_matplotlib=args.use_matplotlib
        )
    else:
        # Render single view
        output_path = renderer.render_single_view(
            use_matplotlib=args.use_matplotlib
        )
        output_paths = [output_path]
    
    print("Rendering completed.")
    return output_paths

if __name__ == "__main__":
    main()