import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from argparse import ArgumentParser

class SimpleRenderer:
    """
    SimpleRenderer class for visualizing 3D models without OpenGL
    """
    def __init__(self, mesh_path, output_dir=None):
        """
        Initialize the SimpleRenderer
        
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
        
        # Load mesh
        self.load_mesh()
    
    def load_mesh(self):
        """Load mesh from file"""
        print(f"Loading mesh from {self.mesh_path}")
        self.mesh = trimesh.load(self.mesh_path)
        
    def render_single_view(self, output_filename=None):
        """
        Render a single view of the mesh using matplotlib
        """
        # Create a new figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get mesh data
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        # Plot the mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                         triangles=faces, color='gray', alpha=0.8)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Remove axis
        ax.set_axis_off()
        
        # Set the view angle
        ax.view_init(elev=30, azim=45)
        
        # Save the figure
        if output_filename is None:
            basename = os.path.basename(self.mesh_path)
            name, _ = os.path.splitext(basename)
            output_filename = f"{name}_rendered.png"
        
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Saved rendered image to {output_path}")
        return output_path
    
    def render_360(self, num_views=8, output_prefix=None):
        """
        Render multiple views of the mesh at different angles
        """
        # Create output directory for 360 renders
        output_dir_360 = os.path.join(self.output_dir, "360")
        os.makedirs(output_dir_360, exist_ok=True)
        
        output_paths = []
        
        for i in range(num_views):
            # Calculate angle
            angle = i * (360 / num_views)
            
            # Create a new figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get mesh data
            vertices = self.mesh.vertices
            faces = self.mesh.faces
            
            # Plot the mesh
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                             triangles=faces, color='gray', alpha=0.8)
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Remove axis
            ax.set_axis_off()
            
            # Set the view angle
            ax.view_init(elev=30, azim=angle)
            
            # Save the figure
            if output_prefix is None:
                basename = os.path.basename(self.mesh_path)
                name, _ = os.path.splitext(basename)
                output_prefix = name
            
            output_filename = f"{output_prefix}_view_{angle:03.0f}.png"
            output_path = os.path.join(output_dir_360, output_filename)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            output_paths.append(output_path)
            print(f"Rendered view at {angle}°")
        
        print(f"Rendered {num_views} views to {output_dir_360}")
        return output_paths

def main():
    """Main function to run rendering"""
    parser = ArgumentParser(description="Render 3D avatar models (simplified version)")
    parser.add_argument("--mesh_path", type=str, required=True,
                        help="Path to the input mesh (.obj file)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save rendered images")
    parser.add_argument("--render_360", action="store_true",
                        help="Render 360 degree views")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views for 360 rendering")
    
    args = parser.parse_args()
    
    # Create renderer
    renderer = SimpleRenderer(
        mesh_path=args.mesh_path,
        output_dir=args.output_dir
    )
    
    if args.render_360:
        # Render 360 degree views
        output_paths = renderer.render_360(num_views=args.num_views)
    else:
        # Render single view
        output_path = renderer.render_single_view()
        output_paths = [output_path]
    
    print("Rendering completed.")
    return output_paths

if __name__ == "__main__":
    main()