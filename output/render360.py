import os
import pyrender
import trimesh
import numpy as np
from PIL import Image

# === CONFIGURATION ===
obj_path = "meshes/test_image/000.obj"  # Change this to your file path
output_folder = "renders_360"
os.makedirs(output_folder, exist_ok=True)

# === LOAD MESH ===
mesh_trimesh = trimesh.load(obj_path)
if not isinstance(mesh_trimesh, trimesh.Trimesh):
    raise TypeError("Loaded file is not a trimesh.Trimesh object.")

# === SCENE SETUP ===
scene = pyrender.Scene()
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
scene.add(mesh)

# === RENDERER SETUP ===
r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=640)

# === LIGHT ===
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

# === CAMERA LOOP (Every 45°) ===
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
radius = 2.5  # distance from object
elevation = 0.0  # fixed camera height

for i, angle_deg in enumerate(range(0, 360, 45)):
    angle_rad = np.radians(angle_deg)
    
    # Camera position in circle
    cam_x = radius * np.sin(angle_rad)
    cam_z = radius * np.cos(angle_rad)
    cam_y = elevation

    # Camera pose (look-at from rotated position)
    camera_pose = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad), cam_x],
        [0,                1, 0,                 cam_y],
        [-np.sin(angle_rad), 0, np.cos(angle_rad), cam_z],
        [0, 0, 0, 1]
    ])

    # Add camera and light, track their nodes
    cam_node = scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    light_node = scene.add(light, pose=camera_pose)

    # Render the scene
    color, _ = r.render(scene)
    img = Image.fromarray(color)
    img.save(os.path.join(output_folder, f"render_{angle_deg:03d}.png"))
    print(f"Saved view at {angle_deg}°")

    # Remove only the nodes we just added
    scene.remove_node(cam_node)
    scene.remove_node(light_node)



# Cleanup
r.delete()
print(f"All renders saved in: {output_folder}")
