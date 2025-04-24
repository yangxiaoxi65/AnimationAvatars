
import os
import pyrender
import trimesh
import numpy as np
from PIL import Image

# === CONFIGURATION ===
obj_path = "meshes/test_image/000.obj"  # Change this to your file path
output_image = "rendered.png"

# === LOAD MESH ===
mesh_trimesh = trimesh.load(obj_path)

# Validate mesh
if not isinstance(mesh_trimesh, trimesh.Trimesh):
    raise TypeError("Loaded file is not a trimesh.Trimesh object.")

# === SET UP SCENE ===
scene = pyrender.Scene()

# Add the mesh
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
scene.add(mesh)

# === ADD CAMERA ===
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -0.2],
    [0.0, 0.0, 1.0, 2.5],  # Positioned 2.5m in front of object
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

# === ADD LIGHT ===
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
scene.add(light, pose=camera_pose)

# === RENDER OFFSCREEN ===
r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=640)
color, _ = r.render(scene)
r.delete()

# === SAVE OUTPUT ===
Image.fromarray(color).save(output_image)
print(f"Rendered image saved to: {output_image}")
