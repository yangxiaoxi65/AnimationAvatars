import os
import argparse
import numpy as np
import trimesh
from PIL import Image
import sys

def prepare_texture_for_smpl(texture_path, output_dir=None):
    """准备SMPL兼容的纹理"""
    if output_dir is None:
        output_dir = "prepared_textures"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始纹理
    print(f"加载纹理: {texture_path}")
    texture = Image.open(texture_path)
    
    # 确保尺寸为512x512（SMPL标准尺寸）
    texture = texture.resize((512, 512), Image.LANCZOS)
    
    # 保存准备好的纹理
    output_path = os.path.join(output_dir, "prepared_texture.png")
    texture.save(output_path)
    
    return output_path

def apply_texture_directly(obj_path, texture_path, output_dir=None):
    """直接应用纹理到模型，使用更精确的身体分区"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(obj_path)), "textured_meshes")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {obj_path}")
    mesh = trimesh.load(obj_path)
    
    # 加载纹理
    print(f"加载纹理: {texture_path}")
    texture_image = Image.open(texture_path)
    
    # 获取顶点数据
    vertices = mesh.vertices
    
    # 根据Y轴（高度）排序顶点
    height_order = np.argsort(vertices[:, 1])
    
    # 创建UV数组
    uv = np.zeros((len(vertices), 2))
    
    # 确定身体区域的比例
    # 使用固定比例映射到纹理的不同区域
    total_verts = len(vertices)
    
    # 头部 - 大约占顶部15%的顶点
    head_end_idx = int(total_verts * 0.85)
    # 躯干 - 大约占中部50%的顶点
    torso_end_idx = int(total_verts * 0.35)
    
    # 为头部区域生成UV（对应于纹理上部）
    head_indices = height_order[head_end_idx:]
    for i, idx in enumerate(head_indices):
        # 计算头部区域的UV坐标
        # 使用径向UV映射生成更自然的头部贴图
        vertex = vertices[idx] - np.mean(vertices, axis=0)
        angle = np.arctan2(vertex[0], vertex[2])
        u = 0.5 + 0.4 * np.cos(angle)
        v = 0.85 + 0.15 * np.sin(angle)
        uv[idx] = [u, v]
    
    # 为躯干区域生成UV（对应于纹理中部）
    torso_indices = height_order[torso_end_idx:head_end_idx]
    for i, idx in enumerate(torso_indices):
        # 躯干UV - 确保正确映射到衬衫区域
        vertex = vertices[idx] - np.mean(vertices, axis=0)
        angle = np.arctan2(vertex[0], vertex[2])
        progress = i / len(torso_indices)
        u = 0.5 + 0.4 * np.cos(angle)
        v = 0.35 + 0.5 * progress
        uv[idx] = [u, v]
    
    # 为腿部区域生成UV（对应于纹理下部）
    legs_indices = height_order[:torso_end_idx]
    for i, idx in enumerate(legs_indices):
        # 腿部UV - 确保正确映射到裤子区域
        vertex = vertices[idx] - np.mean(vertices, axis=0)
        angle = np.arctan2(vertex[0], vertex[2])
        progress = i / len(legs_indices)
        u = 0.5 + 0.4 * np.cos(angle)
        v = progress * 0.35
        uv[idx] = [u, v]
    
    # 应用UV坐标和纹理
    print("应用纹理...")
    material = trimesh.visual.material.SimpleMaterial(image=texture_image)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    
    # 保存结果
    basename = os.path.basename(obj_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, f"{name}_textured{ext}")
    
    mesh.export(output_path)
    print(f"保存带纹理的模型到: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="直接纹理应用工具")
    parser.add_argument("--obj_path", type=str, required=True, help="输入OBJ模型的路径")
    parser.add_argument("--texture_path", type=str, required=True, help="纹理图像的路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    
    # 准备纹理
    prepared_texture = prepare_texture_for_smpl(args.texture_path)
    
    # 应用纹理
    apply_texture_directly(args.obj_path, prepared_texture, args.output_dir)

if __name__ == "__main__":
    main()