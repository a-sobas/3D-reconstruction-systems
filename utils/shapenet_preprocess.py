import os
import shutil
from tqdm import tqdm

import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.datasets.r2n2.utils import read_binvox_coords

DIRS = {
    'output_dir': '',
    'renderings_dir': '/home/asobas/Desktop/m_datasets/ShapeNetRendering',
    'shapenet_dir': '/home/asobas/Desktop/m_datasets/ShapeNetCore.v1',
    'binvox_dir': '/home/asobas/Desktop/m_datasets/ShapeNetVox32'
}

SYNSET_LIST = {'voxels': '03001627', 'meshes': '02691156'}

NUM_CORES = 6
NUM_SAMPLES = 2000


def main():
    if DIRS['output_dir'] == '':
        DIRS['output_dir'] = os.path.join(os.getcwd(), 'preprocessed_datasets')

    if not os.path.isdir(DIRS['output_dir']):
        os.makedirs(DIRS['output_dir'])

    for system, synset_id in SYNSET_LIST.items():
        synset_dir = os.path.join(DIRS['renderings_dir'], synset_id)
        model_ids = os.listdir(synset_dir)

        tasks = []
        for model_id in model_ids:
            renderings_dir = os.path.join(DIRS['renderings_dir'], synset_id, model_id, "rendering")
            obj_dir = os.path.join(DIRS['shapenet_dir'], synset_id, model_id)
            voxels_dir = os.path.join(DIRS['binvox_dir'], synset_id, model_id)

            # check files existence
            images_exists = os.path.isdir(renderings_dir)
            objs_exists = os.path.isdir(obj_dir)
            voxels_exists = os.path.isdir(voxels_dir)

            if images_exists and objs_exists and voxels_exists:
                tasks.append((synset_id, model_id))

        loop = tqdm(tasks, total=len(tasks), leave=False)
        for task in loop:
            handle_model(task[0], task[1], system)


def handle_model(synset_id, model_id, system):
    output_dir = os.path.join(DIRS['output_dir'], synset_id, model_id)
    os.makedirs(output_dir)

    preprocess_images(output_dir, synset_id, model_id)
    if system == 'voxels':
        preprocess_voxels(output_dir, synset_id, model_id)
    elif system == 'meshes':
        preprocess_meshes(output_dir, synset_id, model_id)


# preprocess images
def preprocess_images(output_dir, synset_id, model_id):
    renderings_dir = os.path.join(DIRS['renderings_dir'], synset_id, model_id, "rendering")

    image_list = load_image_list(synset_id, model_id)

    output_img_dir = os.path.join(output_dir, "images")
    os.makedirs(output_img_dir)
    for img_name in image_list:
        src = os.path.join(renderings_dir, img_name)
        dst = os.path.join(output_img_dir, img_name)
        shutil.copy(src, dst)


# preprocess voxels
def preprocess_voxels(output_dir, synset_id, model_id):
    voxels_dir = os.path.join(DIRS['binvox_dir'], synset_id, model_id)

    voxel_path = os.path.join(voxels_dir, "model.binvox")
    with open(voxel_path, "rb") as f:
        voxel_coords = read_binvox_coords(f)

    voxels = coords_to_exact_voxels(voxel_coords)

    voxel_path = os.path.join(output_dir, "voxels.pt")
    torch.save(voxels, voxel_path)


# preprocess meshes
def preprocess_meshes(output_dir, synset_id, model_id):
    obj = os.path.join(DIRS['shapenet_dir'], synset_id, model_id, 'model.obj')

    dst = os.path.join(output_dir, 'model.obj')
    shutil.copy(obj, dst)

    mesh = load_obj(obj, load_textures=False)
    verts = mesh[0]
    faces = mesh[1].verts_idx

    mesh_data = {"verts": verts, "faces": faces}
    mesh_path = os.path.join(output_dir, "mesh.pt")
    torch.save(mesh_data, mesh_path)

    mesh = Meshes(verts=[verts], faces=[faces])
    points, normals = sample_points_from_meshes(
        mesh,
        num_samples=NUM_SAMPLES,
        return_normals=True
    )
    sampled_points = points[0].cpu().detach()
    sampled_normals = normals[0].cpu().detach()
    sampled_points_path = os.path.join(output_dir, "sampled_points.pt")
    sampled_normals_path = os.path.join(output_dir, "sampled_normals.pt")
    torch.save(sampled_points, sampled_points_path)
    torch.save(sampled_normals, sampled_normals_path)


def coords_to_exact_voxels(coords):
    V = 32

    coords = coords.round().to(torch.int64)

    assert coords.min() >= 0 and coords.max() <= (V - 1), f'Invalid coords! {coords.min()} {coords.max()}'

    x, y, z = coords.unbind(dim=1)
    voxels = torch.zeros(V, V, V, dtype=torch.uint8)
    voxels[z, y, x] = 1

    return voxels


def load_image_list(sid, mid):
    path = os.path.join(DIRS['renderings_dir'], sid, mid, "rendering", "renderings.txt")
    with open(path, "r") as f:
        image_list = [line.strip() for line in f]
    return image_list


if __name__ == "__main__":
    main()