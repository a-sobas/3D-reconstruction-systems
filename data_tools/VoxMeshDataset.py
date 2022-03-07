import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pytorch3d.structures import Meshes

SYNSETS = {'02691156': 'airplane',
           '03001627': 'chair'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class VoxMeshDataset(Dataset):

    system = 'voxels'

    def __init__(
            self,
            data_dir,
            split=None,
            system='voxels',
            imgs_per_model=24,
            percentage_per_synset=100,
            normalize_images=True,
            synsets=None,
    ):
        super(VoxMeshDataset, self).__init__()
        print('bbb')
        VoxMeshDataset.system = system

        self.data_dir = data_dir
        self.imgs_per_model = imgs_per_model

        self.synset_ids = []
        self.model_ids = []
        self.image_ids = []
        self.image_list = [f'{id:02}.png' for id in range(24)]

        # image preprocessing
        if normalize_images:
            transforms = [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]
        else:
            transforms = [T.ToTensor()]
        
        self.transform = T.Compose(transforms)

        # extracting correct synset ids
        synset_ids = self.extract_synset_ids(synsets)

        # iter over synsets
        for synset_id in synset_ids:
            model_ids = os.listdir(os.path.join(self.data_dir, synset_id))
            # extracting synset ids from split
            split_m_ids = None
            if split is not None:
                split_m_ids = set(split[synset_id])
            # calculating limit for models per synset
            m_ids_per_synset = int(percentage_per_synset / 100 * len(split_m_ids))
            #print(len(split_m_ids), m_ids_per_synset)

            # iter over models
            i, loaded_items = 0, 0
            for model_id in model_ids:
                # execute models limitation
                if i == m_ids_per_synset:
                    break
                # skip model if it doesn't exist in split
                if split_m_ids is not None and model_id not in split_m_ids:
                    continue
                else:
                    i += 1
                # loading image ids from split
                split_i_ids = None
                if split is not None:
                    split_i_ids = set(split[synset_id][model_id])
                # iter over images
                loaded_images = 0
                for image_id in range(self.imgs_per_model):
                    if split_i_ids is None or image_id in split_i_ids:
                        loaded_images += 1
                        self.synset_ids.append(synset_id)
                        self.model_ids.append(model_id)
                        self.image_ids.append(image_id)
                loaded_items += loaded_images
            print(SYNSETS[synset_id], loaded_items)
        print('Sum:', len(self))


    def __len__(self):
        return len(self.synset_ids)


    def __getitem__(self, idx):
        synset_id = self.synset_ids[idx]
        model_id = self.model_ids[idx]
        image_id = self.image_ids[idx]

        # load image
        img_name = self.image_list[image_id]
        img_path = os.path.join(self.data_dir, synset_id, model_id, "images", img_name)
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = self.transform(img)

        id = f"{synset_id}-{model_id}-{image_id:02}"

        if VoxMeshDataset.system == 'voxels':
            voxels = self.get_voxels(synset_id, model_id)

            return img, voxels, id
        elif VoxMeshDataset.system == 'meshes':
            verts, faces, points, normals = self.get_meshes(synset_id, model_id)

            return img, verts, faces, points, normals, id


    def get_voxels(self, synset_id, model_id):
        voxels = None

        voxel_file = os.path.join(self.data_dir, synset_id, model_id, 'voxels.pt')
        if os.path.isfile(voxel_file):
            with open(voxel_file, "rb") as f:
                voxels = torch.load(f)

        return voxels

    
    def get_meshes(self, synset_id, m_id):
        # read mesh
        mesh_path = os.path.join(self.data_dir, synset_id, m_id, "mesh.pt")
        with open(mesh_path, "rb") as f:
            mesh_data = torch.load(f)
        verts, faces = mesh_data["verts"], mesh_data["faces"]

        # load samples
        sampled_points_path = os.path.join(self.data_dir, synset_id, m_id, "sampled_points.pt")
        sampled_normals_path = os.path.join(self.data_dir, synset_id, m_id, "sampled_normals.pt")
        with open(sampled_points_path, "rb") as f:
            points = torch.load(f)
        with open(sampled_normals_path, "rb") as f:
            normals = torch.load(f)
        idx = torch.randperm(points.shape[0])
        points, normals = points[idx], normals[idx]

        return verts, faces, points, normals


    def extract_synset_ids(self, synsets):
        # load synsets that exists in preprocessed dataset
        synsets_in_dataset = os.listdir(self.data_dir)

        synset_ids = list(SYNSETS.keys())
        synset_names = list(SYNSETS.values())

        if synsets is not None:
            s_ids = []
            for synset in synsets:
                if synset in synset_ids:
                    s_ids.append(synset)
                elif synset in synset_names:
                    s_ids.append(synset_ids[synset_names.index(synset)])
                else:
                    print("Synset %s isn't correct synset id/name!" % synset)
        else:
            return synsets_in_dataset
        return s_ids


    @staticmethod
    def collate_fn(batch):
        if VoxMeshDataset.system == 'voxels':
            imgs, voxels, id = zip(*batch)
            imgs = torch.stack(imgs, dim=0)
            voxels = torch.stack(voxels, dim=0)

            return imgs, voxels, id
        elif VoxMeshDataset.system == 'meshes':
            imgs, verts, faces, points, normals, id = zip(*batch)
            imgs = torch.stack(imgs, dim=0)
            meshes = Meshes(verts=list(verts), faces=list(faces))
            points = torch.stack(points, dim=0)
            normals = torch.stack(normals, dim=0)

            return imgs, meshes, points, normals, id


    def postprocess(self, batch, device):
        if VoxMeshDataset.system == 'voxels':
            imgs, voxels, id = batch
            imgs = imgs.to(device)
            voxels = voxels.to(device)

            return imgs, voxels, id
        elif VoxMeshDataset.system == 'meshes':
            imgs, meshes, points, normals, id = batch
            imgs = imgs.to(device)
            meshes = meshes.to(device)
            points = points.to(device)
            normals = normals.to(device)

            return imgs, meshes, points, normals, id