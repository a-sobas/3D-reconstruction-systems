import json

from torch.utils.data import DataLoader

from data_tools.VoxMeshDataset import VoxMeshDataset
from utils.utils import  get_datasets_path


SPLITS_FILE = '/home/asobas/Desktop/m_datasets/pix2mesh_splits_val05.json'


def _identity(x, device):
    return x


def build_loader(
        system='voxels',
        split_name='train',
        batch_size=32,
        shuffle=True,
):
    dataset_dir = get_datasets_path()

    with open(SPLITS_FILE, "r") as f:
        splits = json.load(f)
    if split_name is not None and split_name in ['train', 'val', 'test']:
        split = splits[split_name]

    if system == 'voxels':
        synset = ['chair']
    elif system == 'meshes':
        synset = ['airplane']

    dataset = VoxMeshDataset(dataset_dir,
                            split=split,
                            system=system,
                            imgs_per_model=24,
                            percentage_per_synset=100,
                            synsets=synset
    )

    # creating dataloader
    collate_fn = VoxMeshDataset.collate_fn
    loader_kwargs = {'batch_size': batch_size, 'collate_fn': collate_fn}
    loader = DataLoader(dataset, shuffle=shuffle, **loader_kwargs)

    if hasattr(dataset, "postprocess"):
        postprocess_fn = dataset.postprocess
    else:
        postprocess_fn = _identity
    loader.postprocess = postprocess_fn

    return loader