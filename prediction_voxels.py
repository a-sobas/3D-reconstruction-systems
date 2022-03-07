import os

import torch

from utils.utils import load_model
from data_tools.vox_mesh_loader import build_loader
from architectures.SystemArch import SystemArch
from utils.visualization import show_predicted_voxels


def prediction(model, loader):
    batch_iter = iter(loader)
    batch = next(batch_iter)

    i = 0
    imgs, _, id = batch

    predicted_voxels = model(torch.unsqueeze(imgs[i], 0))
    voxels = torch.sigmoid(predicted_voxels)

    p_voxels = voxels >= 0.61

    show_predicted_voxels(p_voxels[0], id[i])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SystemArch(system='voxels', backbone='b4')
    model.to(device)

    dst = os.path.join(os.getcwd(), 'voxel_training_results')

    load_model(model, os.path.join(dst), 'voxels_state_dict.pt')
    model.eval()
    loader = build_loader(system='voxels', split_name='test', batch_size=24, shuffle=True)
    prediction(model, loader)


if __name__ == '__main__':
    main()