import os

import torch

from utils.utils import load_model
from utils.visualization import show_predicted_meshes
from data_tools.vox_mesh_loader import build_loader
from architectures.SystemArch import SystemArch


def prediction(model, loader):
    batch_iter = iter(loader)
    batch = next(batch_iter)

    i = 0
    imgs, _, _, _, id = batch

    predicted_meshes = model(torch.unsqueeze(imgs[i], 0))

    show_predicted_meshes(predicted_meshes, id[i])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SystemArch(system='meshes', backbone='b3')
    model.to(device)

    dst = os.path.join(os.getcwd(), 'mesh_training_results')

    load_model(model, os.path.join(dst), 'meshes_state_dict.pt')
    model.eval()
    loader = build_loader(system='meshes', split_name='test', shuffle=True, batch_size=32)
    prediction(model, loader)


if __name__ == "__main__":
    main()