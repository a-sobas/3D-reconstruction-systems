import os
from tqdm import tqdm

import torch
import torch.nn as nn

from architectures.SystemArch import SystemArch
from data_tools.vox_mesh_loader import build_loader
from utils.utils import save_model

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3


def train(model, loader, device, save_dst):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        print('Starting epoch %d / %d' % (epoch + 1, NUM_EPOCHS))

        losses = []
        loop = tqdm(loader, total=len(loader), leave=False)
        for batch in loop:
            batch = loader.postprocess(batch, device)
            imgs, voxels_gt, _ = batch

            voxel_scores = model(imgs)

            loss = loss_fn(voxel_scores, voxels_gt.float())
            losses.append(loss.item())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            loop.set_description(f'Epoch [{epoch}/{NUM_EPOCHS}]')
            loop.set_postfix(loss = loss.item())

        print('Loss on epoch', epoch + 1, ': ', loss.item())
        save_losses(losses, epoch, save_dst)
        
        save_model(model, save_dst, 'voxels_state_dict.pt')


def save_losses(losses, i, save_dst):
    if not os.path.isdir(save_dst):
        os.makedirs(save_dst)
    torch.save(losses, os.path.join(save_dst, f'losses_{i}.pt'))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SystemArch(system='voxels', backbone='b4')
    model.to(device)

    dst = os.path.join(os.getcwd(), 'voxel_training_results')

    model.train()
    loader = build_loader(system='voxels', split_name='train', batch_size=32, shuffle=True)
    
    train(model, loader, device, dst)


if __name__ == '__main__':
    main()