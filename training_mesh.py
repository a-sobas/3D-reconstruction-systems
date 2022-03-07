import os
from tqdm import tqdm

import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes

from utils.utils import save_model
from data_tools.vox_mesh_loader import build_loader
from architectures.SystemArch import SystemArch


NUM_EPOCHS = 10
LEARNING_RATE = 1e-3


def train(model, loader, device, loss_fn, save_dst):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print('Starting epoch %d / %d' % (epoch + 1, NUM_EPOCHS))

        losses_to_save = {'total_loss': [],
                          'losses': []}

        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        for i, batch in loop:
            batch = loader.postprocess(batch, device)
            imgs, _, points_gt, normals_gt, _ = batch

            meshes_pred = model(imgs)

            loss, losses = loss_fn(
                meshes_pred,
                (points_gt, normals_gt)
            )

            if i % 10 == 0:
                losses_to_save['total_loss'].append(loss.item())
                losses_to_save['losses'].append(losses)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            loop.set_postfix(loss=loss.item())
            
        print('Loss on epoch', epoch + 1, ': ', loss)

        save_model(model, save_dst, 'meshes_state_dict.pt')
        save_losses(losses_to_save, epoch, save_dst)


class Loss(nn.Module):
    def __init__(
            self,
            chamfer_weight=1.0,
            normal_weight=2e-4,
            edge_weight=0.8,
            gt_num_samples=2000,
            pred_num_samples=2000,
    ):

        super(Loss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.gt_num_samples = gt_num_samples
        self.pred_num_samples = pred_num_samples

    def forward(self, meshes_pred, meshes_gt):
        points_gt, normals_gt = meshes_gt

        total_loss = torch.tensor(0.0).to(points_gt)
        losses = {}

        meshes_pred = meshes_pred

        for i, cur_meshes_pred in enumerate(meshes_pred):
            cur_out = self.mesh_loss(cur_meshes_pred, points_gt, normals_gt)
            cur_loss, cur_losses = cur_out

            total_loss = total_loss + cur_loss / len(meshes_pred)

            for k, v in cur_losses.items():
                losses['%s_%d' % (k, i)] = v

        return total_loss, losses


    def mesh_loss(self, meshes_pred, points_gt, normals_gt):
        points_pred, normals_pred = sample_points_from_meshes(
            meshes_pred,
            num_samples=self.pred_num_samples,
            return_normals=True
        )
        cham_loss, normal_loss = chamfer_distance(
            points_pred,
            points_gt,
            x_normals=normals_pred,
            y_normals=normals_gt
        )
        edge_loss = mesh_edge_loss(meshes_pred)

        total_loss = torch.tensor(0.0).to(points_pred)
        total_loss = total_loss + self.chamfer_weight * cham_loss
        total_loss = total_loss + self.normal_weight * normal_loss
        total_loss = total_loss + self.edge_weight * edge_loss

        losses = {}
        losses['chamfer'] = cham_loss
        losses['normal'] = normal_loss
        losses['edge'] = edge_loss

        return total_loss, losses


def save_losses(losses, i, save_dst):
    torch.save(losses, os.path.join(save_dst, f'losses_{i}.pt'))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SystemArch(system='meshes', backbone='b3')
    model.to(device)

    dst = os.path.join(os.getcwd(), 'mesh_training_results')

    loss_fn_kwargs = {
        'chamfer_weight': 1.0,
        'normal_weight': 2e-4,
        'edge_weight': 0.8,
        'gt_num_samples': 2000,
        'pred_num_samples': 2000,
    }
    loss_fn = Loss(**loss_fn_kwargs)

    loader = build_loader(system='meshes', split_name='train', shuffle=True, batch_size=32)
    model.train()
    train(model, loader, device, loss_fn, dst)


if __name__ == '__main__':
    main()
