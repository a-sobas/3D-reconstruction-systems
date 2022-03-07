import torch.nn as nn
from pytorch3d.utils import ico_sphere

from architectures.backbones import efficientnet_backbone
from architectures.modules.VoxelsModule import VoxelsModule
from architectures.modules.MeshModule import MeshModule

class SystemArch(nn.Module):
    
    def __init__(self, system='voxels', backbone='b4'):
        super(SystemArch, self).__init__()

        self.system = system
        self.backbone = backbone

        # backbone
        self.backbone, feat_dims = efficientnet_backbone.build_backbone(
            architecture=self.backbone,
            pretrained=True
        )

        if self.system == 'voxels':
            in_channels = feat_dims[-1]
            self.voxel_module = VoxelsModule(in_channels)
        elif self.system == 'meshes':
            self.ico_sphere_level = 3
            in_channels = feat_dims[-2] + feat_dims[-1]
            self.mesh_module = MeshModule(in_channels)


    def forward(self, imgs):
        img_feats = self.backbone(imgs)
        device = imgs.device
            
        if self.system == 'voxels':
            voxel_scores = self.voxel_module(img_feats[-1])

            return voxel_scores
        elif self.system == 'meshes':
            N = imgs.shape[0]
            init_meshes = ico_sphere(self.ico_sphere_level, device).extend(N)
            refined_meshes = self.mesh_module([img_feats[-2], img_feats[-1]], init_meshes)

            return refined_meshes

