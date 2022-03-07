import torch
import torch.nn as nn
from pytorch3d.ops import GraphConv, SubdivideMeshes, vert_align
from torch.nn import functional as F


class MeshModule(nn.Module):
    def __init__(self, in_channels):
        super(MeshModule, self).__init__()

        self.num_stages = 2 # num stages
        hidden_dim      = 96 # graph conv dim
        stage_depth     = 2 # num graph convs

        vert_feat_dim = 0
        self.stage1 = MeshDeformationStage(in_channels, vert_feat_dim, hidden_dim, stage_depth)

        vert_feat_dim = hidden_dim
        self.stage2 = MeshDeformationStage(in_channels, vert_feat_dim, hidden_dim, stage_depth)


    def forward(self, img_feats, meshes, subdivide=True):
        output_meshes = []
        vert_feats = None

        # stage 1
        meshes, vert_feats = self.stage1(img_feats, meshes, vert_feats)
        output_meshes.append(meshes)

        # subdivision
        subdivide = SubdivideMeshes()
        meshes, vert_feats = subdivide(meshes, feats=vert_feats)

        #stage 2
        meshes, vert_feats = self.stage2(img_feats, meshes, vert_feats)
        output_meshes.append(meshes)

        return output_meshes


class MeshDeformationStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, stage_depth):
        super(MeshDeformationStage, self).__init__()

        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        self.vert_offset = nn.Linear(hidden_dim + 3, 3)

        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init='normal', directed=False)
            self.gconvs.append(gconv)


    def forward(self, img_feats, meshes, vert_feats_from_previous=None):

        vert_pos_padded = meshes.verts_padded()
        vert_pos_packed = meshes.verts_packed()

        # vertices features sampling
        vert_align_feats = vert_align(img_feats, vert_pos_padded, return_packed=True)
        vert_align_feats = F.relu(self.bottleneck(vert_align_feats))

        first_layer_feats = [vert_align_feats, vert_pos_packed]

        if vert_feats_from_previous is not None:
            first_layer_feats.append(vert_feats_from_previous)

        vert_feats_from_previous = torch.cat(first_layer_feats, dim=1)

        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats_from_previous, meshes.edges_packed()))
            vert_feats_from_previous = torch.cat([vert_feats_nopos, vert_pos_packed], dim=1)

        vert_offsets = torch.tanh(self.vert_offset(vert_feats_from_previous))

        meshes_out = meshes.offset_verts(vert_offsets)

        return meshes_out, vert_feats_nopos
