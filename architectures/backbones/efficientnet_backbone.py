import torch.nn as nn
import timm


class EfficientNetBackbone(nn.Module):
    
    def __init__(self, net):
        super(EfficientNetBackbone, self).__init__()

        self.stem = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        self.stage1 = net.blocks[0]
        self.stage2 = net.blocks[1]
        self.stage3 = net.blocks[2]
        self.stage4 = net.blocks[3]
        self.stage5 = net.blocks[4]
        self.stage6 = net.blocks[5]
        self.stage7 = net.blocks[6]


    def forward(self, imgs):
        feats = self.stem(imgs)
        conv1 = self.stage1(feats)
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        conv4 = self.stage4(conv3)
        conv5 = self.stage5(conv4)
        conv6 = self.stage6(conv5)
        conv7 = self.stage7(conv6)

        return [conv1, conv2, conv3, conv4, conv5, conv6, conv7]


def build_backbone(architecture,pretrained=True):
    if architecture == 'b3':
        net = timm.create_model('efficientnet_b3', pretrained=pretrained)
        feat_dims = (24, 32, 48, 96, 136, 232, 384)
  
    elif architecture == 'b4':
        net = timm.create_model('efficientnet_b4', pretrained=pretrained)
        feat_dims = (24, 32, 56, 112, 160, 272, 448)  
    backbone = EfficientNetBackbone(net)

    for i, child in enumerate(backbone.children()):
        if i == 6:
            break
        for param in child.parameters():
            param.requires_grad = False

    return backbone, feat_dims