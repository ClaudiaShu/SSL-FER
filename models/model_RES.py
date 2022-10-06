import torchvision.models

from models import *

class RES_featurenet(nn.Module):
    def __init__(self, module='resnet50', drop_out=0.5):
        super(RES_featurenet, self).__init__()

        # pretrained on ImageNet
        if module == "resnet18":
            self.backbone = torchvision.models.resnet18(pretrained=True).cuda()
        elif module == "resnet50":
            self.backbone = torchvision.models.resnet50(pretrained=True).cuda()
        else:
            self.backbone = torchvision.models.resnet34(pretrained=True).cuda()

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 512
        self.out = MLP([in_features, in_features], final_relu=True, drop_out=drop_out)

        # dim_mlp = self.backbone.fc.in_features
        # # add mlp projection head
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x, mode='train'):
        if mode == 'train':
            out = self.backbone(x)
        else:
            x = x.permute(0,3,1,2)
            out = self.backbone(x)
        return self.out(out)