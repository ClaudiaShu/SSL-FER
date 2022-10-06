import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, mode='image'):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "resnet101": models.resnet101(pretrained=False, num_classes=out_dim),
                            "resnet152": models.resnet152(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        if mode == 'audio':
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
    def forward(self, x):
        self.backbone.eval()
        return self.backbone(x)