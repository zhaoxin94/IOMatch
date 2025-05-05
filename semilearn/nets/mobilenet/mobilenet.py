import torch.nn as nn
import torchvision

class MobileNet_V2(nn.Module):
    def __init__(self, weights, num_classes):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(weights=weights)
        self.model.classifier[1] = nn.Identity()

        self._out_features = self.model.last_channel
        self.old_fc = nn.Linear(self._out_features, num_classes)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.old_fc(x)
        
        feat =  self.model(x)
        logits = self.old_fc(feat)

        result_dict = {'logits': logits, 'feat': feat}
        return result_dict


def mobilenet_v2(pretrained=False, pretrained_path=None, **kwargs):
    if pretrained:
        print('Use imagenet pretrained mobilenet_v2 model!')
        weights = 'DEFAULT'
    else:
        weights = None

    model = MobileNet_V2(weights=weights, **kwargs)

    return model
    

