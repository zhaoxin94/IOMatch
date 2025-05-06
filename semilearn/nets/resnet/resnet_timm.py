import timm
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       num_classes=num_classes,
                                       **kwargs)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.model.forward_head(x)
        
        feat =  self.model.forward_features(x)
        logits = self.model.forward_head(feat)

        result_dict = {'logits': logits, 'feat': feat}
        return result_dict
    


def resnet50(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained resnet50 model!')

    model = Resnet('resnet50', pretrained=pretrained, num_classes=num_classes)

    return model



