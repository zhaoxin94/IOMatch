import timm
import torch.nn as nn

class EfficientNetV2(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       num_classes=num_classes,
                                       **kwargs)

        # self._out_features = self.model.num_features

        # self.model.head = nn.Sequential()  # save memory

        # self.old_fc = nn.Linear(self._out_features, num_classes)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.model.forward_head(x)
        
        feat =  self.model.forward_features(x)
        logits = self.model.forward_head(feat)

        result_dict = {'logits': logits, 'feat': feat}
        return result_dict
    


def efficientnetv2_s(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained efficientnetv2_s model!')

    model = EfficientNetV2('efficientnetv2_s', pretrained=pretrained, num_classes=num_classes)

    return model


def efficientnetv2_m(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained efficientnetv2_m model!')

    model = EfficientNetV2('efficientnetv2_m', pretrained=pretrained, num_classes=num_classes)

    return model


