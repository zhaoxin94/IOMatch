import timm
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       **kwargs)

        self._out_features = self.model.head.in_features

        self.model.head = nn.Sequential()  # save memory

        self.old_fc = nn.Linear(self._out_features, num_classes)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.old_fc(x)
        
        feat =  self.model(x)
        logits = self.old_fc(feat)

        result_dict = {'logits': logits, 'feat': feat}
        return result_dict
    


def vit_tiny(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained vit_tiny model!')

    model = Transformer('vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)

    return model

def vit_small(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained vit_small model!')

    model = Transformer('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)

    return model

def vit_base(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained vit_base model!')

    model = Transformer('vit_base_patch16_224', pretrained=pretrained,  num_classes=num_classes)

    return model

def swin_base(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained swin_base model!')

    model = Transformer('swin_base_patch4_window7_224', pretrained=pretrained,  num_classes=num_classes)

    return model

def swin_tiny(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained swin_tiny model!')

    model = Transformer('swin_tiny_patch4_window7_224', pretrained=pretrained,  num_classes=num_classes)

    return model

def swin_small(pretrained=False, pretrained_path=None, num_classes=None, **kwargs):

    if pretrained:
        print('Use imagenet pretrained swin_small model!')

    model = Transformer('swin_small_patch4_window7_224', pretrained=pretrained,  num_classes=num_classes)

    return model

