"""DeepLabV3+ / ResNet-50; SMP is the guide's default backend."""
def build_model(backend='smp', pretrained=True):
    if backend == 'smp':
        import segmentation_models_pytorch as smp
        return smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet' if pretrained else None, classes=4)
    if backend != 'native':
        raise ValueError('Unknown model backend.')
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.models.segmentation.deeplabv3 import ASPP
    class DeepLabV3Plus(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None,
                                    replace_stride_with_dilation=[False, False, True])
            self.aspp = ASPP(2048, [6, 12, 18])
            self.low = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU())
            self.decoder = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 4, 1))
        def forward(self, x):
            shape = x.shape[-2:]
            e = self.encoder
            low = e.layer1(e.maxpool(e.relu(e.bn1(e.conv1(x)))))
            deep = e.layer4(e.layer3(e.layer2(low)))
            high = F.interpolate(self.aspp(deep), size=low.shape[-2:], mode='bilinear', align_corners=False)
            return F.interpolate(self.decoder(torch.cat([self.low(low), high], 1)), size=shape, mode='bilinear', align_corners=False)
    return DeepLabV3Plus()


def load_checkpoint(path, device='cpu'):
    from pathlib import Path
    import torch
    if not Path(path).is_file():
        raise FileNotFoundError('No trained checkpoint. Run python -m mars_terrain.train first.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('kind') != 'ai4mars-trained' or checkpoint.get('classes') != ['soil', 'bedrock', 'sand', 'big rock']:
        raise ValueError('Not a compatible AI4Mars checkpoint.')
    model = build_model(checkpoint['backend'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint
