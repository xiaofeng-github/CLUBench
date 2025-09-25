from . import resnet, preact_resnet

backbone_dict = {
    'bigresnet18': [resnet.ResNet('resnet18', cifar=True), 512],
    'bigresnet34': [resnet.ResNet('resnet34', cifar=True), 512],
    'bigresnet50': [resnet.ResNet('resnet50', cifar=True), 2048],
    'bigresnet18_preact': [preact_resnet.ResNet18, 512],
    'resnet18': [resnet.ResNet('resnet18'), 512],
    'resnet34': [resnet.ResNet('resnet34'), 512],
    'resnet50': [resnet.ResNet('resnet50'), 2048],
}
