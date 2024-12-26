from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    mobilenet_v2, mobilenet_v3_large,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights, MobileNet_V2_Weights, MobileNet_V3_Large_Weights
)


state = {
    "model": None,
    "dataloader": None,
    "num_images": 0,
    "training_status": "Idle",
    "current_epoch": 0,
}


MODEL_MAP = {
    "resnet": {
        "small": resnet18,
        "medium": resnet34,
        "large": resnet50,
        "xlarge": resnet101,
        "xxlarge": resnet152,
    },
    "mobilenet": {
        "small": mobilenet_v2,
        "large": mobilenet_v3_large,
    },
}
WEIGHTS_MAP = {
    "resnet": {
        "small": ResNet18_Weights.IMAGENET1K_V1,
        "medium": ResNet34_Weights.IMAGENET1K_V1,
        "large": ResNet50_Weights.IMAGENET1K_V1,
        "xlarge": ResNet101_Weights.IMAGENET1K_V1,
        "xxlarge": ResNet152_Weights.IMAGENET1K_V1,
    },
    "mobilenet": {
        "small": MobileNet_V2_Weights.IMAGENET1K_V1,
        "large": MobileNet_V3_Large_Weights.IMAGENET1K_V2,  # V2 has better accuracy
    },
}