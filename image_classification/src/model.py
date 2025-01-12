import torch.nn as nn
import pretrainedmodels


# def get_model(pretrained):
#     if pretrained:
#         model = pretrainedmodels.__dict__["alexnet"]( pretrained='imagenet')
#     else:
#         model = pretrainedmodels.__dict__["alexnet"]( pretrained=None)
#
#     # print the model here to know what's going on.
#     model.last_linear = nn.Sequential(
#         nn.BatchNorm1d(4096),
#         nn.Dropout(p=0.25),
#         nn.Linear(in_features=4096, out_features=2048),
#         nn.ReLU(),
#         nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
#         nn.Dropout(p=0.5),
#         nn.Linear(in_features=2048, out_features=1),
#     )
#     return model

def get_model(model_name, pretrained):
    if model_name not in pretrainedmodels.__dict__:
        raise ValueError(f"Model {model_name} is not available in pretrainedmodels")

    if pretrained:
        model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__[model_name](pretrained=None)

    # Modify the last layer based on the model architecture
    if model_name == "alexnet":
        model.last_linear = nn.Sequential(
            nn.BatchNorm1d(4096),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1),
        )
    elif model_name == "resnet18":
        model.last_linear = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1),
        )
    else:
        raise ValueError(f"Model {model_name} is not supported for modification")

    return model

