import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR


def get_optimizer(model, optimizer_name, lr=1e-3, momentum=0.9, weight_decay=5e-4):
    if optimizer_name.lower()=="adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower()=="momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower()=="sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(scheduler_name, optimizer, lr_decay_step, milestones, T_max, eta_min,gamma,epoch=None):
    if scheduler_name.lower()=="steplr":
        return StepLR(optimizer, lr_decay_step, gamma)
    elif scheduler_name.lower()=="multisteplr":
        return MultiStepLR(optimizer,milestone,gamma)
    elif scheduler_name.lower()=="exponentiallr":
        return ExponentialLR(optimizer, gamma)
    elif scheduler_name.lower()=="cosineannealinglr":
        return CosineAnnealingLR(optimizer, T_max, eta_min)
