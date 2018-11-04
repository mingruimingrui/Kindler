import torch


def make_sgd_optimizer(
    model,
    base_lr=0.001,
    bias_lr_factor=2.0,
    momentum=0.9,
    weight_decay=0.0005,
    weight_decay_bias=0.0,
):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        param_lr = base_lr
        param_weight_decay = weight_decay

        if "bias" in key:
            param_lr = base_lr * bias_lr_factor
            param_weight_decay = weight_decay_bias

        params.append({
            'params': [value],
            'lr': param_lr,
            'weight_decay': param_weight_decay
        })

    optimizer = torch.optim.SGD(params, base_lr, momentum=momentum)
    return optimizer
