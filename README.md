**Kindler** is a collection of popular deep learning models that uses the [PyTorch](https://pytorch.org/) high-level APIs. These models are designed with the goals of being efficient and easy to deploy to your production evironment or custom datasets.

Most models are implemented based on their papers but many additional configurable options are provided to improve customizability.

Consider giving this package a try on your next deep learning project!

---

## Usage

### Loading a model

1. Create a `config_file` of the `yaml` format storing your model configs
2. Load your model with the keyword argument `config_file=<Path to your config file>`

As an example, to train a `retinanet` model, you'd want to have at least 2 files
- `retinanet_config.yaml`
    - For all your model configurations
- `main_script.py`
    - For your training process

`retinanet_config.yaml`
```
---
BACKBONE:
  TYPE: resnet50
FPN:
  MIN_LEVEL: 3
  MAX_LEVEL: 7
COMBINED:
  USE: true
  NUM_LAYERS: 4
TARGET:
  NUM_CLASSES: 80
```

`main_script.py`
```
from kindler.retinanet import RetinaNet
model = RetinaNet(config_file='retinanet_config.yaml').cuda()
...
```

### Using the `do_train` engine

To save people the trouble of writing their own training script, the `do_train` function is provided as a means of providing users with a quick and elegant way to carry out, log and checkpoint their training process.

```
# Define your
model = model  # A kindler model
data_loader = <YOUR CUSTOM DATA LOADER>

def loss_fn(batch)
    """ Here batch refers to a batch produced by data_loader """
    # 99% of times you should change this based on the format of batch that
    # your data_loader produces and the input format required of your model
    return model(batch)

do_train(
    model=model,
    data_loader=data_loader,
    loss_fn=loss_fn
    optimizer=torch.optim.Adam(model.parameters())
)

# Of course you can use your favourite optimizer with custom learning rate
# scheduler
# Upon running this function, watch as all your environment information,
# training losses and checkpoint files are logged into neat little log files
# and checkpoint directories
```

## Avaiable models

### [RetinaNet](https://github.com/mingruimingrui/Kindler/tree/master/kindler/retinanet)
> State-of-the-art single shot object detection. [arxiv](https://arxiv.org/abs/1708.02002)
