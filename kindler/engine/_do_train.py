import time
import datetime
import logging

from torch import distributed as dist

from ..utils.misc import to_device
from ..utils.comm import get_world_size, is_main_process

logger = logging.getLogger(__name__)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if is_main_process():
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    loss_fn,
    optimizer,
    scheduler=None,
    device=None,
):
    """
    Args:
        model: The model to train
        data_loader: An torch.utils.data.DataLoader object
        loss_fn: A wrapper that takes in a model and a batch
            and outputs a training loss_dict.
            This loss_dict can be must be a dict containing a key 'total_loss'
        optimizer: The optimizer to specify which parameters to update as well
            as the learning rate
        scheduler: The learning rate scheduler. If no scheduler is present,
            we assume constant learning rate
        device: A torch.device object that model is on
    """
    logger.info('Start training')
    start_training_time = time.time()

    model.train()
    max_iter = len(data_loader)

    t0 = time.time()
    for iter, batch in enumerate(data_loader):
        if device is not None:
            batch = to_device(batch, device)
        data_time = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        loss_dict = loss_fn(model, batch)

        # Comute reduced loss_dict
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        reduce_time = time.time() - t0

        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()

        batch_time = time.time() - t0
        t0 = time.time()

        time_elapsed = time.time() - start_training_time
        eta_seconds = time_elapsed / iter * (max_iter - iter)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        
