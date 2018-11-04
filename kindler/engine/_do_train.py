from __future__ import absolute_import
from __future__ import division

import os
import json
import logging

import time
import datetime

from torch import distributed as dist

from ..utils.metric_logger import MetricLogger
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


def save_model(checkpoint_dir, iter, model, optimizer, scheduler=None):
    data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()

    if hasattr(model, 'config'):
        data['config'] = model.config

    file_name = os.path.join(checkpoint_dir, 'model_{}.pth.tar'.format(iter))
    torch.save(data, file_name)


def do_train(
    model,
    data_loader,
    loss_fn,
    optimizer,
    scheduler=None,
    logging_period=250,
    checkpoint_period=2500,
    checkpoint_dir='./'
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
    """
    logger.info('Start training')
    start_training_time = time.time()

    model.train()
    device = next(model.parameters()).device
    max_iter = len(data_loader)
    meters = MetricLogger(delimiter='  ')

    t0 = time.time()
    for iter, batch in enumerate(data_loader):
        if device is not None:
            batch = to_device(batch, device)
        data_time = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        loss_dict = loss_fn(model, batch)

        # Comute gather and reduce loss
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        meters.update(**loss_dict_reduced)
        reduce_time = time.time() - t0

        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()

        batch_time = time.time() - t0
        meters.update(batch_time=batch_time, data_time=data_time, reduce_time=reduce_time)
        t0 = time.time()

        eta_seconds = meters.batch_time.global_avg / iter * (max_iter - iter)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iter % checkpoint_period == 0 and iter > 0:
            save_model(checkpoint_dir, iter, model, optimizer, scheduler)

        if iter % logging_period == 0 or iter == (max_iter - 1):
            msg = {
                'eta': eta_string,
                'iter': iter,
                'lr': '{:.6f}'.format(optimizer.param_groups[0]["lr"]),
                'max mem': '{:.0f}'.format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            }
            logger.info('{}'.format(msg))

    save_model(checkpoint_dir, iter, model, optimizer, scheduler)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    msg = 'Total training time: {} ({:.4f}s / it)'.format(
        total_time_str,
        total_training_time / max_iter
    )
    logger.info(msg)
