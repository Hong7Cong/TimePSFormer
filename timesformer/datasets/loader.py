# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from timesformer.datasets.multigrid_helper import ShortCycleBatchSampler
# import torch_xla.distributed.parallel_loader as pl
from . import utils as utils
from .build import build_dataset
# import torch_xla.core.xla_model as xm

def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, time, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    time = [item for sublist in time for item in sublist]

    inputs, labels, video_idx, time, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(time),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, time, extra_data
    else:
        return inputs, labels, video_idx, time, extra_data
    
def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, time, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    time = default_collate(time)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(list(itertools.chain(*data))).view(
                -1, 2
            )
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, time, collated_extra_data

# def detection_collate(batch):
#     """
#     Collate function for detection task. Concatanate bboxes, labels and
#     metadata from different samples in the first dimension instead of
#     stacking them to have a batch-size dimension.
#     Args:
#         batch (tuple or list): data batch to collate.
#     Returns:
#         (tuple): collated detection data batch.
#     """
#     inputs, labels, video_idx, extra_data = zip(*batch)
#     inputs, video_idx = default_collate(inputs), default_collate(video_idx)
#     labels = torch.tensor(np.concatenate(labels, axis=0)).float()

#     collated_extra_data = {}
#     for key in extra_data[0].keys():
#         data = [d[key] for d in extra_data]
#         if key == "boxes" or key == "ori_boxes":
#             # Append idx info to the bboxes before concatenating them.
#             bboxes = [
#                 np.concatenate(
#                     [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
#                 )
#                 for i in range(len(data))
#             ]
#             bboxes = np.concatenate(bboxes, axis=0)
#             collated_extra_data[key] = torch.tensor(bboxes).float()
#         elif key == "metadata":
#             collated_extra_data[key] = torch.tensor(
#                 list(itertools.chain(*data))
#             ).view(-1, 2)
#         else:
#             collated_extra_data[key] = default_collate(data)

#     return inputs, labels, video_idx, collated_extra_data


def construct_loader(cfg, split, device='cpu', is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    if cfg.MULTIGRID.SHORT_CYCLE and split in ["train"] and not is_precise_bn:
        # print('Create a sampler for multi-process training MULTIGRID.SHORT_CYCLE')
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        batch_sampler = ShortCycleBatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
        )
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    elif cfg.TRAIN.TPU_ENABLE == False:
        # Create a sampler for multi-process training
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        # Create a loader
        if cfg.DETECTION.ENABLE:
            collate_func = detection_collate
        elif cfg.TRAIN.DATASET == 'ava':
            collate_func = detection_collate
        else:
            collate_func = None
        # print('Create a sampler for multi-process training')
        # Create a sampler for multi-process training
        # sampler = utils.create_sampler(dataset, shuffle, cfg)
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn = collate_func,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    elif cfg.TRAIN.TPU_ENABLE == True:
        assert False, "Code should not go here here"
        # print('Create a sampler for multi-process TRAIN.TPU_ENABLE')
        # device = xm.xla_device()
        # print('Create a sampler for multi-process training')
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        # sampler = DistributedSampler(dataset, num_replicas=xr.world_size(), rank=xr.global_ordinal()) if xr.world_size() > 1 else None
        # print('Create a loader')
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            persistent_workers=True,
            prefetch_factor=32,
        )
        # print('Create MpDeviceLoader')
        # loader = pl.MpDeviceLoader(temp_loader, 
        #                            device,
        #                            loader_prefetch_size=128,
        #                            device_prefetch_size=1,
        #                            host_to_device_transfer_threads=4)
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = (
        loader.batch_sampler.sampler
        if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
        else loader.sampler
    )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
