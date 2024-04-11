import logging

from datasets.cifar_dataset import build_cifar10_dataloader
from datasets.custom_dataset import build_custom_dataloader
from datasets.my_dataset import build_my_dataloader

logger = logging.getLogger("global")


def build(cfg, stage, distributed):

    dataset = cfg["image_dir"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    else:
        data_loader = build_my_dataloader(cfg, stage, distributed)

    return data_loader


def build_dataloader(cfg_dataset, distributed=True):
    train_loader = None
    val_loader = None
    test_loader = None


    train_loader = build(cfg_dataset, stage="train", distributed=distributed)
    val_loader = build(cfg_dataset, stage="val", distributed=distributed)
    test_loader = build(cfg_dataset, stage="test", distributed=distributed)

    logger.info("build dataset done")
    return train_loader, val_loader, test_loader
