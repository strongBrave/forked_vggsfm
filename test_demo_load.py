# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from torch.utils.data import DataLoader
from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.linemod import LineMod
from vggsfm.utils.utils import seed_all_random_engines
from loguru import logger
import numpy as np


@hydra.main(config_path="cfgs/", config_name="demo")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VGGSfMRunner is the main controller.
    """

    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    # Configure CUDA settings
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)

    # Initialize VGGSfM Runner
    vggsfm_runner = VGGSfMRunner(cfg)

    # Load Data
    test_dataset = LineMod(
        root_dir=cfg.SCENE_DIR, 
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
        pose_estimation=cfg.pose_estimation,
        shuffle=cfg.shuffle,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, drop_last=True, 
                                 collate_fn=test_dataset.custom_collate_fn)

    for i in range(len(test_dataset.cls)):
        test_dataset.set_cls_idx(i)
        logger.info(f"Starting load {test_dataset.current_cls_name} data")
        for images, masks, poses, crop_params, original_images, image_paths, _, _ in test_dataloader:
            if cfg.shuffle:
                image_paths = sorted(image_paths)
            start_image_no = str(int(os.path.splitext(os.path.basename(image_paths[0]))[0]))
            
            logger.success(f"Successfully load {test_dataset.current_cls_name} data from {start_image_no} to {str(int(start_image_no) + 10)}")
            
            # Run VGGSfM
            # Both visualization and output writing are performed inside VGGSfMRunner
            logger.info(f"Starting vggsfm {test_dataset.current_cls_name}")
            predictions = vggsfm_runner.run(
                images, # [B, 3, H, W]
                masks=masks, # [B, 1, H, W]
                original_images=original_images,
                image_paths=image_paths,
                crop_params=crop_params,
                seq_name=test_dataset.current_cls_name,
                output_dir=test_dataset.out_dir[test_dataset.current_cls_idx],
                trg_intrinsics=test_dataset.trg_intrinsics
            )

    print("Demo Finished Successfully")

    return True


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.8, device=2)
    with torch.no_grad():
        demo_fn()
