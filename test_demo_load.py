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
from vggsfm.utils.metric import save_metrics_to_json


@hydra.main(config_path="cfgs/", config_name="demo")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VGGSfMRunner is the main controller.
    """

    OmegaConf.set_struct(cfg, False)
    logger.add(sink="linemod.log")

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

    avg_cls_metrics = {} # 存储所有了类的avg_cls_metric
    for i in range(len(test_dataset.cls_dirs)):

        test_dataset.set_cls_idx(i)
        cls_dir = test_dataset.out_dir[test_dataset.current_cls_idx]
        model_path=os.path.join(cls_dir, test_dataset.current_cls_name + ".ply")
        logger.info(f"Starting to Process {test_dataset.current_cls_name} data")

        per_cls_metrics = {} # 存储每个类的每个batch的metric
        avg_cls_metric = {} # 存储每个类的每个metric的平均值
        batch_count = 0 # 对于最后做平均操作

        for images, masks, poses, crop_params, original_images, image_paths, _, _ in test_dataloader:
            if cfg.shuffle:
                image_paths = sorted(image_paths)
            start_image_no = str(int(os.path.splitext(os.path.basename(image_paths[0]))[0]))
            batch_count += 1
            
            logger.success(f"Successfully load {test_dataset.current_cls_name} data from {start_image_no} to {str(int(start_image_no) + 10)}")
            
            # Run VGGSfM
            # Both visualization and output writing are performed inside VGGSfMRunner
            predictions = vggsfm_runner.run(
                images, # [B, 3, H, W]
                gt_poses=poses,
                masks=masks, # [B, 1, H, W]
                original_images=original_images,
                image_paths=image_paths,
                crop_params=crop_params,
                seq_name=test_dataset.current_cls_name,
                output_dir=test_dataset.out_dir[test_dataset.current_cls_idx],
                trg_intrinsics=test_dataset.trg_intrinsics,
                model_path=model_path,
            )

            metric = predictions["metric"]
            logger.info(f"{start_image_no} metric: {metric}")
            
            for key, value in metric.items():
                avg_cls_metric[key] = avg_cls_metric.get(key, 0) + value
            per_cls_metrics[start_image_no] = metric
        
        batch_count = avg_cls_metric["batch_count"]
        # calculate average metrics
        avg_cls_metric = {key: value / batch_count for key, value in avg_cls_metric.items() if key != 'batch_count'}
        avg_cls_metric["batch_count"] = batch_count    
        avg_cls_metrics[test_dataset.current_cls_name] = avg_cls_metric   


        logger.info(f"{test_dataset.current_cls_name} mean metric: {avg_cls_metric}")
        save_metrics_to_json(save_path=os.path.join(cls_dir, "metrics.json"), metrics=per_cls_metrics)
        logger.success(f"Successfully save {test_dataset.current_cls_name} metrics json to {os.path.join(cls_dir, 'metrics.json')}.")

    logger.info(f"avg_cls_metrics information after {i + 1} update: {avg_cls_metrics}")
    save_metrics_to_json(save_path="avg_cls_metrics.json", metrics=avg_cls_metrics) # save mean metrics
    logger.success(f"Successfully save avg_cls_metrics.json")
    logger.info("Demo Finished Successfully")
    
    return True


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.8, device=2)
    with torch.no_grad():
        demo_fn()
