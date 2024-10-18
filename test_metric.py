# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import hydra
import json
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader
from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.linemod import DemoLoader
from vggsfm.utils.metric import (
    compoute_metric,
    write_metrics,
)
from vggsfm.utils.align import align_gt
from vggsfm.utils.utils import seed_all_random_engines
from loguru import logger
import os
import numpy as np

cls = ['ape',  # 0
         'benchvise', # 1
         'cam', # 2
         'can', # 3
         'cat', # 4
         'driller', # 5
         'duck', # 6
         'eggbox', # 7
         'glue', # 8
         'holepuncher', # 9
         'iron', # 10
         'lamp', # 11
         'phone'] # 12


# custome collate function
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    poses = np.stack([item['pose'] for item in batch])
    crop_params = torch.stack([item['crop_params'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    pose_paths = [item['pose_path'] for item in batch]

        # 处理 original_image 是字典的情况
    original_images = {}
    for item in batch:
        for key, value in item['original_image'].items():  # 动态遍历每个 original_image 中的键和值
            if key not in original_images:
                original_images[key] = value

    return (images, masks, poses, crop_params, original_images, image_paths, mask_paths, pose_paths)

@hydra.main(config_path="cfgs/", config_name="demo")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VGGSfMRunner is the main controller.
    """

    OmegaConf.set_struct(cfg, False)

    # Print configuration
    # print("Model Config:", OmegaConf.to_yaml(cfg))

    # Configure CUDA settings
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)



    # Load Data
    test_dataset = DemoLoader(
        root_dir=cfg.SCENE_DIR, 
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
        pose_estimation=cfg.pose_estimation,
        cls=cls,
        shuffle=cfg.shuffle,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, drop_last=True, 
                                 collate_fn=custom_collate_fn)

    mean_metrics = {} # 存储所有了类的mean_metric
    for i in range(len(cls)):
        test_dataset.set_cls_idx(i)
        cls_dir = test_dataset.out_dir[test_dataset.current_cls_idx]
        metrics = {} # 存储每个类的每个batch的metric
        mean_metric = {} # 存储每个类的每个metric的平均值
        batch_count = 0 # 对于最后做平均操作
        logger.info(f"Starting load {test_dataset.current_cls_name} data")

        for (images, masks, poses, crop_params, original_images, image_paths, _, _) in test_dataloader:
            batch_count += 1
            if cfg.shuffle:
                image_paths = sorted(image_paths)
            start_image_no = str(int(os.path.splitext(os.path.basename(image_paths[0]))[0]))
            logger.debug(f"image path: {image_paths}")
            logger.success(f"Successfully load {test_dataset.current_cls_name} data from {start_image_no} to {str(int(start_image_no) + 10)}")
            # Run VGGSfM
            # Both visualization and output writing are performed inside VGGSfMRunner
            logger.info(f"Starting vggsfm {test_dataset.current_cls_name}")
            
            test_pred_pose, test_gt_pose = align_gt(output_dir=cls_dir,
                     gt_poses=poses,
                     batch_size=cfg.batch_size,
                     start_image_no=start_image_no)
            
            metric = compoute_metric(model_path=os.path.join(cls_dir, cls[i] + ".ply"),
                                      pred_pose=test_pred_pose,
                                      gt_pose=test_gt_pose,
                                      )

            for key, value in metric.items():
                mean_metric[key] = mean_metric.get(key, 0) + value

            metrics[start_image_no] = metric
            logger.info(f"{start_image_no} metric: {metric}")

        mean_metric["batch_count"] = batch_count
        mean_metric = {key: value / mean_metric['batch_count'] for key, value in mean_metric.items()}
        mean_metric["batch_count"] = batch_count     
        mean_metrics[cls[i]] = mean_metric   


        logger.info(f"{cls[i]} mean metric: {mean_metric}")
        write_metrics(output_dir=cls_dir, metrics=metrics)

    with open("mean_metric_per_class.json", "w") as f1:
        json.dump(mean_metrics, f1)


    print("Demo Finished Successfully")

    return True


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.8, device=2)
    with torch.no_grad():
        demo_fn()
