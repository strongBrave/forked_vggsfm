# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader
from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.linemod import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines
from loguru import logger

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
    images = torch.stack([item[0] for item in batch])  # 假设第一个元素是 image
    masks = torch.stack([item[1] for item in batch])  # 假设第二个元素是 mask
    crop_params = torch.stack([item[2] for item in batch])  # 假设第三个元素是 crop_params
    
    # 处理 original_image 是字典的情况
    original_images = {}
    for item in batch:
        for key, value in item[3].items():  # 动态遍历每个 original_image 中的键和值
            if key not in original_images:
                original_images[key] = []
            original_images[key].append(value)
    
    image_paths = [item[4] for item in batch]  # 假设第五个是 image_path

    return (images, masks, crop_params, original_images, image_paths)


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
    test_dataset = DemoLoader(
        root_dir=cfg.SCENE_DIR, 
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
        pose_estimation=cfg.pose_estimation,
        cls=cls,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True, 
                                 collate_fn=custom_collate_fn)

    for i in range(4, 5):
        test_dataset.set_cls_idx(i)
        logger.info(f"Starting load {test_dataset.current_cls_name} data")
        for (images, masks, crop_params, original_images, image_paths) in test_dataloader:
            print("image shape: ", images.shape)
            print("mask shape: ", masks.shape)
            print("crop_params shape: ", crop_params.shape)
            # print("original_image: ", original_image)
            print("image path: ", image_paths)
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
        logger.success(f"Successfully load {test_dataset.current_cls_name} data")
            

    # sequence_list = test_dataset. # 有几种种类, e.g. [cat, mug, ...]

    # seq_name = sequence_list[0]  # Run on one Scene, e.g. kitchen

    # Load the data for the selected sequence
    # todo: watch
    # batch, image_paths, trg_intrinsics = test_dataset.get_data(
    #     sequence_name=seq_name, return_path=True
    # )

    # output_dir = batch[
    #     "scene_dir"
    # ]  # which is also cfg.SCENE_DIR for DemoLoader

    # images = batch["image"]
    # # IMPORTANT: MASK
    # masks = batch["masks"] if batch["masks"] is not None else None # 1 filter out
    # crop_params = (
    #     batch["crop_params"] if batch["crop_params"] is not None else None
    # )

    # # Cache the original images for visualization, so that we don't need to re-load many times
    # original_images = batch["original_images"]
    # trg_intrinsics=None


    # print(torch.argwhere(masks[0, 0] == 0).shape)
    print("Demo Finished Successfully")

    return True


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.8, device=2)
    with torch.no_grad():
        demo_fn()
