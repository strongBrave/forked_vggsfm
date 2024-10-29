# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import glob
import torch
import copy
import pycolmap
import numpy as np


from typing import Optional, List

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

from minipytorch3d.cameras import PerspectiveCameras

from .camera_transform import (
    normalize_cameras,
    adjust_camera_to_bbox_crop_,
    adjust_camera_to_image_scale_,
    bbox_xyxy_to_xywh,
)


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def merge_batch(train_batch, test_batch):
    batch = copy.deepcopy(train_batch)
    for key, value in train_batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = torch.concat((train_batch[key], test_batch[key]), dim=0)
        elif key == "pose":
             batch[key] = np.concatenate((train_batch[key], test_batch[key]), axis=0)            
        elif key in ['image_path', 'mask_path', 'pose_path']:
            batch[key].extend(test_batch[key])
        elif key == 'n':
            batch[key] = test_batch['n']
        elif key == "original_image":
            batch[key].update(test_batch[key])
        else:
            pass
    return batch


class LineMod(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 1024,
        eval_time: bool = True,
        normalize_cameras: bool = True,
        sort_by_filename: bool = True,
        load_gt: bool = False,
        prefix: str = "images",
        pose_estimation: bool=False,
        shuffle: bool=False,
        split="train",
    ):
        """
        Initialize the DemoLoader dataset.

        Args:
            SCENE_DIR (str): Directory containing the scene data. Assumes the following structure:
                - cls/: Contains all the needed data of a class in the linemod datasset.
                    images/: Contains images of this class
                    note: - (optional) masks/: Contains masks for the images with the same name.
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
            img_size (int): Size to resize images to.
            eval_time (bool): Flag to indicate if it's evaluation time.
            normalize_cameras (bool): Flag to indicate if cameras should be normalized.
            sort_by_filename (bool): Flag to indicate if images should be sorted by filename.
            load_gt (bool): Flag to indicate if ground truth data should be loaded.
                    If True, the gt is assumed to be in the sparse/0 directory, in a colmap format.
                    colmap format: cameras.bin, images.bin, and points3D.bin
            cls (list): List containing the class name for the linemod dataset.
        """
        if not root_dir:
            raise ValueError("SCENE_DIR cannot be None")

        self.root_dir = root_dir
        self.cls_dirs = sorted(os.listdir(root_dir))

        # NOTE: This remove operation is optional, which depends on your file structure.
        self.cls_dirs.remove(".DS_Store")
        self.cls_dirs.remove("intrinsics.npy")
        self.processed_image_path_txt_paths = []

        if isinstance(split, str):
            split = [split]

        if "train" in split:
            self.processed_image_path_txt_paths = [os.path.join(root_dir, class_name, "train.txt") for class_name in self.cls_dirs]
        if "test" in split:
            self.processed_image_path_txt_paths = [os.path.join(root_dir, class_name, "test.txt") for class_name in self.cls_dirs]
        

        self.crop_longest = True
        self.load_gt = load_gt
        self.sort_by_filename = sort_by_filename 
        self.sequences = {}

        for processed_image_path_txt_path in self.processed_image_path_txt_paths:
            class_name = os.path.basename(os.path.dirname(processed_image_path_txt_path))
            self.sequences[class_name] = self._load_images(self._load_processed_image_path(processed_image_path_txt_path))
        

        self.prefix = prefix
        self.pose_estimation = pose_estimation # indicate if this is pose_estimation task
        self.first_pose = True # 控制self.trg_intrinsics更新一次，防止增量更新
        self.cls_len = [int(len(self.sequences[class_name])) for class_name in self.cls_dirs] # 每个类的照片的个数

        self.current_cls_idx = 0 # 目前的所在类的下标
        self.current_cls_name = self.cls_dirs[self.current_cls_idx] # 目前所在类的名字
        self.current_cls_len = self.cls_len[self.current_cls_idx] # 目前所在类的照片的个数
        self.out_dir = [os.path.join(root_dir, class_name) for class_name in self.cls_dirs]

        # bag_name = os.path.basename(os.path.normpath(SCENE_DIR)) # e.g. cat
        self.have_mask = []
        for cls in self.cls_dirs:
            self.have_mask.append(os.path.exists(os.path.join(root_dir, cls, "masks"))) # 判断有无mask

        intrinsics_path = os.path.join(root_dir, "intrinsics.npy")
        if os.path.exists(intrinsics_path):
            self.trg_intrinsics = torch.from_numpy(np.load(intrinsics_path)) # 需要后续根据crop和resize做变换



        # if self.sort_by_filename and not shuffle: # 对文件照片排序
        #     self.sequences = {class_name: self._load_images(sorted(self._load_processed_image_path(processed_image_path_txt_path)))}

        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size, antialias=True)]
        )

        self.jitter_scale = [1, 1]
        self.jitter_trans = [0, 0]
        self.img_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras

        print(f"Data size of Sequence: {len(self)}")

    def _load_processed_image_path(self, processed_image_path_txt_path):
        with open(processed_image_path_txt_path, 'r') as file:
            address_list = [line.strip() for line in file if line.strip()]
        return address_list
    
    # custome collate function
    def custom_collate_fn(self, batch):
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        poses = np.stack([item['pose'] for item in batch])
        crop_params = torch.stack([item['crop_params'] for item in batch])
        image_paths = [item['image_path'] for item in batch]
        mask_paths = [item['mask_path'] for item in batch]
        pose_paths = [item['pose_path'] for item in batch]
        n = [item['n'] for item in batch][0]

            # 处理 original_image 是字典的情况
        original_images = {}
        for item in batch:
            for key, value in item['original_image'].items():  # 动态遍历每个 original_image 中的键和值
                if key not in original_images:
                    original_images[key] = value
        
        batch = {
            'image': images, 
            "mask": masks,  
            "pose": poses, 
            "crop_params": crop_params, 
            "original_image": original_images, 
            "image_path": image_paths,
            "mask_path": mask_paths, 
            "pose_path": pose_paths, 
            'n': n,
        }

        return batch

    def set_cls_idx(self, cls_idx):
        self.current_cls_idx = cls_idx
        self.current_cls_name = self.cls_dirs[cls_idx]
        self.current_cls_len = self.cls_len[cls_idx]
        

    def _load_images(self, img_filenames: list) -> list:
        """
        Load images (just image file name) and optionally their annotations.

        Args:
            img_filenames (list): List of image file paths.

        Returns:
            list: List of dictionaries containing image paths and annotations.
        """
        filtered_data = []
        calib_dict = self._load_calibration_data() if self.load_gt else {} # 每个image的外参，内参，焦距以及cx,cy

        for img_name in img_filenames:
            frame_dict = {"img_path": img_name}
            if self.load_gt:
                anno_dict = calib_dict[os.path.basename(img_name)] # basename选取文件路径最后一个单词
                frame_dict.update(anno_dict) # 默认基本不做这个操作
            filtered_data.append(frame_dict)
        return filtered_data

    # TODO: modify thid code, cause the R, T, focal_length, pp is given in LineMod dataset.
    def _load_calibration_data(self) -> dict:
        """
        Load calibration data from the colmap reconstruction.

        Returns:
            dict: Dictionary containing calibration data for each image.
        """
        reconstruction = pycolmap.Reconstruction(
            os.path.join(self.SCENE_DIR, "sparse", "0")
        )
        calib_dict = {}
        for image_id, image in reconstruction.images.items():
            extrinsic = image.cam_from_world.matrix()
            intrinsic = reconstruction.cameras[
                image.camera_id
            ].calibration_matrix()

            R = torch.from_numpy(extrinsic[:, :3])
            T = torch.from_numpy(extrinsic[:, 3])
            fl = torch.from_numpy(intrinsic[[0, 1], [0, 1]])
            pp = torch.from_numpy(intrinsic[[0, 1], [2, 2]])

            calib_dict[image.name] = {
                "R": R,
                "T": T,
                "focal_length": fl,
                "principal_point": pp,
            }
        return calib_dict

    def __len__(self) -> int:
        return self.current_cls_len
        # return 30

    def __getitem__(self, index: int):
        """
        Get data for a specific index.

        Args:
            idx_N (int): Index of the data to retrieve.

        Returns:
            dict: Data for the specified index.
        """
        if self.eval_time:
            return self.get_data(index=index, ids=None)
        else:
            raise NotImplementedError("Do not train on Sequence.")

    def _load_image_and_mask(self, anno: dict) -> tuple:
        """
        Load images and masks from annotations.

        Args:
            annos (dict): The annotations specified by the index

        Returns:
            tuple: Tuple containing image, mask, and image path of the specified index.
        """
        
        image_path = anno["img_path"]
        image = Image.open(image_path).convert("RGB")
        image_no = int(os.path.splitext(os.path.basename(image_path))[0])
        pose_path = image_path.replace(f"/{self.prefix}", "/pose")
        pose_path = pose_path.replace(os.path.basename(image_path), "pose" + str(image_no) + ".npy")
        pose = np.load(pose_path)

        if self.have_mask:
            mask_path = image_path.replace(f"/{self.prefix}", "/masks")
            mask_path = mask_path.replace("jpg", "png") # NOTE: mask为png图片
            mask = Image.open(mask_path).convert("L") # 转换为灰度图，单通道

        return image, mask, pose, image_path, mask_path, pose_path

    def _load_path(self, image_path) -> dict:
        """
        Load a bunch of paths for a image, containg image_path, mask_path, npy_path
        
        Args:
            image_path (str) : The path of a image
        Returns
            path (dict): A dict contains 'image_path', 'mask_path', 'pose_path'.
        """
        path = {}
        p_dir = os.path.dirname(os.path.dirname(image_path))
        image_base = os.path.basename(image_path)
        mask_base = copy.copy(image_base).replace("jpg", "png")
        image_no = int(os.path.splitext(image_base)[0]) # int去掉前置0
        needed = ['images', 'masks', 'pose']
        path['image_path'] = image_path
        path['mask_path'] = os.path.join(p_dir, "masks", mask_base)
        path['pose_path'] = os.path.join(p_dir, "pose", "pose" + str(image_no) + ".npy")
        return path


    def get_data(
        self,
        index: Optional[int] = None,
        ids: Optional[np.ndarray] = None,
        return_path: bool = False,
    ) -> dict:
        """
        Get data for a specific sequence or index.

        Args:
            index (Optional[int]): Index of the image.
            sequence_name (Optional[str]): Name of the sequence.
            ids (Optional[np.ndarray]): Array of indices to retrieve.
            return_path (bool): Flag to indicate if image paths should be returned.

        Returns:
            dict: Batch of data.
        """


        metadata = self.sequences[self.current_cls_name] # 包含很多字典的list, {'img_path', 'R', 'R', 'ff', 'ppxy'}
        n = len(metadata)
        if ids is None:
            ids = np.arange(len(metadata)) # frame的数量


        anno = metadata[index] # a dict containing {'img_path', 'R', 'R', 'ff', 'ppxy'}

        image, mask, pose, image_path, mask_path, pose_path = self._load_image_and_mask(anno) # 返回是一个image, mask, image_path
        image, mask, image_path, crop_paras, original_image = self._prepare_batch(
            anno, image, mask, image_path
        ) #这里的image, pask是转换为tensor的

        sample = {
            'image': image, # 3 x H x W, tensor
            "mask": mask,  # 1 x H x W, tensor
            "pose": pose, # 3 x 3, numpy
            "crop_params": crop_paras, # 1 x 8, tensor
            "original_image": original_image, # , dict
            "image_path": image_path, # string
            "mask_path": mask_path, # string
            "pose_path": pose_path, # string
            "n": n,
        }
        return sample


    def _prepare_batch(
        self,
        anno: dict,
        image,
        mask,
        image_path,
    ) -> tuple:
        """
        Prepare a batch of data for a given sequence.

        This function processes the provided sequence name, metadata, annotations, images, masks, and image paths
        to create a batch of data. It handles the transformation of images and masks, the adjustment of camera parameters,
        and the preparation of ground truth camera data if required.

        Args:
            sequence_name (str): Name of the sequence.
            metadata (list): List of metadata for the sequence. 包含了annos，长度为frame的数量S
            annos (list): List of annotations for the sequence. 保存了image_names
            images (list): List of images for the sequence.
            masks (list): List of masks for the sequence.
            image_paths (list): List of image paths for the sequence.

        Returns:
            dict: Batch of data containing transformed images, masks, crop parameters, original images, and other relevant information.
        """
        # batch = {"seq_name": self.current_cls_name, "frame_num": len(metadata)}
        # crop_parameters, images_transformed, masks_transformed = [], [], []
        # original_images = (
        #     {}
        # )  # Dictionary to store original images before any transformations

        if self.load_gt: # 要load ground-truth point
            new_fls, new_pps = [], [] # 焦距，中心点

        mask = mask if self.have_mask[self.current_cls_idx] else None

        # Store the original image in the dictionary with the basename of the image path as the key
        original_image = {os.path.basename(image_path): np.array(image)} # e.g. 001

        # Transform the image and mask, and get crop parameters and bounding box
        (image_transformed, mask_transformed, crop_paras, bbox) = (
            pad_and_resize_image(
                image,
                self.crop_longest,
                self.img_size,
                mask=mask,
                transform=self.transform,
            )
        )


        if self.load_gt:
            bbox_xywh = torch.FloatTensor(bbox_xyxy_to_xywh(bbox)) # 转换为左上角xy坐标加上bbox的wh, 这里的bbox是没有recale过的
            (focal_length_cropped, principal_point_cropped) = (
                adjust_camera_to_bbox_crop_(
                    anno["focal_length"],
                    anno["principal_point"],
                    torch.FloatTensor(image.size),
                    bbox_xywh,
                )
            )
            (new_focal_length, new_principal_point) = (
                adjust_camera_to_image_scale_(
                    focal_length_cropped,
                    principal_point_cropped,
                    torch.FloatTensor(image.size),
                    torch.FloatTensor([self.img_size, self.img_size]),
                )
            )
            # new_fls.append(new_focal_length) # 因为照片被裁减了，所以焦距和pps要发生变化
            # new_pps.append(new_principal_point)
            # NOTE: new_fls, new_pps 是NDC Space 中的坐标 [-1, 1]

        # images = torch.stack(images_transformed)
        # masks = torch.stack(masks_transformed) if self.have_mask else None

        # TODO: check this
        if self.load_gt:
            # batch.update(self._prepare_gt_camera_batch(anno, new_fl, new_pp))
            pass
        if self.pose_estimation and self.first_pose:
            # update target intrinsic matrix based on crop and resize.
            # NOTE: only update self.trg_intrinsic once
            s = crop_paras[3]
            bbx_top_left_afer_scale = crop_paras[4:6]
            ff = self.trg_intrinsics[[0, 1], [0, 1]].clone()
            ppxy = self.trg_intrinsics[[0, 1], [2, 2]].clone()
            new_f = ff * s
            new_ppxy = s * ppxy - bbx_top_left_afer_scale # note: not right, the s should be 1024 / crop_longest_size 
            self.trg_intrinsics[[0, 1], [0, 1]] = new_f
            self.trg_intrinsics[[0, 1], [2, 2]] = new_ppxy

            self.first_pose = False

        image = image_transformed.clamp(0, 1)
        mask = mask_transformed.clamp(0, 1) if self.have_mask else None
        return image, mask, image_path, crop_paras, original_image

    def _prepare_gt_camera_batch(
        self, anno, new_fl, new_pp
    ) -> dict:
        """

        Prepare a batch of ground truth camera data from annotations and adjusted camera parameters.

        This function processes the provided annotations and adjusted camera parameters (focal lengths and principal points)
        to create a batch of ground truth camera data. It also handles the conversion of camera parameters from the
        OpenCV/COLMAP format to the PyTorch3D format. If normalization is enabled, the cameras are normalized, and the
        resulting parameters are included in the batch.

        Args:
            annos (list): List of annotations, where each annotation is a dictionary containing camera parameters.
            new_fls (list): List of new focal lengths after adjustment.
            new_pps (list): List of new principal points after adjustment.

        Returns:
            dict: A dictionary containing the batch of ground truth camera data, including raw and processed camera
                  parameters such as rotation matrices (R), translation vectors (T), focal lengths (fl), and principal
                  points (pp). If normalization is enabled, the normalized camera parameters are included.
        """
        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        batchR = torch.cat([data["R"][None] for data in annos])
        batchT = torch.cat([data["T"][None] for data in annos])

        batch = {"rawR": batchR.clone(), "rawT": batchT.clone()}

        # From OPENCV/COLMAP to PT3D
        batchR = batchR.clone().permute(0, 2, 1) # 转置操作
        batchT = batchT.clone()
        batchR[:, :, :2] *= -1
        batchT[:, :2] *= -1

        cameras = PerspectiveCameras(
            focal_length=new_fls.float(),
            principal_point=new_pps.float(),
            R=batchR.float(),
            T=batchT.float(),
        )

        if self.normalize_cameras:
            normalized_cameras, _ = normalize_cameras(cameras, points=None)
            if normalized_cameras == -1:
                raise RuntimeError(
                    "Error in normalizing cameras: camera scale was 0"
                )

            batch.update(
                {
                    "R": normalized_cameras.R,
                    "T": normalized_cameras.T,
                    "fl": normalized_cameras.focal_length,
                    "pp": normalized_cameras.principal_point,
                }
            )

            if torch.any(torch.isnan(batch["T"])):
                raise RuntimeError("NaN values found in camera translations")
        else:
            batch.update(
                {
                    "R": cameras.R,
                    "T": cameras.T,
                    "fl": cameras.focal_length,
                    "pp": cameras.principal_point,
                }
            )

        return batch


def calculate_crop_parameters(image, bbox, crop_dim, img_size):
    """
    Calculate the parameters needed to crop an image based on a bounding box.

    Args:
        image (PIL.Image.Image): The input image.
        bbox (np.array): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        crop_dim (int): The dimension to which the image will be cropped.
        img_size (int): The size to which the cropped image will be resized.

    Returns:
        torch.Tensor: A tensor containing the crop parameters, including width, height, crop width, scale, and adjusted bounding box coordinates.
    """
    crop_center = (bbox[:2] + bbox[2:]) / 2
    # convert crop center to correspond to a "square" image
    width, height = image.size
    length = max(width, height)
    s = length / min(width, height)
    crop_center = crop_center + (length - np.array([width, height])) / 2
    # convert to NDC
    # cc = s - 2 * s * crop_center / length
    crop_width = 2 * s * (bbox[2] - bbox[0]) / length
    s = img_size / crop_dim # e.g. 1024 / max( w, h)
    bbox_after = bbox * s 
    crop_parameters = torch.tensor(
        [
            width,
            height,
            crop_width,
            s,
            bbox_after[0],
            bbox_after[1],
            bbox_after[2],
            bbox_after[3],
        ]
    ).float()
    return crop_parameters


def pad_and_resize_image(
    image: Image.Image,
    crop_longest: bool,
    img_size,
    mask: Optional[Image.Image] = None,
    bbox_anno: Optional[np.array] = None,
    transform=None,
):
    """
    Pad (through cropping) and resize an image, optionally with a mask.

    Args:
        image (PIL.Image.Image): Image to be processed.
        crop_longest (bool): Flag to indicate if the longest side should be cropped.
        img_size (int): Size to resize the image to.
        mask (Optional[PIL.Image.Image]): Mask to be processed.
        bbox_anno (Optional[np.array]): Bounding box annotations.
        transform (Optional[transforms.Compose]): Transformations to apply.
    """
    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size, antialias=True)]
        ) # NOTE: ToTensor()会做归一化操作

    w, h = image.width, image.height
    if crop_longest:  # 如果true，则将图形裁剪为max(w, h)的正方形，多加的区域用pad
        crop_dim = max(h, w)
        top = (h - crop_dim) // 2
        left = (w - crop_dim) // 2
        bbox = np.array([left, top, left + crop_dim, top + crop_dim]) # 计算bbox的两个顶点
    else:
        assert bbox_anno is not None
        bbox = np.array(bbox_anno)

    crop_paras = calculate_crop_parameters(image, bbox, crop_dim, img_size)

    # Crop image by bbox
    image = _crop_image(image, bbox) # 先将image crop成正方形
    # QUESTION: Why force the RGB image to the range of [0, 1] ?
    image_transformed = transform(image).clamp(0.0, 1.0) # 再将 image transfomer成指定的size

    if mask is not None:
        mask = _crop_image(mask, bbox)
        mask_transformed = transform(mask).clamp(0.0, 1.0)
        mask_transformed = 1 - mask_transformed
        mask_transformed[mask_transformed < 0.5] = 0 # 彻底变为0与1
        mask_transformed[mask_transformed >= 0.5] = 1
        assert torch.all((mask_transformed == 0) | (mask_transformed == 1)), "The tensor contains values other than 0 and 1"
    else:
        mask_transformed = None

    return image_transformed, mask_transformed, crop_paras, bbox


def _crop_image(image, bbox, white_bg=False):
    """
    Crop an image to a bounding box. When bbox is larger than the image, the image is padded.

    Args:
        image (PIL.Image.Image): Image to be cropped.
        bbox (np.array): Bounding box for the crop. [left, top, right, bottom]
        white_bg (bool): Flag to indicate if the background should be white.

    Returns:
        PIL.Image.Image: Cropped image.
    """
    if white_bg:
        # Only support PIL Images
        image_crop = Image.new(
            "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
        )
        image_crop.paste(image, (-bbox[0], -bbox[1]))
    else:
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
    return image_crop


if __name__ == "__main__":
    pass