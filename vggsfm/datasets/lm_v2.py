import os
import os.path as osp
import numpy.random as random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import time
from loguru import logger
import copy

from ray_diffusion.utils.bbox import mask_to_bbox

cmap = plt.get_cmap("hsv")

CATEGORIES = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone"
]
LM_DIR = 'lm_data/'

def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.

    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2 # 
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    ) # 有负数，短边的结果为负数
    return square_bbox

def opencv_to_pytorch3d(anno, image_size):
    w, h = image_size
    image_size = np.array(image_size)
    s = min(w, h)
    focal_length = np.array(anno['focal_length']) * 2 / s
    principal_point = -(np.array(anno['principal_point']) - image_size / 2.0) * 2.0 / s

    # opencv to pytorch3d 
    R = anno['R'].transpose(0, 1)
    T = anno['T']
    R[:, :2] *= -1        
    T[:2] *= -1
    new_data = {
        'focal_length': focal_length.tolist(),
        "principal_point": principal_point.tolist(),
        "R": R,
        "T": T
    }
    anno.update(new_data)

    return anno

def pytorch3d_to_opencv(R, T):
    """
    Params:
        R torch.Tensor: (B, 3, 3)
        T torch.Tensor: (B, 3)
    Returns:
        R: (B, 3, 3)
        T: (B, 3)
    """

    T[:, :2] *= -1
    R[:, :, :2] *= -1
    R = R.permute(0, 2, 1)

    return R, T


def merge_batch(train_batch, test_bacth, device="cuda:0"):
    batch = copy.deepcopy(train_batch)
    for key, value in train_batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = torch.concat((train_batch[key].to(device), test_bacth[key].to(device)), dim=0)
        elif key == 'n':
            batch[key] = test_bacth['n']
        elif key == 'filename':
            batch[key].extend(test_bacth[key])
        else:
            pass
    return batch

def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)

def save_batch_images(images, fname):
    cmap = plt.get_cmap("hsv")
    num_frames = len(images)
    num_rows = 2
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows * num_cols):
        if i < num_frames:
            axs[i].imshow(unnormalize_image(images[i]))
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(fname)

def load_pose(pose_path):
    '''
        Load the poses (R, T) 
        pose files format: *-pose.txt
    '''
    import numpy as np
    def read_pose(file):
        with open(file, 'r') as f:
            lines = f.readlines()
            # Assuming each line corresponds to one row of the 3x4 matrix
            matrix_values = [list(map(float, line.split())) for line in lines]
            assert len(matrix_values) == 4 and all(len(row) == 4 for row in matrix_values), "File should contain a 3x4 matrix"
            # Convert to numpy array
            matrix_3x4 = np.array(matrix_values)
            # Extract R and T
            R = matrix_3x4[:-1, :3].reshape(3, 3)  # First 3 columns
            T = matrix_3x4[:-1, 3].reshape(3, )   # Last column
            
        return R, T
    return read_pose(file=pose_path)

def load_bbox(bbox_path):
    """
        Load the bbox
    """
    bbox = []
    with open(bbox_path, 'r') as f:
        bbox = [float(line.strip()) for line in f]
    return bbox

def load_intrinsics():
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    return K

class lm_v2(Dataset):
    def __init__(
        self,
        category,
        lm_dir="lm_data/",
        split='test',
        num_images=8,        
        img_size=224,
        seed=0,        
        transform=None,
        ids=None,
        bounding_boxes=True,        
        mask_images=True,
        apply_augmentation=True,        
        no_images=False,
        crop_images=True,
    ):
        """
        Dataset for custom images. If mask_dir is provided, bounding boxes are extracted
        from the masks. Otherwise, bboxes must be provided.
        """

        start_time = time.time()
        
        self.sequences = {}
        self.dir_dict_map = {}
        self.len_metadata_map = {}
        self.model_path_map = {}
        self.apply_augmentation = apply_augmentation
        self.crop_images = crop_images
        self.img_size = img_size
        self.ids = ids
        self.no_images = no_images
        self.num_images = num_images
        self.transform = transform
        self.category = category
        self.mask_images = mask_images
        self.bounding_boxes = bounding_boxes

        # Fixing seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        if lm_dir is None:
            self.lm_dir = lm_dir
        else:
            self.lm_dir = LM_DIR


        if isinstance(self.category, str):
            self.category = [self.category]
        if "full" in self.category:
            self.category = CATEGORIES
        self.category = sorted(self.category)
        self.is_single_category = len(self.category) == 1


        for cat in self.category:
            dir_dict = self._load_dir(cat, split)
            self.dir_dict_map[cat] = dir_dict
            # todo: 
            self.sequences[cat], counter = self._load_data(dir_dict)
            self.len_metadata_map[cat] = len(self.sequences[cat])
            self.model_path_map[cat] = osp.join(dir_dict['model_dir'], f"{cat}.ply")
            logger.info(f"Loaded {counter} examples of the {cat} category.")

        self.sequence_list = list(self.sequences.keys())

        if self.apply_augmentation:
            self.jitter_scale = [1.15, 1.15]
            self.jitter_trans = [0, 0]
        else:
            self.jitter_scale = [1, 1]
            self.jitter_trans = [0.0, 0.0]
        
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.img_size, antialias=True),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        logger.info(f"Data size: {len(self)}")
        logger.info(f"Data loading took {(time.time() - start_time)} seconds.")

    def __len__(self):
        return len(self.sequence_list)
    
    def _load_dir(self, cat, split):
        dir_dict = {
            "image_dir": osp.join(self.lm_dir, cat, split, "images"),
            "mask_dir": osp.join(self.lm_dir, cat, split, "masks"),
            "pose_dir": osp.join(self.lm_dir, cat, split, "poses"),
            "bbox_dir": osp.join(self.lm_dir, cat, split, "bboxes"),
            "model_dir": osp.join(self.lm_dir, cat, split, "models"),
        }
        return dir_dict
    
    def _load_data(self, dir_dict):
        image_dir = dir_dict['image_dir']
        pose_dir = dir_dict['pose_dir']
        bbox_dir = dir_dict['bbox_dir']
        filtered_data = []

        counter = 0
        for image_name in tqdm(
            sorted(os.listdir(image_dir))
        ):
            counter += 1
            pose_name = image_name.replace(".png", ".txt")
            bbox_name = image_name.replace(".png", ".txt")
            intrinsic = load_intrinsics() # np.array
            pose_path = osp.join(pose_dir, pose_name)
            bbox_path = osp.join(bbox_dir, bbox_name)
            bbox = load_bbox(bbox_path) # list
            R, T= load_pose(pose_path) # np.array
            focal_length = [intrinsic[0, 0], intrinsic[1, 1]]
            principal_point = [intrinsic[0, 2], intrinsic[1, 2]]
            filtered_data.append(
                {
                    "filepath": image_name,
                    "bbox": bbox,
                    "R": R,
                    "T": T,
                    "focal_length": focal_length,
                    "principal_point": principal_point,
                }
            )

        return filtered_data, counter

    def _jitter_bbox(self, bbox):
        bbox = square_bbox(bbox.astype(np.float32)) # 将长方形的bbx变为以最长那条边的长度作为边长的正方形的bbx
        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0] # 边长
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s # jitter后的边的长度的一半

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
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

    def __getitem__(self, index):
        num_to_load = self.num_images
        sequence_name = self.sequence_list[index % len(self.sequence_list)]
        metadata = self.sequences[sequence_name]

        if self.ids is None:
            self.ids = np.random.choice(len(metadata), num_to_load, replace=False)
        

        return self.get_data(index=index, ids=self.ids, no_images=self.no_images)
    
    def get_data(self, index=None, sequence_name=None, ids=(0, 1, 2, 3, 4, 5, 6, 7), no_images=False, select_train=False):
        # if select_train:
        #     ids = np.random.choice(range(0, self.n), 7)

        if sequence_name is None:
            index = index % len(self.sequence_list)
            sequence_name = self.sequence_list[index]
        metadata = self.sequences[sequence_name]
        dir_dict = self.dir_dict_map[sequence_name]
        image_dir = dir_dict['image_dir']
        mask_dir = dir_dict['mask_dir']
        cateory = sequence_name

        # Read image & camera information from annotations
        annos = [metadata[i] for i in ids]
        images = []
        image_sizes = []
        PP = []
        FL = []
        crop_parameters = []
        filenames = []

        for anno in annos:
            filepath = anno['filepath']

            if not no_images:
                image = Image.open(osp.join(image_dir, filepath)).convert("RGB")
                if self.mask_images:
                    black_image = Image.new("RGB", image.size, (0, 0, 0))

                    mask_path = osp.join(mask_dir, filepath)
                    mask = Image.open(mask_path).convert("L")

                    if mask.size != image.size:
                        mask = mask.resize(image.size)
                    mask = Image.fromarray(np.array(mask) > 125)
                    image = Image.composite(image, black_image, mask)
                
                # Determine crop, Resnets want square images
                bbox_init = (
                    anno['bbox']
                    if self.crop_images
                    else [0, 0, image.width, image.height]
                )
                bbox = square_bbox(np.array(bbox_init))
                if self.apply_augmentation:
                    bbox = self._jitter_bbox(bbox)
                bbox = np.around(bbox).astype(int)

                # Crop parameters
                crop_center = (bbox[:2] + bbox[2:]) / 2
                # convert crop center to correspond to a "square" image
                width, height = image.size
                length = max(width, height)
                s = length / min(width, height)
                crop_center = crop_center + (length - np.array([width, height])) / 2
                # convert to NDC
                cc = s - 2 * s * crop_center / length
                crop_width = 2 * s * (bbox[2] - bbox[0]) / length
                crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])

                # opencv to pytorch3d
                # anno = opencv_to_pytorch3d(anno, image.size)

                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

                # Crop and normalize image
                image = self._crop_image(image, bbox)
                image = self.transform(image)
                images.append(image[:, : self.img_size, : self.img_size])
                crop_parameters.append(crop_params)

            else:
                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

            PP.append(principal_point)
            FL.append(focal_length)
            image_sizes.append(torch.tensor([self.img_size, self.img_size]))
            filenames.append(filepath)

        if not no_images:
            images = torch.stack(images)
            crop_parameters = torch.stack(crop_parameters)
        else:
            images = None
            crop_parameters = None

        # Assemble batch info to send back
        R = torch.stack([torch.tensor(anno["R"]) for anno in annos])
        T = torch.stack([torch.tensor(anno["T"]) for anno in annos])
        focal_lengths = torch.stack(FL)
        principal_points = torch.stack(PP)
        image_sizes = torch.stack(image_sizes)

        batch = {
            "model_id": sequence_name,
            "category": cateory,
            "n": len(metadata),
            "ind": torch.tensor(ids),
            "image": images,
            "R": R,
            "T": T,
            "focal_length": focal_lengths,
            "principal_point": principal_points,
            "image_size": image_sizes,
            "crop_parameters": crop_parameters,
            "filename": filenames,
        }

        return batch

if __name__ == "__main__":
    
    dataset_train = lm_v2(
    category=CATEGORIES,
    split='train',
    bounding_boxes=True,
    mask_images=False,
    num_images=7,
    )

    dataset_test = lm_v2(
    category=CATEGORIES,
    split='test',
    bounding_boxes=True,
    mask_images=False,
    num_images=1,
    )

    train_batch = dataset_train[4]
    test_batch = dataset_test.get_data(index=4, ids=np.arange(9, 10), no_images=False)
    batch = merge_batch(train_batch, test_batch)

    print("train: ", train_batch['filename'])
    print("train: ", train_batch['ind'])
    print("test: ", test_batch['filename'])
    print("batch: ", batch['filename'])
    print("batch: ", batch['image'].shape)
    print("batch: ", batch['n'])
    print("batch: ", batch['ind'])
    print("batch: ", batch['R'].shape)
    print("batch: ", batch['T'].shape)
    print("batch: ", batch['image'].shape)
    print("batch: ", batch['principal_point'].shape)
    print("batch: ", batch['focal_length'].shape)
    print("batch: ", batch['image_size'].shape)
    print("batch: ", batch['crop_parameters'].shape)
    
    
    
    
    
    # save_batch_images(batch['image'], "temp.png")

    # sequences = dataset.sequences
    # for key in sequences.keys():
    #     sequence = sequences[key]
    #     print(f"{key}: ", sequence.keys())
    #     print("len R: ", len(sequence['R']))
    #     print("len T: ", len(sequence['T']))
    #     print("n: ", sequence['n'])
    #     print("image path: ", sequence['image_path'][:10])
    # # print(batch['R'].shape)
    # print(batch['T'].shape)
    # print(batch['focal_length'].shape)
    # print(batch['principal_point'].shape)
    # print(batch['image_size'].shape)
    # print(batch['focal_length'])

    
    # crop_params = batch["crop_params"] 
    # print(crop_params)

    # view_color_coded_images_from_tensor(images)
