import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_cluster import knn_graph
from scipy.spatial import distance
from models.distributed_sampler_no_evenly_divisible import *
from sklearn.neighbors import NearestNeighbors
import cv2

f = 1266.417203046554
cx = 816.2670197447984
cy = 491.50706579294757

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})



def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

def radar_lidar_image_collection(batch):
    
    radar_points = [sample['radar_points'] for sample in batch]
    radar_channels = [sample['radar_channels'] for sample in batch]
    lidar_points = [sample['lidar_points'] for sample in batch]
    centroid = [sample['centroid'] for sample in batch]
    furthest_distance = [sample['furthest_distance'] for sample in batch]
    depth = [sample['depth'] for sample in batch]
    nointer_depth = [sample['nointer_depth'] for sample in batch]
    image_data = [sample['image'] for sample in batch]
    focal = [sample['focal'] for sample in batch]
    K = [sample['K'] for sample in batch]

    radar_points_batch = Batch.from_data_list(radar_points)
    radar_channels_batch = torch.stack(radar_channels)
    lidar_points_batch = torch.stack(lidar_points)
    if centroid[0] is not None:
        centroid_batch = torch.stack(centroid)
        furthest_distance_batch = torch.stack(furthest_distance)
    else:
        centroid_batch = None
        furthest_distance_batch = None
    image_batch = torch.stack(image_data)
    depth_batch = torch.stack(depth)
    nointer_depth_batch = torch.stack(nointer_depth)
    focal = torch.tensor(focal)
    K_batch = torch.stack(K)


    return {'image': image_batch, 'depth': depth_batch, 'nointer_depth': nointer_depth_batch, 'radar_channels': radar_channels_batch, \
            'radar_points': radar_points_batch, 'focal': focal, 'lidar_points': lidar_points_batch, \
            'centroid': centroid_batch, 'furthest_distance': furthest_distance_batch, 'K':K_batch}


class GCNRADDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.train_image_path = args.train_image_path
            self.train_radar_path = args.train_radar_path
            self.train_ground_truth_path = args.train_ground_truth_path
            self.train_ground_truth_nointer_path = args.train_ground_truth_nointer_path
            self.train_lidar_path = args.train_lidar_path
            train_image_paths = read_paths(self.train_image_path)
            train_radar_paths = read_paths(self.train_radar_path)
            train_ground_truth_paths = read_paths(self.train_ground_truth_path)
            train_ground_truth_nointer_paths = read_paths(self.train_ground_truth_nointer_path)
            train_lidar_paths = read_paths(self.train_lidar_path)
        
        self.test_image_path = args.test_image_path
        self.test_radar_path = args.test_radar_path
        self.test_ground_truth_path = args.test_ground_truth_path
        self.main_path = args.main_path
        test_image_paths = read_paths(self.test_image_path)
        test_radar_paths = read_paths(self.test_radar_path)
        test_ground_truth_paths = read_paths(self.test_ground_truth_path)

        if mode == 'train':

            self.training_samples = DataLoadPreprocess(args, mode, ground_truth_paths=train_ground_truth_paths, \
            ground_truth_nointer_paths=train_ground_truth_nointer_paths, main_path=self.main_path,\
            image_paths=train_image_paths, radar_paths=train_radar_paths, lidar_paths=train_lidar_paths,\
            ground_truth_paths_eval=test_ground_truth_paths, image_paths_eval=test_image_paths,\
            radar_paths_eval=test_radar_paths, transform=preprocessing_transforms(mode))

            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                    shuffle=(self.train_sampler is None),
                                    num_workers=args.num_threads,
                                    pin_memory=True,
                                    sampler=self.train_sampler,
                                    collate_fn=radar_lidar_image_collection)

        else:
            self.testing_samples = DataLoadPreprocess(args, mode, ground_truth_paths=None, \
                image_paths=None, radar_paths=None, main_path=self.main_path,\
                ground_truth_paths_eval=test_ground_truth_paths, image_paths_eval=test_image_paths,\
                radar_paths_eval=test_radar_paths, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None

            self.data = DataLoader(self.testing_samples, 1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True,
                                sampler=self.eval_sampler,
                                collate_fn=radar_lidar_image_collection)

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, main_path=None, ground_truth_paths=None, ground_truth_nointer_paths=None,\
                 image_paths=None, radar_paths=None, lidar_paths=None, ground_truth_paths_eval=None,\
                 image_paths_eval=None, radar_paths_eval=None, lidar_paths_eval=None, transform=None):
        self.args = args
        self.main_path = main_path
        self.ground_truth_paths = ground_truth_paths
        self.ground_truth_nointer_paths = ground_truth_nointer_paths
        self.image_paths = image_paths
        self.radar_paths = radar_paths
        self.lidar_paths = lidar_paths
        self.ground_truth_paths_eval = ground_truth_paths_eval
        self.image_paths_eval = image_paths_eval
        self.radar_paths_eval = radar_paths_eval
        self.lidar_paths_eval = lidar_paths_eval
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor

    def __getitem__(self, idx):
        focal = float(567.0)

        if self.mode == 'train':
            K = np.array([
                [f, 0, cx, 0],
                [0, f, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            image_path = self.image_paths[idx]
            radar_path = self.main_path + self.radar_paths[idx]
            depth_path = self.main_path + self.ground_truth_paths[idx]
            nointer_depth_path = self.main_path + self.ground_truth_nointer_paths[idx]

            lidar_path = self.main_path + self.lidar_paths[idx]
            lidar = Image.open(lidar_path)
            lidar = np.asarray(lidar, dtype=np.float32)
            lidar = np.expand_dims(lidar, axis=2)
            lidar = lidar / 256.0

            image = Image.open(image_path)
            image = np.asarray(image, dtype=np.float32) / 255.0
            width = image.shape[1]
            height = image.shape[0]

            depth_gt = Image.open(depth_path)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 256.0

            nointer_depth_gt = Image.open(nointer_depth_path)
            nointer_depth_gt = np.asarray(nointer_depth_gt, dtype=np.float32)
            nointer_depth_gt = np.expand_dims(nointer_depth_gt, axis=2)
            nointer_depth_gt = nointer_depth_gt / 256.0

            radar_points_2d = np.load(radar_path)
            radar_channels = np.zeros((height, width, radar_points_2d.shape[-1]-3), dtype=np.float32)

            for i in range(radar_points_2d.shape[0]):
                x = int(radar_points_2d[i, 0])
                y = int(radar_points_2d[i, 1])
                radar_depth = radar_points_2d[i, 2]
                # last feature is alignment, not useful in this project
                
                # generate radar channels
                if radar_channels[y, x, 0] == 0:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]
                elif radar_channels[y, x, 0] > radar_points_2d[i, 2]:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]
                elif radar_channels[y, x, -1] == 0 and radar_points_2d[i, -1] != 0:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]

            # random crop & augments
            image, depth_gt, nointer_depth_gt, radar_channels,  K, lidar_channels = self.random_crop(image, depth_gt, nointer_depth_gt, radar_channels, self.args.input_height, self.args.input_width, K, lidar)
            image, depth_gt, nointer_depth_gt, radar_channels, K, lidar_channels = self.train_preprocess(image, depth_gt, nointer_depth_gt, radar_channels, K, lidar_channels)
            
            lidar_points = self.channel_back_to_points(lidar_channels)
            radar_points2d_crop = self.channel_back_to_points(radar_channels)
            # first project 2d radar points back to 3d plane, without normalization
            radar_points3d_crop = self.point2d_to_3d(radar_points2d_crop, K)
            # also project 2d lidar points back to 3d
            lidar_points3d_all = self.point2d_to_3d(lidar_points, K)
            
            # if num_lidar points > args.lidar_points
            if lidar_points3d_all.shape[0] >= self.args.lidar_points and radar_points3d_crop.shape[0] != 0:
                dist = distance.cdist(radar_points3d_crop[:, :3], lidar_points3d_all[:, :3]) #(num_radar, num_lidar)
                flattened_dist = dist.ravel()
                radar_indices, lidar_indices = np.unravel_index(np.argsort(flattened_dist), dist.shape)
                unique_lidar_indices = set()
                selected_indices = []
                for radar_idx, lidar_idx in zip(radar_indices, lidar_indices):
                    if lidar_idx not in unique_lidar_indices and len(unique_lidar_indices) < self.args.lidar_points:
                        unique_lidar_indices.add(lidar_idx)
                        selected_indices.append(lidar_idx)
                    if len(unique_lidar_indices) >=self.args.lidar_points:
                        break
                selected_lidar_points = np.array(selected_indices).astype(np.int32)
                lidar_points3d = lidar_points3d_all[selected_lidar_points]
                
            else:
                if lidar_points3d_all.shape[0] < self.args.lidar_points and lidar_points3d_all.shape[0] != 0:
                    lidar_points3d_all = np.repeat(lidar_points3d_all, self.args.lidar_points//lidar_points3d_all.shape[0], axis=0)
                elif lidar_points3d_all.shape[0] == 0:
                    lidar_points3d_all = np.zeros((self.args.lidar_points+1, lidar_points3d_all.shape[1]))
                    lidar_points[:, 2:3] = 1e-3
                random_idx_pos = np.random.randint(lidar_points3d_all.shape[0], size=self.args.lidar_points)
                lidar_points3d = lidar_points3d_all[random_idx_pos, :]

            if self.args.norm_point:
                # normalize lidar point cloud
                lidar_points3d, centroid, furthest_distance = self.normalize_point_cloud(lidar_points3d[:, :3])

                # normalize radar point cloud itself
                if radar_points3d_crop.shape[0] != 0:
                    radar_points3d_crop_xyz, _, _ = self.normalize_point_cloud(radar_points3d_crop[:, :3])
                    radar_points3d_crop[:, :3] = radar_points3d_crop_xyz
            else:
                centroid = None
                furthest_distance = None
            
            # if no radar points, add one with all zeros
            if radar_points3d_crop.shape[0] == 0:
                radar_points3d_crop = np.zeros((1, radar_points3d_crop.shape[1]))


            radar_points3d_crop = torch.from_numpy(radar_points3d_crop).float()
            edge_index = knn_graph(radar_points3d_crop[:, :3], k=self.args.k)
            data = Data(x=radar_points3d_crop, edge_index=edge_index)

            sample = {'image': image, 'depth': depth_gt, 'nointer_depth': lidar_channels, 'radar_channels': radar_channels, 'radar_points': data, 'focal': focal,\
                        'lidar_points': lidar_points3d, 'centroid': centroid, 'furthest_distance': furthest_distance, 'K': K}

        else:
            K = np.array([
                [f, 0, cx, 0],
                [0, f, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            image_path = self.image_paths_eval[idx]
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            depth_path = self.main_path + self.ground_truth_paths_eval[idx]
            depth_gt = Image.open(depth_path)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 256.0

            radar_path = self.main_path + self.radar_paths_eval[idx]
            radar_points = np.load(radar_path)

            radar_channels = np.zeros((image.shape[0], image.shape[1], radar_points.shape[-1]-3), dtype=np.float32)
            for i in range(radar_points.shape[0]):
                x = int(radar_points[i, 0])
                y = int(radar_points[i, 1])
                if radar_channels[y, x, 0] == 0:
                    radar_channels[y, x] = radar_points[i, 2:-1]
                elif radar_channels[y, x, 0] > radar_points[i, 2]:
                    radar_channels[y, x] = radar_points[i, 2:-1]
                elif radar_channels[y, x, -1] == 0 and radar_points[i, -1] != 0:
                    radar_channels[y, x] = radar_points[i, 2:-1]

            # crop
            image = image[4:, ...] # (894, 1600, 3)
            depth_gt = depth_gt[4:, ...]
            radar_channels = radar_channels[4:, ...]
            K[1, 2] = K[1, 2] - 4

            radar_points2d = self.channel_back_to_points(radar_channels)
            radar_points3d = self.point2d_to_3d(radar_points2d, K)

            # extract lidar points
            lidar_points = self.channel_back_to_points(depth_gt)
            lidar_points3d_all= self.point2d_to_3d(lidar_points, K, normalize=False)
            dist = distance.cdist(radar_points3d[:, :3], lidar_points3d_all[:, :3]) #(num_radar, num_lidar)
            flattened_dist = dist.ravel()
            radar_indices, lidar_indices = np.unravel_index(np.argsort(flattened_dist), dist.shape)
            unique_lidar_indices = set()
            selected_indices = []
            for radar_idx, lidar_idx in zip(radar_indices, lidar_indices):
                if lidar_idx not in unique_lidar_indices and len(unique_lidar_indices) < self.args.lidar_points:
                    unique_lidar_indices.add(lidar_idx)
                    selected_indices.append(lidar_idx)
                if len(unique_lidar_indices) >=self.args.lidar_points:
                    break
            selected_lidar_points = np.array(selected_indices).astype(np.int32)
            lidar_points3d = lidar_points3d_all[selected_lidar_points]

            if self.args.norm_point:
            
                lidar_points3d, centroid, furthest_distance = self.normalize_point_cloud(lidar_points3d[:, :3])
                
                radar_points3d_xyz, _, _ = self.normalize_point_cloud(radar_points3d[:, :3])
                radar_points3d[:, :3] = radar_points3d_xyz
            else:
                centroid = None
                furthest_distance = None


            radar_points3d = torch.from_numpy(radar_points3d).float()
            edge_index = knn_graph(radar_points3d[:, :3], k=self.args.k)
            data = Data(x=radar_points3d, edge_index=edge_index)

            sample = {'image': image, 'depth': depth_gt, 'nointer_depth': depth_gt, 'radar_channels': radar_channels, 'radar_points': data, 'focal': focal, \
                        'lidar_points': lidar_points3d, 'centroid': centroid, 'furthest_distance': furthest_distance, 'K': K}
            
        sample = self.transform(sample)

        return sample

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_paths)
        else:
            return len(self.image_paths_eval)
        
    def normalize_point_cloud(self, inputs, centroid=None, furthest_distance=None):
        """
        input: pc [N, P, 3]
        output: pc, centroid, furthest_distance
        """
        if centroid is not None:
            inputs = inputs - centroid
            inputs = inputs / furthest_distance
            return inputs
        else:
            if len(inputs.shape) == 2:
                axis = 0
            elif len(inputs.shape) == 3:
                axis = 1
            centroid = np.mean(inputs, axis=axis, keepdims=True)
            inputs = inputs - centroid
            furthest_distance = np.amax(inputs[:, -2:-1], axis=axis, keepdims=True)
            if furthest_distance[0,0] < 1e-3:
                furthest_distance = np.array([[1e-3]])
            inputs = inputs / furthest_distance
            return inputs, centroid, furthest_distance
    
    def point2d_to_3d(self, radar_points, K, normalize=False):
        viewpad_inv = np.linalg.inv(K)
        depth = radar_points[:,2:3]
        radar_point3d = np.concatenate((radar_points[:, 0:2], np.ones((depth.shape[0], 1)), 1.0/depth), axis=-1)
        radar_point3d = np.transpose(radar_point3d)    
        radar_point3d = depth.transpose().repeat(4, 0).reshape(4, -1) * np.dot(viewpad_inv, radar_point3d) # (4, N)
        if normalize:
            point3d = radar_point3d[:3, :].transpose()
            point3d, centroid, furthest_distance = self.normalize_point_cloud(point3d)
            radar_point3d = np.concatenate((point3d, radar_points[:, 3:], radar_points[:, 0:2]), axis=-1)
            return radar_point3d, centroid, furthest_distance

        else:
            radar_point3d = np.concatenate((radar_point3d[:3, :].transpose(), radar_points[:, 3:], radar_points[:, 0:2]), axis=-1)
            return radar_point3d

    def channel_back_to_points(self, radar_channels):
        y, x = np.where(radar_channels[..., 0] != 0)
        radar_points = np.concatenate([x[:, None], y[:, None], radar_channels[y, x]], axis=-1) # x, y, depth, rcs, vx, vy

        return radar_points

    def random_crop(self, img, depth, nointer_depth, rad, height, width, K, lidar=None):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)

        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        nointer_depth = nointer_depth[y:y + height, x:x + width, :]
        rad = rad[y:y + height, x:x + width, :]
        K[0, 2] = K[0, 2]- x
        K[1, 2] = K[1, 2] - y

        if lidar is not None:
            lidar = lidar[y:y + height, x:x + width, :]
            return img, depth, nointer_depth, rad, K, lidar

        return img, depth, nointer_depth, rad, K
    

    def train_preprocess(self, image, depth_gt, nointer_depth_gt, radar, K, lidar=None):
        # Random flipping
        w = image.shape[1]
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            nointer_depth_gt = (nointer_depth_gt[:, ::-1, :]).copy()
            radar = (radar[:, ::-1, :]).copy()
            K[0, 2] = w - K[0, 2] - 1
            if lidar is not None:
                lidar = (lidar[:, ::-1, :]).copy()
            
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        if lidar is not None:
            return image, depth_gt, nointer_depth_gt, radar, K, lidar

        return image, depth_gt, nointer_depth_gt, radar, K

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, radar_channels, radar_points, focal = sample['image'], sample['radar_channels'], sample['radar_points'], sample['focal']
        image = self.to_tensor(image)
        radar_channels = self.to_tensor(radar_channels)
        image = self.normalize(image)

        depth = sample['depth']
        depth = self.to_tensor(depth)

        nointer_depth = sample['nointer_depth']
        nointer_depth = self.to_tensor(nointer_depth)


        lidar_points = sample['lidar_points']
        lidar_points = torch.from_numpy(lidar_points).float()

        centroid = sample['centroid']
        furthest_distance = sample['furthest_distance']

        if centroid is not None:
            centroid = torch.from_numpy(centroid).float()   
            furthest_distance = torch.from_numpy(furthest_distance).float()

        K = sample['K']
        K = torch.from_numpy(K).float()

        return {'image': image, 'depth': depth, 'nointer_depth': nointer_depth, 'radar_channels': radar_channels, 'radar_points': radar_points,\
                'focal': focal, 'lidar_points': lidar_points, 'centroid': centroid, 'furthest_distance': furthest_distance, 'K': K}

  
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            if len(pic.shape) > 2:
                img = torch.from_numpy(pic.transpose((2, 0, 1)))
                return img
            else:
                arr = torch.from_numpy(pic)
                return arr
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
