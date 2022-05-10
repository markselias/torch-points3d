import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
from torch_geometric.datasets import S3DIS as S3DIS1x1
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
import open3d as o3d

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

# from ipb_planters_toolkit import data_processor
# from ipb_planters_toolkit.utils import SerializablePcdT, pcd_select_by_id, visualize_o3d, SerializablePcd, SerializableMesh, cache, np2o3d

class SerializablePcdT:
    def __init__(self, pcd: o3d.t.geometry.PointCloud, attributes=["colors", "normals", "leaf_ids", "plant_ids", "confidence", "label", "keypt_ids"]):
        self.attributes = attributes
        self.points = pcd.point["positions"].numpy()
        for attr in self.attributes:
            if pcd.point.__contains__(attr):
                setattr(self, attr, pcd.point[attr].numpy())

    def to_open3d(self) -> o3d.t.geometry.PointCloud:
        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(self.points)
        for attr in self.attributes:
            if hasattr(self, attr):
                pcd.point[attr] = o3d.core.Tensor(getattr(self, attr))
        return pcd
    
def pcd_select_by_id(pcd, ids):
    ids = np.squeeze(ids)
    attributes=["colors", "normals", "leaf_ids", "plant_ids", "confidence", "label"]

    pcd.point["positions"] = o3d.core.Tensor(pcd.point["positions"].numpy()[ids])
    for attr in attributes:
        if pcd.point.__contains__(attr):
            pcd.point[attr] = o3d.core.Tensor(pcd.point[attr].numpy()[ids])

    # pcd.point["colors"] = o3d.core.Tensor(pcd.point["colors"].numpy()[ids])
    # pcd.point["normals"] = o3d.core.Tensor(pcd.point["normals"].numpy()[ids])
    # pcd.point["confidence"] = o3d.core.Tensor(pcd.point["confidence"].numpy()[ids])
    return pcd

class DataProcessor:
    def readPcd(self, pcd_path, center_from_points=True, mm_format=False, 
                 estimate_normals=False, parse_labels=False, parse_keypoints=False,
                 downsample_size=0.001, center_on_origin=True, n_keypoint_types=4, return_center_points=False):
        # if not center_from_points and plant_center_path == None:
        #     raise ValueError("No plant center path specified and it's not computed from points.")
        # loading full plant
        if not os.path.isfile(pcd_path):
            raise Exception("Can't find pointcloud at {}.".format(pcd_path))
        print("Loading {}...".format(pcd_path))

        ply_pcd = o3d.t.io.read_point_cloud(pcd_path)

        if center_from_points:
            if ply_pcd.point.__contains__("leaf_ids"):
                plant_center_id = -1
                center_points = ply_pcd.point["positions"].numpy()[np.squeeze(ply_pcd.point["leaf_ids"].numpy()) == -1]
                plant_center = np.mean(center_points, axis=0)
                plant_center = o3d.core.Tensor(plant_center, dtype=o3d.core.Dtype.Float32)
                plant_center = plant_center.numpy()
            else:
                if center_on_origin:
                    raise ValueError("Requested to center on origin but no center is provided.")
                plant_center = None
        else:
            # centroid
            plant_center = ply_pcd.get_center()
            # lower the center to minimum z
            plant_center[2] = ply_pcd.get_min_bound()[2]
            plant_center = plant_center.numpy()


        if parse_keypoints:
            leaf_list = self.compute_leaf_list(ply_pcd)
            
            # keypoints = ply_pcd.get_keypoints()
            plant_keypoints = np.zeros((np.max(leaf_list).astype(int)+1, n_keypoint_types, 3))
            
            for leaf_it in leaf_list:
                if leaf_it < 0:
                    continue
                leaf_mask = torch.tensor(ply_pcd.point["leaf_ids"].numpy()) == leaf_it
                if ply_pcd.point.__contains__("keypt_ids"):
                    keypoint_field = "keypt_ids"
                    keypts_mask = torch.tensor(
                        ply_pcd.point[keypoint_field].numpy()) == torch.tensor([1, 2, 3, 4])
                elif ply_pcd.point.__contains__("keypoint_ids"):
                    keypoint_field = "keypoint_ids"
                    keypts_mask = torch.tensor(
                        ply_pcd.point[keypoint_field].numpy()) == torch.tensor([0,1,2,3])
                else:
                    assert(False, "No keypoint field in pcd")
                combined_mask = torch.logical_and(leaf_mask, keypts_mask)
                
                plant_pcd = torch.tensor(ply_pcd.point["positions"].numpy())
                
                # for every leaf compute the centroid of the points labeled as keypoints
                leaf_keypoints = np.zeros((n_keypoint_types, 3))
                for i in range(4):
                    leaf_keypoints[i,:] = torch.mean(plant_pcd[combined_mask[:, i]], axis=0)
                plant_keypoints[int(leaf_it)] = leaf_keypoints
            if center_on_origin:
                plant_keypoints -= plant_center
                
        # downsample pcd
        ply_pcd.voxel_down_sample(voxel_size=downsample_size)
        
        if center_on_origin:
            ply_pcd.translate(0-plant_center)
        if mm_format:
            ply_pcd.scale(1000, o3d.core.Tensor((0.,0.,0.), dtype=o3d.core.Dtype.Float32))
            plant_keypoints *= 1000

        # # downsample (and estimate normals)
        # if estimate_normals:
        #     full_pcd = self.preprocessPcd(full_pcd)

        # convert to np for caching
        full_pcd = SerializablePcdT(ply_pcd)

        # finally translate also the plant center :)
        if center_on_origin:
            plant_center-=plant_center
            
        return_elements = [full_pcd, plant_center]

        if parse_keypoints:
            return_elements.append(plant_keypoints)
        
        if return_center_points:
            return_elements.append(center_points)
        
        return return_elements


    def writePcd(self, pcd_path, points, colors, labels=None):
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point["positions"] = o3d.core.Tensor(points, dtype, device)
        pcd.point["colors"] = o3d.core.Tensor(colors, dtype, device)        
        if labels is not None: 
            pcd.point["leaf_ids"] = o3d.core.Tensor(labels, o3d.core.int32, device)    
        
        o3d.t.io.write_point_cloud(pcd_path, pcd)
        
    def compute_leaf_list(self, pcd):
        leaf_label_list = np.unique(
                    pcd.point["leaf_ids"].numpy())
        # filter nans
        leaf_label_list = leaf_label_list[~np.isnan(leaf_label_list)]
        # remove plant center
        leaf_label_list = leaf_label_list[leaf_label_list >= 0]
        return leaf_label_list

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)
data_proc = DataProcessor()

SBUB3D_NUM_CLASSES = 13

INV_OBJECT_LABEL = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}

OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'ceiling' .-> .yellow
        [95, 156, 196],  # 'floor' .-> . blue
        [179, 116, 81],  # 'wall'  ->  brown
        [241, 149, 131],  # 'beam'  ->  salmon
        [81, 163, 148],  # 'column'  ->  bluegreen
        [77, 174, 84],  # 'window'  ->  bright green
        [108, 135, 75],  # 'door'   ->  dark green
        [41, 49, 101],  # 'chair'  ->  darkblue
        [79, 79, 76],  # 'table'  ->  dark grey
        [223, 52, 52],  # 'bookcase'  ->  red
        [89, 47, 95],  # 'sofa'  ->  purple
        [81, 109, 114],  # 'board'   ->  grey
        [233, 233, 229],  # 'clutter'  ->  light grey
        [0, 0, 0],  # unlabelled .->. black
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}


VALIDATION_ROOMS = [
    "hallway_1",
    "hallway_6",
    "hallway_11",
    "office_1",
    "office_6",
    "office_11",
    "office_16",
    "office_21",
    "office_26",
    "office_31",
    "office_36",
    "WC_2",
    "storage_1",
    "storage_5",
    "conferenceRoom_2",
    "auditorium_1",
]

################################### UTILS #######################################

################################### s1m cylinder s3di ###################################


class S3DIS1x1Dataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        pre_transform = self.pre_transform
        self.train_dataset = S3DIS1x1(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            pre_transform=self.pre_transform,
            transform=self.train_transform,
        )
        self.test_dataset = S3DIS1x1(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            pre_transform=pre_transform,
            transform=self.test_transform,
        )
        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


################################### Used for fused s3dis radius sphere ###################################


class SBUBData(InMemoryDataset):
    """ Sugar Beets Uni Bonn dataset loader. This dataset contains point clouds of sugar beet breeding plots.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    split: str
        can be one of train, trainval, val or testS
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """
    download_url = "/media/elias/Data/SBUB3D.zip"
    zip_name = "SBUB3D.zip"
    file_name = "SBUB3D"
    # zip_name = "iros_sbub.zip"
    # file_name = "iros_sbub"
    folders = ["train", "val", "test"]
    num_classes = SBUB3D_NUM_CLASSES

    def __init__(
        self,
        root,
        split="train",
        sample_per_epoch=100,
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        # self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self._sample_per_epoch = sample_per_epoch
        super(SBUBData, self).__init__(root, transform, pre_transform, pre_filter)
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))
        self._load_data(path)

        # if split == "test":
        #     self.raw_test_data = torch.load(self.raw_areas_paths[test_area - 1])

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        return self.folders

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        return [os.path.join(self.processed_dir, "%s.pt" % i) for i in ["train", "val", "test"]]

    @property
    def processed_file_names(self):
        # test_area = self.test_area
        return (
            # ["{}_{}.pt".format(s, test_area) for s in ["train", "val", "test", "trainval"]]
            self.raw_areas_paths
            + [self.pre_processed_path]
        )

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            if not os.path.exists(osp.join(self.root, self.zip_name)):
                # log.info("WARNING: You are downloading SBUB3D dataset")
                # shutil.copyfile("/media/elias/Data/SBUB3D.zip", osp.join(self.root, self.zip_name))
                print("copy manually you lazy ass")
            extract_zip(os.path.join(self.root, self.zip_name), self.root)
            shutil.rmtree(self.raw_dir)
            os.rename(osp.join(self.root, self.file_name), self.raw_dir)
            # shutil.copy(self.path_file, self.raw_dir)
            # cmd = "patch -ruN -p0 -d  {} < {}".format(self.raw_dir, osp.join(self.raw_dir, "s3dis.patch"))
            # os.system(cmd)
        else:
            intersection = len(set(self.folders).intersection(set(raw_folders)))
            if intersection != 3:
                shutil.rmtree(self.raw_dir)
                os.makedirs(self.raw_dir)
                self.download()
                
    def process_one(self, file):
        plant_pcd, center, keypoints = data_proc.readPcd(os.path.join(
        self.root, "raw", file), parse_labels=True, center_on_origin=False, parse_keypoints=True, downsample_size=0.001)
        
        plant_pcd = plant_pcd.to_open3d()
        if plant_pcd.point.__contains__("plant_ids"):
            plant_mask = np.where(plant_pcd.point["plant_ids"].numpy() >= 0)[0]
            plant_pcd = pcd_select_by_id(plant_pcd, plant_mask)
        
        current_leaf_label_list = data_proc.compute_leaf_list(plant_pcd)
        
        
        # keypoint_vectors = self.compute_keypoint_vectors(plant_pcd, keypoints, current_leaf_label_list)
        
        xyz = torch.tensor(plant_pcd.point["positions"].numpy()).float()
        semantic_labels = torch.ones(len(plant_pcd.point["positions"].numpy())).long().squeeze()
        rgb = torch.tensor(plant_pcd.point["colors"].numpy()).float()
        data_type = file.split("/")[0]
        
        data = Data(pos=xyz, y=semantic_labels, rgb=rgb)
        data.set = data_type
        instance_labels = torch.tensor(plant_pcd.point["leaf_ids"].numpy()).long().squeeze()
        data.instance_labels = instance_labels
        # data.grid_size = torch.tensor(1)
        return data

    def process(self):
        if not os.path.exists(self.pre_processed_path):
            train_files = ["train/"+f for f in os.listdir(osp.join(self.root, "raw", "train"))]
            val_files = ["val/"+f for f in os.listdir(osp.join(self.root, "raw", "val"))]
            test_files = ["test/"+f for f in os.listdir(osp.join(self.root, "raw", "test"))]
            
            data_list = []
            
            
            
            for file in tq(train_files + val_files + test_files):
                data = self.process_one(file)
                data_list.append(data)
                
            # raw_areas = cT.PointCloudFusion()(data_list)
            # for i, area in enumerate(raw_areas):
            #     torch.save(area, self.raw_areas_paths[i])

            # for area_datas in data_list:
            #     # Apply pre_transform
            #     if self.pre_transform is not None:
            #         for data in area_datas:
            #             data = self.pre_transform(data)
            torch.save(data_list, self.pre_processed_path)
        else:
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return
        

        train_data_list = []
        val_data_list = []
        trainval_data_list = []
        test_data_list = []
        # for i in range(3):
        #     train_data_list[i] = []
        #     val_data_list[i] = []
        for data in data_list:
            data_set = data.set
            del data.set
            if data_set == "val":
                val_data_list.append(data)
            elif data_set == "train":
                train_data_list.append(data)
            elif data_set == "test":
                test_data_list.append(data)
        trainval_data_list = val_data_list + train_data_list

        # train_data_list = list(train_data_list.values())
        # val_data_list = list(val_data_list.values())
        # trainval_data_list = list(trainval_data_list.values())
        # test_data_list = list(test_data_list.values())
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)

        self._save_data(train_data_list, val_data_list, test_data_list, trainval_data_list)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        # torch.save(self.collate(train_data_list), self.processed_paths[0])
        # torch.save(self.collate(val_data_list), self.processed_paths[1])
        # torch.save(self.collate(test_data_list), self.processed_paths[2])
        # torch.save(self.collate(trainval_data_list), self.processed_paths[3])
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self.data = torch.load(path)
        
    # def __init__(self, root, sample_per_epoch=100, *args, **kwargs):
    #     self._sample_per_epoch = sample_per_epoch
    #     super().__init__(root, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def len(self):
        return len(self)

    def get(self, idx):
        data = self._get_item(idx, subsample=True)
        setattr(data, "sampleid", torch.tensor([idx]))
        return data
    
    def _get_item(self, idx, subsample=False):
        chosen_plant = idx % len(self.data)
        data = self.data[chosen_plant]
        
        # subsampling mask
        num_points = data.pos.shape[0]
        mask = np.zeros(num_points, dtype='bool')
        mask[:100000] = 1
        np.random.shuffle(mask)
        
        indices = torch.tensor(mask)
        
        if subsample:
            new_data = Data()
            for key in data.keys:
                if key == "kd_tree":
                    continue
                item = data[key]
                if torch.is_tensor(item) and num_points == item.shape[0]:
                    item = item[indices].clone()
                elif torch.is_tensor(item):
                    item = item.clone()
                setattr(new_data, key, item)
        else:
            new_data = data
            
        # random_r = Rotation.random().as_matrix()
        # points = random_r.dot(new_data.pos.T).T.astype('float32')
        
        # new_data.pos = points
        return new_data
    
    def _get_random(self):
        chosen_plant = np.random.randint(len(self.data))
        data = self.data[chosen_plant]
        
        # subsampling mask
        num_points = data.pos.shape[0]
        mask = np.zeros(num_points, dtype='bool')
        mask[:100000] = 1
        np.random.shuffle(mask)
        
        indices = torch.tensor(mask)
        
        
        new_data = Data()
        for key in data.keys:
            if key == "kd_tree":
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[indices].clone()
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
            
        # random_r = Rotation.random().as_matrix()
        # points = random_r.dot(new_data.pos.T).T.astype('float32')
        
        # new_data.pos = points
        return new_data

    # def _load_data(self, path):
    #     self._datas = torch.load(path)
    #     if not isinstance(self._datas, list):
    #         self._datas = [self._datas]
    #     if self._sample_per_epoch > 0:
    #         self._centres_for_sampling = []
    #         for i, data in enumerate(self._datas):
    #             assert not hasattr(
    #                 data, cT.SphereSampling.KDTREE_KEY
    #             )  # Just to make we don't have some out of date data in there
    #             low_res = self._grid_sphere_sampling(data.clone())
    #             centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
    #             centres[:, :3] = low_res.pos
    #             centres[:, 3] = i
    #             centres[:, 4] = low_res.y
    #             self._centres_for_sampling.append(centres)
    #             tree = KDTree(np.asarray(data.pos), leaf_size=10)
    #             setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

    #         self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
    #         uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
    #         uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
    #         self._label_counts = uni_counts / np.sum(uni_counts)
    #         self._labels = uni
    #     else:
    #         grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
    #         self._test_spheres = grid_sampler(self._datas)


class SBUBDataset(BaseDataset):
    """ Wrapper around S3DISSphere that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        dataset_cls = SBUBData

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=1000,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=1000,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=1000,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        # if dataset_opt.class_weight_method:
        #     self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.s3dis_tracker import S3DISTracker

        return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


                             
        