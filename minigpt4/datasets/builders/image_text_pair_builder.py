import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder


from minigpt4.datasets.datasets.object_point_dataset import ObjectPointCloudDataset

@registry.register_builder("Objaverse_brief")
class ObjaverseBuilder_brief(BaseDatasetBuilder):
    train_dataset_cls = ObjectPointCloudDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/Objaverse/brief_description.yaml",
    }

    def build_datasets(self):

        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            data_path=build_info.data_path,
            anno_path=build_info.ann_path,
            pointnum=build_info.pointnum,
            conversation_types=build_info.conversation_types,  # * default is simple_des, used for stage1 pre-train
            use_color=build_info.use_color,
            text_processor=self.text_processors["train"],
        )

        return datasets


@registry.register_builder("Objaverse_detailed")
class ObjaverseBuilder_detailed(BaseDatasetBuilder):
    train_dataset_cls = ObjectPointCloudDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/Objaverse/detailed_description.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            data_path=build_info.data_path,
            anno_path=build_info.ann_path,
            pointnum=build_info.pointnum,
            conversation_types=build_info.conversation_types,  # * default is simple_des, used for stage1 pre-train
            use_color=build_info.use_color,
            text_processor=self.text_processors["train"],
        )

        return datasets

@registry.register_builder("Objaverse_single_round")
class ObjaverseBuilder_single_round(BaseDatasetBuilder):
    train_dataset_cls = ObjectPointCloudDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/Objaverse/single_round.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            data_path=build_info.data_path,
            anno_path=build_info.ann_path,
            pointnum=build_info.pointnum,
            conversation_types=build_info.conversation_types,  # * default is simple_des, used for stage1 pre-train
            use_color=build_info.use_color,
            text_processor=self.text_processors["train"],
        )

        return datasets

@registry.register_builder("Objaverse_mutil_round")
class ObjaverseBuilder_mutil_round(BaseDatasetBuilder):
    train_dataset_cls = ObjectPointCloudDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/Objaverse/mutil_round.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            data_path=build_info.data_path,
            anno_path=build_info.ann_path,
            pointnum=build_info.pointnum,
            conversation_types=build_info.conversation_types,  # * default is simple_des, used for stage1 pre-train
            use_color=build_info.use_color,
            text_processor=self.text_processors["train"],
        )

        return datasets
