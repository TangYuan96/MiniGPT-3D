import os
import json
import torch
import numpy as np

import copy
import transformers
from torch.utils.data import Dataset

# from .utils import *

import os
import json
import numpy as np

from minigpt4.datasets.datasets.base_dataset import BaseDataset


class ObjectPointCloudDataset(Dataset):
    """Dataset utilities for objaverse."""

    def __init__(self,
                 text_processor,
                 data_path=None,
                 anno_path=None,
                 pointnum=8192,
                 conversation_types=None,  # * default is simple_des, used for stage1 pre-train
                 use_color=True, ):

        """
        split: only considered when data_args.split_train_val is True.
        conversation_types: tuple, used to filter the data, default is ('simple_description'), other types is:
            "detailed_description", "single_round", "multi_round".
        tokenizer: load point clouds only if None
        """
        # super(ObjectPointCloudDataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.data_path = data_path
        self.anno_path = anno_path
        self.text_processor = text_processor

        if conversation_types is None:
            self.conversation_types = ("simple_description",)
        else:
            self.conversation_types = conversation_types

        self.normalize_pc = True
        self.use_color = use_color

        self.pointnum = pointnum

        self.point_indicator = '<point>'
        self.connect_sym = "!@#"

        # Load the data list from JSON
        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)

        # * print the conversations_type
        print(f"Using conversation_type: {self.conversation_types}")
        # * print before filtering
        print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")

        # * iterate the list and filter
        # * these two ids have corrupted colored point files, so filter them when use_color is True
        filter_ids = ['6760e543e1d645d5aaacd3803bcae524', 'b91c0711149d460a8004f9c06d3b7f38'] if self.use_color else []

        # Iterate the list, filter those "conversation_type" not in self.conversation_types
        self.list_data_dict = [
            data for data in self.list_data_dict
            if data.get('conversation_type', 'simple_description') in self.conversation_types
               and data.get('object_id') not in filter_ids
        ]

        # * print after filtering
        print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")
        # # * print the size of different conversation_type
        # if isinstance(self.conversation_types, str):
        #     data_type = []
        #     data_type.append(self.conversation_types)
        # for conversation_type in data_type:
        #     print(
        #         f"Number of {conversation_type}: {len([data for data in self.list_data_dict if data.get('conversation_type', 'simple_description') == conversation_type])}")

    def _load_point_cloud(self, object_id, type='objaverse'):
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id)

    def _load_objaverse_point_cloud(self, object_id):
        filename = f"{object_id}_{self.pointnum}.npy"
        point_cloud = np.load(os.path.join(self.data_path, filename))

        if not self.use_color:
            point_cloud = point_cloud[:, :3]

        return point_cloud

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc

    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"

        sources = sources[0]
        if self.point_indicator in sources['conversations'][0]['value']:

            object_id = self.list_data_dict[index]['object_id']

            # Point cloud representation
            point_cloud = self._load_point_cloud(object_id)  # * N, C
            if self.normalize_pc:
                point_cloud = self.pc_norm(point_cloud)  # * need to norm since point encoder is norm

            point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
            # point_cloud = point_cloud[0]

        # multi_round的需要分成 多个QA
        if len(sources['conversations']) > 2:

            first_instruction = sources['conversations'][0]['value'].replace('<point>', '').replace('\n', '').strip()
            first_instruction = '<PC><PointCloudHere></PC> {} '.format(first_instruction)

            questions = [first_instruction]
            answers = []

            for i, item in enumerate(sources["conversations"][1:]):
                if i % 2 == 0:  # assistant
                    assistant_answer = item["value"]
                    answers.append(assistant_answer)
                else:
                    human_instruction = item["value"] + " "
                    questions.append(human_instruction)

            questions = self.connect_sym.join(questions)
            answers = self.connect_sym.join(answers)

            return {
                "pc": point_cloud,
                "conv_q": questions,
                'conv_a': answers,
                "PC_id": sources['object_id'],
                "connect_sym": self.connect_sym
            }


        #  biref_description, detailed_description, single_round
        else:
            instruction = sources['conversations'][0]['value']
            instruction = instruction.replace('<point>', '').replace('\n', '').strip()

            instruction = "<PC><PointCloudHere></PC> {} ".format(self.text_processor(instruction))

            answers = sources['conversations'][1]['value']

            # point_cloud = point_cloud[0]
            return {
                "pc": point_cloud,
                "instruction_input": instruction,
                "answer": answers,
                "PC_id": sources['object_id'],
            }

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)

