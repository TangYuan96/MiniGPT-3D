import argparse
import time
from threading import Thread
from PIL import Image
import open3d as o3d
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

import os
import numpy as np
import plotly.graph_objects as go
from minigpt4.common.registry import registry
from minigpt4.models.pointbert import misc

tokenizer = AutoTokenizer.from_pretrained("./params_weight/Phi_2")


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()




@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_pc: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_pc=self.system_pc,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_pc": self.system_pc,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop):] == stop).item():
                return True

        return False


CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <PC>PointCloudHere</PC>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <PC>PointCloudHere</PC>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_minigptv2 = Conversation(
    system="",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)


class Chat:
    def __init__(self, model, device='cuda:0', stopping_criteria=None):
        self.device = device
        self.model = model

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</PC>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer_prepare(self, conv, pc_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                       repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        embs = self.model.get_context_emb(prompt, pc_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
            pad_token_id=tokenizer.eos_token_id,  # when set it with tokenizer.pad_token_id,it do not work
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return generation_kwargs

    def answer(self, conv, pc_list, **kargs):
        generation_dict = self.answer_prepare(conv, pc_list, **kargs)
        output_token = self.model_generate(**generation_dict)[0]
        output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def stream_answer(self, conv, pc_list, **kargs):
        generation_kwargs = self.answer_prepare(conv, pc_list, **kargs)
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

        generated = input_ids
        for _ in range(max_length):
            output = self.forward(input_ids=generated).logits
            next_word_id = output[:, -1, :].argmax(1)
            generated = torch.cat((generated, next_word_id.unsqueeze(-1)), dim=1)

    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output

    def get_fig(self, points):
        colors = points[:, 3:]
        if colors is not None:
            # * if colors in range(0-1)
            if np.max(colors) <= 1:
                color_data = np.multiply(colors, 255).astype(int)  # Convert float values (0-1) to integers (0-255)
            # * if colors in range(0-255)
            elif np.max(colors) <= 255:
                color_data = colors.astype(int)
        else:
            color_data = np.zeros_like(points).astype(int)  # Default to black color if RGB information is not available
        colors = color_data.astype(np.float32) / 255  # model input is (0-1)

        # Convert the RGB color data to a list of RGB strings in the format 'rgb(r, g, b)'
        color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in color_data]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1.2,
                        color=color_strings,  # Use the list of RGB strings for the marker colors
                    )
                )
            ],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)
                ),
                paper_bgcolor='rgb(255,255,255)'  # Set the background color to dark gray 50, 50, 50
            ),
        )

        return fig

    def encoder_pc_file(self, point_cloud_input, pc_list):

        # file = point_cloud_input.name
        # print(f"Uploading file: {file}.")
        #
        # manual_no_color = "no_color" in file
        #
        # try:
        #     if '.ply' in file:
        #         pcd = o3d.io.read_point_cloud(file)
        #         points = np.asarray(pcd.points)  # xyz
        #         colors = np.asarray(pcd.colors)  # rgb, if available
        #         # * if no colors actually, empty array
        #         if colors.size == 0:
        #             colors = None
        #     elif '.npy' in file:
        #         data = np.load(file)
        #         if data.shape[1] >= 3:
        #             points = data[:, :3]
        #         else:
        #             raise ValueError("Input array has the wrong shape. Expected: [N, 3]. Got: {}.".format(data.shape))
        #         colors = None if data.shape[1] < 6 else data[:, 3:6]
        # except:
        #         raise ValueError("Not supported data format.")
        #
        # if manual_no_color:
        #     colors = None
        #
        # if colors is not None:
        #     # * if colors in range(0-1)
        #     if np.max(colors) <= 1:
        #         color_data = np.multiply(colors, 255).astype(int)  # Convert float values (0-1) to integers (0-255)
        #     # * if colors in range(0-255)
        #     elif np.max(colors) <= 255:
        #         color_data = colors.astype(int)
        # else:
        #     color_data = np.zeros_like(points).astype(int)  # Default to black color if RGB information is not available
        # colors = color_data.astype(np.float32) / 255  # model input is (0-1)

        file = point_cloud_input.name
        print(f"Uploading file: {file}.")

        try:
            if '.ply' in file:
                pcd = o3d.io.read_point_cloud(file)
                points = np.asarray(pcd.points)  # xyz
                colors = np.asarray(pcd.colors)  # rgb, if available
                # * if no colors actually, empty array
                if colors.size == 0:
                    colors = None
            elif '.npy' in file:
                data = np.load(file)
                if data.shape[1] == 3 or data.shape[1] == 6:
                    points = data[:, :3]
                else:
                    raise ValueError("Input array has the wrong shape. Expected: [N, 3] xyz or [N, 6] xyzrgb . Got: {}.".format(data.shape))

                if data.shape[1] == 3:
                    colors = None
                elif data.shape[1] == 6:
                    colors = data[:, 3:6]
        except:
            raise ValueError("Not supported data format.")




        if colors is not None:
            # * if colors in range(0-1)
            if np.max(colors) <= 1:
                color_data = np.multiply(colors, 255).astype(int)  # Convert float values (0-1) to integers (0-255)
            # * if colors in range(0-255)
            elif np.max(colors) <= 255:
                color_data = colors.astype(int)
        else:
            color_data = np.zeros_like(points).astype(int)  # Default to black color if RGB information is not available
        colors = color_data.astype(np.float32) / 255  # model input is (0-1)

        point_cloud = np.concatenate((points, colors), axis=1)
        if 8192 < point_cloud.shape[0]:
            indices = np.random.permutation(point_cloud.shape[0])[:2048]
            point_cloud = point_cloud[indices]

        else:
            print("Your point cloud has too few points, possibly leading to bad performance.")


        point_cloud = self.pc_norm(point_cloud)
        pc_fig = self.get_fig(point_cloud)
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32)).unsqueeze(0).to(self.device)
        pc_emb, _ = self.model.encode_pc(point_cloud)

        pc_list.append(pc_emb)

        return pc_fig, pc_list

    def encode_pc_id(self, pc_list):

        ###
        object_id = pc_list[0]
        pc_list.pop(0)
        print("object_id:", object_id)
        point_cloud = self._load_point_cloud(object_id)  # * N, C

        point_cloud = self.pc_norm(point_cloud)  # * need to norm since point encoder is norm

        pc_fig = self.get_fig(point_cloud)

        point_cloud = torch.from_numpy(point_cloud.astype(np.float32)).unsqueeze(0).to(self.device)
        pc_emb, _ = self.model.encode_pc(point_cloud)

        pc_list.append(pc_emb)

        return pc_fig, pc_list

    def upload_pc(self, Object_ID_input, conv, pc_list):
        conv.append_message(conv.roles[0], "<PC><PointCloudHere></PC>")
        pc_list.append(Object_ID_input)
        msg = "Received."

        return msg

    def upload_pc_v2(self, conv):
        conv.append_message(conv.roles[0], "<PC><PointCloudHere></PC>")
        msg = "Received."

        return msg

    def _load_point_cloud(self, object_id, type='objaverse'):
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id)

    def _load_objaverse_point_cloud(self, object_id):

        data_path = './data/objaverse_data'

        filename = f"{object_id}_8192.npy"
        point_cloud = np.load(os.path.join(data_path, filename))

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
