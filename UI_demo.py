import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import open3d as o3d
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, CONV_VISION, \
    StoppingCriteriaSub
import plotly.graph_objects as go

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="./eval_configs/MiniGPT_3D_conv_UI_demo.yaml",
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2,
             'pretrain': CONV_VISION}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished, you can chat with me using the below link!!!!')


# ========================================
#             Gradio Setting
# ========================================



def change_input_method(input_method):
    if input_method == 'File':
        return gr.update(visible=True), gr.update(visible=False)
    elif input_method == 'Object ID':
        return gr.update(visible=False), gr.update(visible=True)


def gradio_reset(chat_state, pc_list):
    if chat_state is not None:
        chat_state.messages = []
    if pc_list is not None:
        pc_list = []
    return None, None, gr.update(value=None, interactive=True), None, gr.update(
        placeholder='Please upload your object ID first',
        interactive=False), gr.update(
        value="Upload object ID  of file & Start Chat", interactive=True), chat_state, pc_list





def upload_pc(Object_ID_input, text_input, chat_state):
    if Object_ID_input is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    pc_list = []
    llm_message = chat.upload_pc(Object_ID_input, chat_state, pc_list)
    pc_fig, pc_list = chat.encode_pc(pc_list)

    return pc_fig, gr.update(interactive=False), gr.update(interactive=True,
                                                           placeholder='Type and press Enter'), gr.update(
        value="Start Chatting", interactive=False), chat_state, pc_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, pc_list, num_beams, temperature, max_new_tokens, max_length, min_length):
    llm_message = chat.answer(conv=chat_state,
                              pc_list=pc_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=max_new_tokens,
                              min_length=min_length,
                              max_length=max_length)[0]

    chatbot[-1][1] = llm_message

    last_item = len(chatbot)
    print("chat round-" + str(last_item) + ": ", chatbot[last_item - 1])

    return chatbot, chat_state, pc_list


def upload_pc_v2(input_choice, Object_ID_input, point_cloud_input, text_input, chat_state):
    if input_choice == 'File':
        chat_state = CONV_VISION.copy()
        pc_list = []
        llm_message = chat.upload_pc_v2(chat_state, )
        pc_fig, pc_list = chat.encoder_pc_file(point_cloud_input, pc_list)

        return pc_fig, gr.update(interactive=False), gr.update(interactive=True,
                                                               placeholder='Type and press Enter'), gr.update(
            value="Start Chatting", interactive=False), chat_state, pc_list

    elif input_choice == 'Object ID':

        if Object_ID_input is None:
            return None, None, gr.update(interactive=True), chat_state, None
        chat_state = CONV_VISION.copy()
        pc_list = []
        llm_message = chat.upload_pc(Object_ID_input, chat_state, pc_list)
        pc_fig, pc_list = chat.encode_pc_id(pc_list)

        return pc_fig, gr.update(interactive=False), gr.update(interactive=True,
                                                               placeholder='Type and press Enter'), gr.update(
            value="Start Chatting", interactive=False), chat_state, pc_list


def start_chat():

    print("[INFO] Starting conversation...")
    while True:
        print("-" * 80)


        title = """<h1 align="center">Demo of MiniGPT-3D</h1>"""

        description_1 = """<h3>MiniGPT-3D takes the first step in efficient 3D-LLM, training with <span style="color: green;">47.8M</span> learnable parameters in just <span style="color: green;">26.8 hours on a single RTX 3090 GPU!</span></h3>"""
        #
        description = """
                    ##### Usage:
                    1. Upload object file (.ply or .npy), or input [Objaverse object id](https://drive.google.com/file/d/1gLwA7aHfy1KCrGeXlhICG9rT2387tWY8/view?usp=sharing) (660K objects, page end show some ids).
                    2. Start chatting.
                     """



        with gr.Blocks() as demo:
            gr.Markdown(title)
            gr.Markdown(description_1)
            gr.Markdown(
                """
                [[Project Page](https://tangyuan96.github.io/minigpt_3d_project_page/)]   [[Paper](https://arxiv.org/pdf/2405.01413)]   [[Code](https://github.com/TangYuan96/MiniGPT-3D)]
                """
            )
            gr.Markdown(description)

            with gr.Row():
                with gr.Column():
                    input_choice = gr.Radio(['File', 'Object ID'], value='Object ID', interactive=True,
                                            label='Input Method',
                                            info="How do you want to load point clouds?")

                    point_cloud_input = gr.File(file_types=[".ply", ".npy"], visible=False,
                                                label="Upload Point Cloud File (.ply or .npy), format: [N,xyz] or [N,xyzrgb]")

                    Object_ID_input = gr.Textbox(label="Object ID", placeholder='Please input the Object ID ',
                                                 interactive=True)

                    with gr.Accordion("More settings", open=True):
                        with gr.Row():
                            num_beams = gr.Slider(
                                minimum=1, maximum=10, value=1, step=1, interactive=True, label="beam number", )
                            temperature = gr.Slider(
                                minimum=0.1, maximum=2.0, value=0.2, step=0.1, interactive=True, label="Temperature", )

                        with gr.Row():
                            max_new_tokens = gr.Slider(
                                minimum=10, maximum=200, value=60, step=10, interactive=True, label="Max words per reply", )
                            max_length = gr.Slider(
                                minimum=400, maximum=1500, value=400, step=100, interactive=True,
                                label="Max words in conv.", )

                        min_length = gr.Slider(
                            minimum=1, maximum=200, value=1, step=5, interactive=True, label="Min words per reply", )

                    with gr.Row():
                        upload_button = gr.Button(value="Start Chat", interactive=True, variant="primary")
                        clear = gr.Button("Restart")

                output = gr.Plot()

                with gr.Column():
                    chat_state = gr.State()
                    pc_list = gr.State()
                    chatbot = gr.Chatbot(label='MiniGPT-3D', height=500)
                    text_input = gr.Textbox(label='User', placeholder='Please upload your object ID', interactive=False)

            upload_button.click(upload_pc_v2, [input_choice, Object_ID_input, point_cloud_input, text_input, chat_state],
                                [output, Object_ID_input, text_input, upload_button, chat_state, pc_list])

            text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
                gradio_answer,
                [chatbot, chat_state, pc_list, num_beams, temperature, max_new_tokens, max_length, min_length],
                [chatbot, chat_state, pc_list])
            clear.click(gradio_reset, [chat_state, pc_list],
                        [output, chatbot, Object_ID_input, point_cloud_input, text_input, upload_button, chat_state, pc_list], queue=False)



            with gr.Accordion("Example Objaverse object ids in the validation set!", open=False):
                example_object_ids = [
                    ["b4bbf2116b1a41a5a3b9d3622b07074c", "0b8da82a3d7a436f9b585436c4b72f56",
                     "650c53d68d374c18886aab91bcf8bb54"],
                    ["983fa8b23a084f5dacd157e6c9ceba97", "8fe23dd4bf8542b49c3a574b33e377c3",
                     "83cb2a9e9afb47cd9f45461613796645"],
                    ["3d679a3888c548afb8cf889915af7fd2", "7bcf8626eaca40e592ffd0aed08aa30b",
                     "69865c89fc7344be8ed5c1a54dbddc20"],
                    ["252f3b3f5cd64698826fc1ab42614677", "e85ebb729b02402bbe3b917e1196f8d3",
                     "97367c4740f64935b7a5e34ae1398035"],
                    ["e833d5ed1c654ff0960ffea76c741c2a", "8257772b0e2f408ba269264855dfea00",
                     "d6a3520486bb474f9b5e72eda8408974"],
                    ["3d10918e6a9a4ad395a7280c022ad2b9", "1f5246ceecb14f22bc547911e5e430ff",
                     "76ba80230d454de996878c2763fe7e5c"]]
                gr.DataFrame(
                    type="array",
                    headers=["Example Object IDs"] * 3,
                    row_count=6,
                    col_count=3,
                    value=example_object_ids
                )
            gr.Markdown(
                """
                #### Terms of use
                By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
                """
            )
            gr.Markdown(
                """
                #### Acknowledgements
                 [[PointLLM](https://github.com/OpenRobotLab/PointLLM/tree/master)] [[TinyGPT-V](https://github.com/DLYuanGod/TinyGPT-V)] [[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)]
                """
            )

            input_choice.change(change_input_method, input_choice, [point_cloud_input, Object_ID_input])
        demo.launch(share=False)
        demo.queue()

if __name__ == "__main__":
    start_chat()