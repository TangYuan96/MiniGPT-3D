
import torch
import torch.nn as nn
import torch.nn.functional as F
from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
import logging
from peft import LoraConfig, inject_adapter_in_model

logger = logging.getLogger(__name__)

@registry.register_model("minigpt_3d")
class MiniGPT_3D(MiniGPTBase):
    """
    MiniGPT_3D model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_3d.yaml",
    }

    def __init__(
            self,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            pc_precision="fp16",
            freeze_pc=True,
            llama_model="",
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=['query_key_value', 'dense'],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put pc in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            use_MQE=False,
            query_expert_num=8,  # work ony when use_MQE   is True
            query_expert_router_type='Sparse Router',  # Three types： Constant Router, Sparse Router,  Soft Router
            query_expert_top_k=2,  # the top_k number in 'Sparse Router'
            QFormer_lora_r=-1,
            train_QFormer_norm=False,
            freeze_Qformer=True,
            only_train_MQE=False,
            QFormer_lora_module=["query", "key", "value"],
            pc_linear_layer=2,
            only_train_pc_linear=False,
    ):
        super().__init__(
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        print('Init pc encoder: pc_encoder')
        self.pc_encoder = self.init_pc_encoder(pc_precision, freeze_pc)

        print('Create the MLP: point_2_Qformer_proj')
        if pc_linear_layer == 1:
            projection_layers = []
            projection_layers.append(nn.Linear(self.pc_encoder.trans_dim, 1408))
            self.point_2_Qformer_proj = nn.Sequential(*projection_layers)
        elif pc_linear_layer == 2:
            self.point_2_Qformer_proj = nn.Sequential(
                nn.Linear(self.pc_encoder.trans_dim, 768),
                nn.GELU(),
                nn.Linear(768, 1408),
            )
        elif pc_linear_layer == 3:
            self.point_2_Qformer_proj = nn.Sequential(
                nn.Linear(self.pc_encoder.trans_dim, 768),
                nn.GELU(),
                nn.Linear(768, 1152),
                nn.GELU(),
                nn.Linear(1152, 1408),
            )

        print('Load Q-Former')
        self.freeze_Qformer = freeze_Qformer
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token=32, vision_width=1408, freeze=self.freeze_Qformer,
            QFormer_lora_r=QFormer_lora_r,
            train_QFormer_norm=train_QFormer_norm,
            QFormer_lora_module=QFormer_lora_module,
        )
        self.load_from_pretrained(
            url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")  # load q-former weights here
        print('Load Q-Former done')

        print('Create the Projector: llama_proj and llama_proj2')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, 4096
        )
        self.llama_proj2 = nn.Linear(
            4096, self.llama_model.config.hidden_size
        )

        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()

        ######### mixture of query expert  ##############
        self.use_MQE = use_MQE
        if self.use_MQE:

            self.query_expert_num = query_expert_num
            self.query_expert_top_k = query_expert_top_k
            self.router_type = query_expert_router_type  # Constant Router, Sparse Router,  Soft Router

            self.query_expert_dict = nn.ParameterDict()
            self.query_tokens.requires_grad = True
            self.query_expert_dict[str(0)] = self.query_tokens

            for i in range(self.query_expert_num - 1):
                copied_param = nn.Parameter(self.query_tokens.data.clone()).to(self.query_tokens.device)
                copied_param.requires_grad = True
                self.query_expert_dict[str(i + 1)] = copied_param
            print('Create query expert ok, query_expert_num: ', self.query_expert_num)

            self.query_router = nn.Sequential(
                nn.Linear(self.pc_encoder.trans_dim * 2, 256),
                nn.GELU(),
                nn.Linear(256, self.query_expert_num),
            )
            print('Create query_router ok, router type: ', self.router_type)
        ######### mixture of query expert   ##############


        # fix the pc encoder in all training stages
        self.freeze_pc = freeze_pc
        if self.freeze_pc:
            for name, param in self.pc_encoder.named_parameters():
                param.requires_grad = False
            self.point_encoder = self.pc_encoder.eval()
            self.point_encoder.train = disabled_train

            print("Freeze pc encoder")


        ############### only Stage I ###
        self.only_train_pc_linear = only_train_pc_linear
        if self.only_train_pc_linear:

            # PC Encoder in Stage I of paper figure 3
            for name, param in self.pc_encoder.named_parameters():
                param.requires_grad = False

            # MLP in Stage I of paper figure 3
            for name, param in self.point_2_Qformer_proj.named_parameters():
                param.requires_grad = True

            # Qformer in Stage I of paper figure 3
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False

            self.query_tokens.requires_grad = False

            # Projector in Stage I of paper figure 3
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.llama_proj2.named_parameters():
                param.requires_grad = False

            # LLM (Phi-2) with LoRA  in Stage I of paper figure 3
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            print("Stage I: only train the MLP: point_2_Qformer_proj")
        ############### only Stage I ###

        ############### only Stage IV ###
        self.only_train_MQE = only_train_MQE
        if self.only_train_MQE:

            # PC Encoder in Stage IV of paper figure 3
            for name, param in self.pc_encoder.named_parameters():
                param.requires_grad = False

            # MLP in Stage IV of paper figure 3
            for name, param in self.point_2_Qformer_proj.named_parameters():
                param.requires_grad = False

            # Mixture of Query Experts in Stage IV of paper figure 3
            for query_expert in self.query_expert_dict:
                self.query_expert_dict[str(query_expert)].requires_grad = True
            self.query_router.requires_grad = True

            # Qformer  with LoRA in Stage IV of paper figure 3
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False

            # Projector in Stage IV of paper figure 3
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.llama_proj2.named_parameters():
                param.requires_grad = False

            # LLM (Phi-2) with LoRA in Stage IV of paper figure 3
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            print("Stage  IV: only train Mixture of Query Experts")
        ############### only Stage IV ###

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze, QFormer_lora_r,
                     train_QFormer_norm, QFormer_lora_module):

        encoder_config = BertConfig.from_pretrained("./params_weight/bert-base-uncased")
        # encoder_config = BertConfig.from_pretrained("bert-base-uncased")

        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # Q-former不可训练
        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer.bert = Qformer.bert.eval()
            Qformer.train = disabled_train

            print("    Freeze Q-former backbone")


        if QFormer_lora_r > 0:
            lora_config = LoraConfig(
                lora_alpha=QFormer_lora_r * 2,
                lora_dropout=0.1,
                r=QFormer_lora_r,
                bias="none",
                target_modules=QFormer_lora_module,
                # ["query", "key",  "value",   "dense"]  ["query", "key",  "value"]
            )

            print("    QFormer_lora_module:", QFormer_lora_module)
            Qformer = inject_adapter_in_model(lora_config, Qformer)

            print("    Use lora on Q-former")


            if train_QFormer_norm:

                for i, layer in enumerate(Qformer.bert.encoder.layer):
                    layer.attention.output.LayerNorm.weight.requires_grad = True
                    layer.output_query.LayerNorm.weight.requires_grad = True

                    layer.attention.output.LayerNorm.weight.data = layer.attention.output.LayerNorm.weight.data.float()
                    layer.output_query.LayerNorm.weight.data = layer.output_query.LayerNorm.weight.data.float()

                    layer.attention.output.LayerNorm.bias.data = layer.attention.output.LayerNorm.bias.data.float()
                    layer.output_query.LayerNorm.bias.data = layer.output_query.LayerNorm.bias.data.float()

                    # Even layer has cross_attention
                    if i % 2 == 0:
                        layer.crossattention.output.LayerNorm.weight.requires_grad = True

                        layer.crossattention.output.LayerNorm.weight.data = layer.crossattention.output.LayerNorm.weight.data.float()
                        layer.crossattention.output.LayerNorm.bias.data = layer.crossattention.output.LayerNorm.bias.data.float()

                print("    Train Q-Former layerNorm")

        else:
            print("    Not use lora on Q-former")

        return Qformer, query_tokens

    def encode_pc(self, pc):
        device = pc.device

        # Stage IV
        if self.use_MQE:

            if self.router_type == 'Sparse Router':
                with self.maybe_autocast():

                    pc_feature = self.pc_encoder(pc)

                    # query expert router
                    pc_global_feature = torch.cat([pc_feature[:, 0], pc_feature[:, 1:].max(1)[0]], dim=-1)
                    router_weights = self.query_router(pc_global_feature)
                    router_weights = F.softmax(router_weights, dim=-1)
                    # query expert router

                    # get top_k weight
                    _, indices = torch.topk(router_weights, k=self.query_expert_top_k, dim=1, largest=True)
                    sorted_indices, _ = torch.sort(indices, dim=1)
                    router_weights_top_k = router_weights.gather(1, sorted_indices)
                    # get top_k weight

                    # MLP
                    pc_embeds = self.point_2_Qformer_proj(pc_feature).to(device)
                    # MLP

                    # pc tokens
                    pc_embeds_repeat = pc_embeds.repeat(self.query_expert_top_k, 1, 1, 1).transpose(0, 1)
                    pc_embeds_repeat = pc_embeds_repeat.reshape(-1, pc_embeds_repeat.size()[2],
                                                                pc_embeds_repeat.size()[3])
                    # pc tokens

                    # some values to Q-former
                    pc_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(device)
                    pc_atts_repeat = pc_atts.repeat(self.query_expert_top_k, 1, 1)
                    pc_atts_repeat = pc_atts_repeat.reshape(-1, pc_atts_repeat.size()[2])
                    # some values to Q-former

                    # top-k query experts
                    query_tokens_list = []
                    for query_expert in self.query_expert_dict:
                        query_tokens = self.query_expert_dict[str(query_expert)]
                        query_tokens_list.append(query_tokens)
                    query_tokens_repeat = torch.stack(query_tokens_list, dim=0).squeeze(dim=1)
                    query_tokens_repeat = query_tokens_repeat.expand(pc_embeds.shape[0], -1, -1, -1)

                    query_tokens_repeat_list = []
                    for item in range(sorted_indices.size()[0]):
                        query_tokens_repeat_list.append(
                            query_tokens_repeat[item][sorted_indices[item], :, :].unsqueeze(0))

                    query_tokens_repeat = torch.stack(query_tokens_repeat_list, dim=0).squeeze(dim=1)

                    query_tokens_repeat = query_tokens_repeat.reshape(-1, query_tokens_repeat.size()[2],
                                                                      query_tokens_repeat.size()[3], )
                    # top-k query experts

                    # Q-former
                    query_output_repeat = self.Qformer.bert(
                        query_embeds=query_tokens_repeat,
                        encoder_hidden_states=pc_embeds_repeat,
                        encoder_attention_mask=pc_atts_repeat,
                        return_dict=True,
                    )
                    B_Top_k, n, D = query_tokens_repeat.size()
                    B, top_k = router_weights_top_k.size()
                    query_outputs = query_output_repeat.last_hidden_state.reshape(B, top_k, n, D)
                    # Q-former

                    # output of MQE
                    query_output = torch.einsum('ijkm,ij->ikm', [query_outputs, router_weights_top_k])
                    # output of MQE

                    # Projector
                    inputs_llama = self.llama_proj(query_output)
                    inputs_llama = self.llama_proj2(inputs_llama)
                    # Projector

                    atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(pc.device)
                return inputs_llama, atts_llama

            elif self.router_type == 'Soft Router' or self.router_type == 'Constant Router':
                with self.maybe_autocast():

                    pc_feature = self.pc_encoder(pc)

                    # query expert router
                    pc_global_feature = torch.cat([pc_feature[:, 0], pc_feature[:, 1:].max(1)[0]], dim=-1)
                    router_weights = self.query_router(pc_global_feature)
                    router_weights = F.softmax(router_weights, dim=-1)

                    # MLP
                    pc_embeds = self.point_2_Qformer_proj(pc_feature).to(device)

                    pc_embeds_repeat = pc_embeds.repeat(self.query_expert_num, 1, 1, 1)
                    pc_embeds_repeat = pc_embeds_repeat.reshape(-1, pc_embeds_repeat.size()[2],
                                                                pc_embeds_repeat.size()[3])

                    pc_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(device)
                    pc_atts_repeat = pc_atts.repeat(self.query_expert_num, 1, 1)
                    pc_atts_repeat = pc_atts_repeat.reshape(-1, pc_atts_repeat.size()[2])

                    query_tokens_list = []
                    for query_expert in self.query_expert_dict:
                        query_tokens = self.query_expert_dict[str(query_expert)].expand(pc_embeds.shape[0], -1, -1)
                        query_tokens_list.append(query_tokens)
                    query_tokens_repeat = torch.stack(query_tokens_list, dim=0)
                    query_tokens_repeat = query_tokens_repeat.reshape(-1, query_tokens_repeat.size()[2],
                                                                      query_tokens_repeat.size()[3], )

                    query_output_repeat = self.Qformer.bert(
                        query_embeds=query_tokens_repeat,
                        encoder_hidden_states=pc_embeds_repeat,
                        encoder_attention_mask=pc_atts_repeat,
                        return_dict=True,
                    )
                    _, Q, D = query_tokens_repeat.size()
                    query_outputs = query_output_repeat.last_hidden_state.reshape(self.query_expert_num, -1, Q,
                                                                                  D).transpose(0, 1)

                    # query expert router
                    if self.router_type == 'Constant Router':
                        router_weights = torch.ones_like(router_weights).to(device) * (1 / self.query_expert_num)
                    elif self.router_type == 'Soft Router':
                        pass
                    # elif self.router_type == 'Sparse Router':
                    #     _, indices = torch.topk(router_weights, k=self.query_expert_top_k, dim=1, largest=True)
                    #     indices = indices.unsqueeze(-1).expand(-1, -1, 2).contiguous().view(-1, 4)
                    #     output_top = torch.zeros_like(router_weights).to(device)
                    #     output_top.scatter_(1, indices, 1)
                    #     router_weights = torch.einsum('ij,ij->ij', [output_top, router_weights])

                    query_output = torch.einsum('ijkm,ij->ikm', [query_outputs, router_weights])

                    inputs_llama = self.llama_proj(query_output)
                    inputs_llama = self.llama_proj2(inputs_llama)
                    atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(pc.device)
                return inputs_llama, atts_llama

        else:
            # Stage I, II, III
            with self.maybe_autocast():

                # pc tokens
                pc_embeds = self.point_2_Qformer_proj(self.pc_encoder(pc)).to(device)
                # pc tokens

                # some values to Q-former
                pc_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(device)
                # some values to Q-former

                # query_tokens
                query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
                # query_tokens

                # Q-former
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=pc_embeds,
                    encoder_attention_mask=pc_atts,
                    return_dict=True,
                )
                # Q-former

                # Projector
                inputs_llama = self.llama_proj(query_output.last_hidden_state)
                inputs_llama = self.llama_proj2(inputs_llama)
                # Projector

                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(pc.device)

            return inputs_llama, atts_llama


    @classmethod
    def from_config(cls, cfg):
        pc_model = cfg.get("pc_model", "pointbert")

        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        pc_precision = cfg.get("pc_precision", "fp16")
        freeze_pc = cfg.get("freeze_pc", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        QFormer_lora_r = cfg.get("QFormer_lora_r", -1)

        train_QFormer_norm = cfg.get("train_QFormer_norm", False)

        query_expert_router_type = cfg.get("query_expert_router_type", 'Sparse Router')
        query_expert_num = cfg.get("query_expert_num", 8)
        query_expert_top_k = cfg.get("query_expert_top_k", 2)

        use_MQE = cfg.get("use_MQE", False)
        only_train_MQE = cfg.get("only_train_MQE", False)

        QFormer_lora_module = cfg.get("QFormer_lora_module", ["query", "key", "value"])

        pc_linear_layer = cfg.get("pc_linear_layer", 2)

        only_train_pc_linear = cfg.get("only_train_pc_linear", False)

        freeze_Qformer = cfg.get("freeze_Qformer", True)

        model = cls(
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            pc_precision=pc_precision,
            freeze_pc=freeze_pc,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
            QFormer_lora_r=QFormer_lora_r,
            train_QFormer_norm=train_QFormer_norm,
            query_expert_router_type=query_expert_router_type,
            query_expert_num=query_expert_num,
            query_expert_top_k=query_expert_top_k,
            use_MQE=use_MQE,
            only_train_MQE=only_train_MQE,
            QFormer_lora_module=QFormer_lora_module,
            pc_linear_layer=pc_linear_layer,
            only_train_pc_linear=only_train_pc_linear,
            freeze_Qformer=freeze_Qformer,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load MiniGPT-3D first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        else:
            print("No load first ckpt!!!")

        stage_3_ckpt = cfg.get("second_ckpt", "")
        if stage_3_ckpt:
            print("Load MiniGPT-3D second_ckpt Checkpoint: {}".format(stage_3_ckpt))
            ckpt = torch.load(stage_3_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        else:
            print("No load second_ckpt!!!")

        return model
