import torch
import torch.nn as nn
from transformers import (
    InstructBlipQFormerModel,
    InstructBlipQFormerConfig
)

import sys
sys.path.append("../")
from networks.detr_3d.detector_Vote2Cap_DETR.config import model_config
from networks.detr_3d.detector_Vote2Cap_DETR.detector import detector
from networks.cast import Uni_CAST, Adapter
from networks.LanguageEncoder import LanguageEncoder


class QAModel(nn.Module):
    def __init__(self, 
                 num_semcls: int = 18, frame_feature_dim: int = 768, cast_choice: str = "decoder", freeze_detector: bool = True, 
                 adapter_mlp_ratio: float = 0.25, 
                 num_query_qformer: int = 32,
                 llm_name="microsoft/deberta-v3-large", lora_r=8, lora_alpha=16,
                 dropout: float = 0.1, ans_num: int = 1277) -> None:
        super().__init__()
        
        # Detector
        self.detector_cfg = model_config(num_semcls=num_semcls)
        self.detector = detector(cfg=self.detector_cfg)
        if freeze_detector is True:
            for name, param in self.detector.named_parameters():
                param.requires_grad = False
        
        self.detector_enc_dim = self.detector_cfg.enc_dim
        self.detector_dec_dim = self.detector_cfg.dec_dim
        assert self.detector_enc_dim == self.detector_dec_dim
        self.detector_enc_nlayer = self.detector_cfg.enc_nlayers
        self.detector_dec_nlayer = self.detector_cfg.dec_nlayers
        
        # CAST
        assert cast_choice in ["encoder", "decoder", "encoder_decoder"], "Don't you use frame feature?"
        self.cast_choice = cast_choice
        cast_module = nn.ModuleDict({
            "pre_adapter": Adapter(dim=self.detector_enc_dim, mlp_ratio=adapter_mlp_ratio),
            "cast": Uni_CAST(updim=frame_feature_dim, downdim=self.detector_enc_dim),
            "post_adapter": Adapter(dim=self.detector_enc_dim, mlp_ratio=adapter_mlp_ratio),
        })
        self.encoder_cast = None
        self.decoder_cast = None
        if "encoder" in self.cast_choice:
            self.encoder_cast = nn.ModuleList([cast_module for _ in range(self.detector_enc_nlayer)])
        if "decoder" in self.cast_choice:
            self.decoder_cast = nn.ModuleList([cast_module for _ in range(self.detector_dec_nlayer)])
        
        # Multi-modality Transformer
        self.num_query_qformer = num_query_qformer
        
        qformer_config = InstructBlipQFormerConfig(
            num_hidden_layers=6,
            encoder_hidden_size=self.detector_dec_dim
            )
        self.qformer = InstructBlipQFormerModel.from_pretrained(
            "CH3COOK/bert-base-embedding", 
            config=qformer_config
            )
        self.qformer_dim = qformer_config.hidden_size  # dim in qformer
        self.latent_query = nn.Embedding(self.num_query_qformer, self.qformer_dim)
        
        # Language Model
        self.language_model = LanguageEncoder(llm_name=llm_name, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=dropout)
        
        # Projection
        self.encoder_to_qformer_projection = nn.Sequential(
            nn.Linear(self.detector_dec_dim, qformer_config.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_config.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
        )  # scene feature -> q-former
        self.llm_dim = self.language_model.model.config.hidden_size  # dim in Language Models
        self.qformer_to_language_projection = nn.Linear(self.qformer_dim, self.llm_dim)  # q-former -> language model

        self.classifier = nn.Linear(in_features=self.llm_dim, out_features=ans_num)
        
    
    def forward(self, point_inputs: dict, frame_feature: torch.Tensor, frame_mask: torch.Tensor, qformer_inputs: dict, question_prompt: tuple) -> torch.Tensor:
        BS = frame_feature.size(0)
        device = frame_feature.device
        
        enc_features, dec_features = self.detector(
            inputs=point_inputs, enc_cast=self.encoder_cast, dec_cast=self.decoder_cast, 
            frame_feature=frame_feature, frame_mask=frame_mask
            )
        # enc_features: [batch, npoints, d_pc]
        # dec_features: [dec_nlayers, batch, object_nproposal, d_pc]
        
        dec_features = dec_features[-1]  # last layer
        
        # 相较于 ./models/ll3da/captioner.py 我们没有prompt feature
        query_tokens = self.latent_query.weight.unsqueeze(0).repeat(BS, 1, 1)
        query_attention_mask = torch.ones(BS, self.num_query_qformer).to(device)
        
        query_attention_mask = torch.cat((query_attention_mask, qformer_inputs['qformer_attention_mask']), dim=1)
        
        # q-former
        query_outputs = self.qformer(
            input_ids=qformer_inputs['qformer_input_ids'],
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=self.encoder_to_qformer_projection(dec_features),
        )
        query_outputs = query_outputs[0][:, : self.num_query_qformer, :]
        qformer_feature = self.qformer_to_language_projection(query_outputs)
        
        output = self.language_model(visual_feature=qformer_feature, visual_mask=None, qas_word=question_prompt)
        
        output = self.classifier(output)
        
        return output


if __name__ == "__main__":
    r"""
    测试模型输入输出
    """
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    
    point_cloud = np.random.random(size=(2, 5000, 10)).astype(np.float32)  # [xyz, rgb, normal, height]
    point_cloud_dims_min = point_cloud[..., :3].min(axis=1).astype(np.float32)
    point_cloud_dims_max = point_cloud[..., :3].max(axis=1).astype(np.float32)
    print(point_cloud.shape)
    print(point_cloud_dims_min.shape)
    print(point_cloud_dims_max.shape)
    
    frame_feature = torch.rand(2, 10, 768).to("cuda")
    frame_mask = torch.Tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    ]).bool().to("cuda")
    
    # 直接用torch.rand会导致tensor没有shape属性
    model = QAModel().to("cuda")
    input = {
        'point_clouds': torch.from_numpy(point_cloud).to("cuda"),
        'point_cloud_dims_min': torch.from_numpy(point_cloud_dims_min).to("cuda"),
        'point_cloud_dims_max': torch.from_numpy(point_cloud_dims_max).to("cuda"),
    }
    
    # Tokenizer
    llm_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(llm_name, add_bos_token=False)
    qtokenizer = AutoTokenizer.from_pretrained("CH3COOK/bert-base-embedding")
    qtokenizer.pad_token = tokenizer.eos_token
    qtokenizer.padding_side = 'right'
    tokenizer_config = dict(
        max_length=512, 
        padding='max_length', 
        truncation='longest_first',
        return_tensors='pt' 
    )
    
    question = ("Hello I'm a", "I like")
    question_prompt = ("Hello I'm a [MASK]", "I like [MASK]")
    
    qformer_inputs = qtokenizer.batch_encode_plus(question, **tokenizer_config)
    qformer_inputs = {
        'qformer_input_ids': qformer_inputs['input_ids'].int().to("cuda"), 
        'qformer_attention_mask': qformer_inputs['attention_mask'].float().to("cuda")
        }
    
    output = model(point_inputs=input, frame_feature=frame_feature, frame_mask=frame_mask, qformer_inputs=qformer_inputs, question_prompt=question_prompt)
    print(output.size())
    
    
    r"""
    测试weights能否正确load
    """
    # pretrain_path = "/storage_fast/cfeng/scanqa/3d_backbone/Vote2Cap-DETR/pretrained/Vote2Cap_DETR_XYZ_COLOR_NORMAL/checkpoint_best.pth"
    # weight_dict = torch.load(pretrain_path)
    # # print(weight_dict["model"].keys())
    # model = QAModel()
    # model.load_state_dict(weight_dict["model"])
    
    # for param_name in weight_dict["model"].keys():
    #     if param_name not in model.state_dict():
    #         print(param_name)
    
    # print("--------------------------")
    
    # for param_name in model.state_dict():
    #     if param_name not in weight_dict["model"].keys():
    #         print(param_name)