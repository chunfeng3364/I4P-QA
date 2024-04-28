import math, os
from functools import partial
import sys
sys.path.append("../../..")

import numpy as np
import torch
import torch.nn as nn
from networks.detr_3d.third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes

from networks.detr_3d.detector_Vote2Cap_DETR.helpers import GenericMLP
from networks.detr_3d.detector_Vote2Cap_DETR.vote_query import VoteQuery
from networks.detr_3d.detector_Vote2Cap_DETR.position_embedding import PositionEmbeddingCoordsSine
from networks.detr_3d.detector_Vote2Cap_DETR.transformer import (
    MaskedTransformerEncoder, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer
)


class Model_Vote2Cap_DETR(nn.Module):
    
    def __init__(
        self,
        tokenizer,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        criterion=None
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.encoder = encoder
        
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        
        self.vote_query_generator = VoteQuery(decoder_dim, num_queries)
        
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.criterion = criterion
        


    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=18 + 1)  # dataset_config.num_semcls

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=1)  # dataset_config.num_angle_bin
        angle_reg_head = mlp_func(output_dim=1)  # dataset_config.num_angle_bin

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)


    def _break_up_pc(self, pc):
        # pc may contain color/normals.
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features


    def run_encoder(self, point_clouds, enc_cast: nn.ModuleList = None, frame_feature: torch.Tensor = None, frame_mask: torch.Tensor = None):
        xyz, features = self._break_up_pc(point_clouds)
        
        ## pointcloud tokenization
        # xyz: [batch, npoints,  3]
        # features: [batch, channel, npoints]
        # inds: [batch, npoints]
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.tokenizer(xyz, features)

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)  # [batch, channel, npoints] -> [npoints, batch, channel]

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz, enc_cast=enc_cast, frame_feature=frame_feature, frame_mask=frame_mask
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.long())
        return enc_xyz, enc_features, enc_inds


    def forward(self, inputs: dict, enc_cast: nn.ModuleList = None, dec_cast: nn.ModuleList = None, frame_feature: torch.Tensor = None, frame_mask: torch.Tensor = None):
        
        point_clouds = inputs["point_clouds"]
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        
        ## feature encoding
        # encoder features: npoints x batch x channel -> batch x channel x npoints
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds, enc_cast=enc_cast, frame_feature=frame_feature, frame_mask=frame_mask)
        enc_features = enc_features.permute(1, 2, 0)
        
        ## vote query generation
        query_outputs = self.vote_query_generator(enc_xyz, enc_features)
        query_outputs['seed_inds'] = enc_inds
        query_xyz = query_outputs['query_xyz']
        query_features = query_outputs["query_features"]
        
        
        ## decoding
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        
        # batch x channel x npenc
        enc_features = self.encoder_to_decoder_projection(enc_features)
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_features = enc_features.permute(2, 0, 1)
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = query_features.permute(2, 0, 1)
        
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos, dec_cast=dec_cast, frame_feature=frame_feature, frame_mask=frame_mask
        )[0]    # nlayers x nqueries x batch x channel
        
        dec_features = box_features.permute(0, 2, 1, 3)   # nlayers x batch x nqueries x channel
        enc_features = enc_features.permute(1, 0, 2)      # batch x npoints x channel

        
        return enc_features, dec_features



def build_preencoder(cfg):
    mlp_dims = [cfg.in_channel, 64, 128, cfg.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=cfg.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(cfg):
    if cfg.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.enc_dim,
            nhead=cfg.enc_nhead,
            dim_feedforward=cfg.enc_ffn_dim,
            dropout=cfg.enc_dropout,
            activation=cfg.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=cfg.enc_nlayers
        )
    elif cfg.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.enc_dim,
            nhead=cfg.enc_nhead,
            dim_feedforward=cfg.enc_ffn_dim,
            dropout=cfg.enc_dropout,
            activation=cfg.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=cfg.preenc_npoints // 2,
            mlp=[cfg.enc_dim, 256, 256, cfg.enc_dim],
            normalize_xyz=True,
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {cfg.enc_type}")
    return encoder


def build_decoder(cfg):
    decoder_layer = TransformerDecoderLayer(
        d_model=cfg.dec_dim,
        nhead=cfg.dec_nhead,
        dim_feedforward=cfg.dec_ffn_dim,
        dropout=cfg.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=cfg.dec_nlayers, return_intermediate=True
    )
    return decoder


def detector(cfg ):
    tokenizer = build_preencoder(cfg)  # PointNet++
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    
    # criterion = build_criterion(cfg, dataset_config)  # We don't need criteration
    
    model = Model_Vote2Cap_DETR(
        tokenizer,
        encoder,
        decoder,
        dataset_config=None,
        encoder_dim=cfg.enc_dim,
        decoder_dim=cfg.dec_dim,
        mlp_dropout=cfg.mlp_dropout,
        num_queries=cfg.nqueries,
        criterion=None  # We don't need criteration
    )
    return model
