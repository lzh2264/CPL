"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: https://github.com/google-research/deeplab2/blob/main/model/transformer_decoder/kmax.py
"""

from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from timm.models.layers import DropPath
# from timm.models.layers import trunc_normal_tf_ as trunc_normal_

# I changed to
from timm.models.layers import trunc_normal_

from detectron2.config import configurable
from detectron2.utils.registry import Registry

from .kmax_pixel_decoder import get_norm, ConvBN

import math

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module.
"""


def build_transformer_decoder(cfg, input_shape_from_backbone):
    """
    Build a instance embedding branch from `cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NAME`.
    """
    name = cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, input_shape_from_backbone)


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/decoder/max_deeplab.py#L60
def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
        (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L41
class AttentionOperation(nn.Module):
    def __init__(self, channels_v, num_heads):
        super().__init__()
        # self._batch_norm_similarity = get_norm('syncbn', num_heads)
        # self._batch_norm_retrieved_value = get_norm('syncbn', channels_v)
        self._batch_norm_similarity = get_norm('BN', num_heads)
        self._batch_norm_retrieved_value = get_norm('1b', channels_v)

    def forward(self, query, key, value):
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('bhdl,bhdm->bhlm', query, key)
        similarity_logits = self._batch_norm_similarity(similarity_logits)

        with autocast(enabled=False):
            attention_weights = F.softmax(similarity_logits.float(), dim=-1)
        retrieved_value = torch.einsum(
            'bhlm,bhdm->bhdl', attention_weights, value)
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = self._batch_norm_retrieved_value(
            retrieved_value)
        retrieved_value = F.gelu(retrieved_value)
        return retrieved_value


# https://github.com/google-research/deeplab2/blob/main/model/kmax_deeplab.py#L32
class kMaXPredictor(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133 + 1):
        super().__init__()
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5,
                                                   groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='bn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm='bn',
                                                   act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm='bn', act=None)
        trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN(256, 128, kernel_size=1, bias=False, norm='1b', act=None, conv_type='1d')
        self._transformer_class_head = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm('3b', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        # mask_embeddings/class_embeddings: B x C x N
        # pixel feature: B x C x H x W
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)  # 256->128
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        mask_logits = torch.einsum('bchw,bcn->bnhw', pixel_space_normalized_feature, cluster_mask_kernel)#[B, 32, 28, 28] 聚类中心 * 像素特征
        # 下一步  Query response: 每个像素与聚类中心之间的相似度; query-wise softmax （C*P）
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)
        # mask_logits = self._pixel_space_mask_batch_norm(mask_logits)

        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature
        }


class kMaXPredictor_final(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133 + 1):
        super().__init__()
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5, groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='bn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm='bn', act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm='bn', act=None)
        trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN(in_channel_pixel, 128, kernel_size=1, bias=False, norm='1b', act=None, conv_type='1d')
        self._transformer_class_head = ConvBN(in_channel_query, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm('3b', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        # mask_embeddings/class_embeddings: B x C x N  [B 16 256]
        # pixel feature: B x C x H x W   [B 16 224 224]
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)  # 256->128
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()#B 32 4; class_embeddings B 16 32
        cluster_class_logits = add_bias_towards_void(cluster_class_logits) #B 32 4
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        mask_logits = torch.einsum('bchw,bcn->bnhw',
                                   pixel_space_normalized_feature, cluster_mask_kernel)
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)
        # mask_logits = self._pixel_space_mask_batch_norm(mask_logits)

        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature,
            'cluster_mask_kernel': cluster_mask_kernel
        }


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L107
class kMaXTransformerLayer(nn.Module):
    def __init__(
            self,
            num_classes=133,
            in_channel_pixel=2048,
            in_channel_query=16,
            base_filters=128,
            num_heads=8,
            bottleneck_expansion=2,
            key_expansion=1,
            value_expansion=2,
            drop_path_prob=0.0,
    ):
        super().__init__()

        self._num_classes = num_classes
        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        # Per tf2 implementation, the same drop path prob are applied to:
        # 1. k-means update for object query
        # 2. self/cross-attetion for object query
        # 3. ffn for object query
        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        initialization_std = self._bottleneck_channels ** -0.5
        self._query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='1b', act='gelu', conv_type='1d')

        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='bn', act='gelu')

        self._query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                         norm='1b', act=None, conv_type='1d')
        trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                       norm='bn', act=None)
        trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                      norm='1b', act=None, conv_type='1d', norm_init=0.0)

        self._query_ffn_conv1_bn_act = ConvBN(in_channel_query, 2048, kernel_size=1, bias=False,
                                              norm='1b', act='gelu', conv_type='1d')
        self._query_ffn_conv2_bn = ConvBN(2048, in_channel_query, kernel_size=1, bias=False,
                                          norm='1b', act=None, conv_type='1d', norm_init=0.0)

        self._predcitor = kMaXPredictor(in_channel_pixel=self._bottleneck_channels,
                                        in_channel_query=self._bottleneck_channels, num_classes=num_classes)
        self._kmeans_query_batch_norm_retrieved_value = get_norm('1b', self._total_value_depth)
        self._kmeans_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                             norm='1b', act=None, conv_type='1d', norm_init=0.0)

    def forward(self, pixel_feature, query_feature):
        N, C, H, W = pixel_feature.shape
        _, D, L = query_feature.shape
        pixel_space = self._pixel_conv1_bn_act(F.gelu(pixel_feature))  # N C H W
        query_space = self._query_conv1_bn_act(query_feature)  # N x C x L

        # k-means cross-attention.
        pixel_value = self._pixel_v_conv_bn(pixel_space)  # N C H W
        pixel_value = pixel_value.reshape(N, self._total_value_depth, H * W)
        # k-means assignment.
        prediction_result = self._predcitor(
            mask_embeddings=query_space, class_embeddings=query_space, pixel_feature=pixel_space)

        with torch.no_grad():
            clustering_result = prediction_result['mask_logits'].flatten(2).detach()  # N L HW
            index = clustering_result.max(1, keepdim=True)[1]
            clustering_result = torch.zeros_like(clustering_result,
                                                 memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            # print(clustering_result[0,0,:])
            # print(clustering_result.max())
        with autocast(enabled=False):
            # k-means update.
            kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float())  # N x C x L

        kmeans_update = self._kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_query_conv3_bn(kmeans_update)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        # query self-attention.
        query_qkv = self._query_qkv_conv_bn(query_space)
        query_q, query_k, query_v = torch.split(query_qkv,
                                                [self._total_key_depth, self._total_key_depth, self._total_value_depth],
                                                dim=1)
        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        query_k = query_k.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        query_v = query_v.reshape(N, self._num_heads, self._total_value_depth // self._num_heads, L)
        self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        self_attn_update = self._query_conv3_bn(self_attn_update)
        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)# query response
        # FFN.
        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)#cluster_centers

        return query_feature, prediction_result


class ASPP(nn.Module):
    def __init__(self, in_channels, output_channels, atrous_rates):
        super().__init__()

        self._aspp_conv0 = ConvBN(in_channels, output_channels, kernel_size=1, bias=False,
                                  norm='bn', act='gelu')

        rate1, rate2, rate3 = atrous_rates
        self._aspp_conv1 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate1, padding=rate1,
                                  bias=False,
                                  norm='bn', act='gelu')

        self._aspp_conv2 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate2, padding=rate2,
                                  bias=False,
                                  norm='bn', act='gelu')

        self._aspp_conv3 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate3, padding=rate3,
                                  bias=False,
                                  norm='bn', act='gelu')

        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._aspp_pool = ConvBN(in_channels, output_channels, kernel_size=1, bias=False,
                                 norm='bn', act='gelu')

        self._proj_conv_bn_act = ConvBN(output_channels * 5, output_channels, kernel_size=1, bias=False,
                                        norm='bn', act='gelu')
        # https://github.com/google-research/deeplab2/blob/main/model/decoder/aspp.py#L249
        self._proj_drop = nn.Dropout(p=0.1)

    def forward(self, x):
        results = []
        results.append(self._aspp_conv0(x))
        results.append(self._aspp_conv1(x))
        results.append(self._aspp_conv2(x))
        results.append(self._aspp_conv3(x))
        align_corners = (x.shape[-1] % 2 == 1)
        results.append(F.interpolate(self._aspp_pool(self._avg_pool(x)), size=x.shape[-2:], mode='bilinear',
                                     align_corners=align_corners))

        x = torch.cat(results, dim=1)
        x = self._proj_conv_bn_act(x)
        x = self._proj_drop(x)

        return x


class SemanticPredictor(nn.Module):
    def __init__(self, in_channels, os8_channels, os4_channels, num_classes):
        super().__init__()

        # Below is PanopticDeepLabSingleDecoder
        self._aspp = ASPP(
            in_channels=in_channels,
            # https://github.com/google-research/deeplab2/blob/main/configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto#L35
            output_channels=256,
            # https://github.com/google-research/deeplab2/blob/main/configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto#L36
            atrous_rates=[6, 12, 18])

        self._low_level_projection_os8 = ConvBN(os8_channels, 64, kernel_size=1, bias=False,
                                                norm='bn', act='gelu')

        self._low_level_fusion_os8_conv0_bn_act = ConvBN(256 + 64, 256 + 64, groups=256 + 64, kernel_size=5, padding=2,
                                                         bias=False,
                                                         norm='bn', act='gelu', conv_init='xavier_uniform')
        self._low_level_fusion_os8_conv1_bn_act = ConvBN(256 + 64, 256, kernel_size=1, bias=False,
                                                         norm='bn', act='gelu')

        self._low_level_projection_os4 = ConvBN(os4_channels, 32, kernel_size=1, bias=False,
                                                norm='bn', act='gelu')

        self._low_level_fusion_os4_conv0_bn_act = ConvBN(256 + 32, 256 + 32, groups=256 + 32, kernel_size=5, padding=2,
                                                         bias=False,
                                                         norm='bn', act='gelu', conv_init='xavier_uniform')
        self._low_level_fusion_os4_conv1_bn_act = ConvBN(256 + 32, 256, kernel_size=1, bias=False,
                                                         norm='bn', act='gelu')

        # Below is PanopticDeepLabSingleHead
        self.conv_block_0 = ConvBN(256, 256, groups=256, kernel_size=5, padding=2, bias=False,
                                   norm='bn', act='gelu', conv_init='xavier_uniform')
        self.conv_block_1 = ConvBN(256, 256, kernel_size=1, bias=False,
                                   norm='bn', act='gelu')
        self.final_conv = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None)
        trunc_normal_(self.final_conv.conv.weight, std=0.01)

    def forward(self, x, low_features_os8, low_features_os4):
        x = self._aspp(x)
        align_corners = (x.shape[-1] % 2 == 1)
        low_features_os8 = self._low_level_projection_os8(low_features_os8)
        x = F.interpolate(x, size=low_features_os8.shape[-2:], mode='bilinear', align_corners=align_corners)
        x = torch.concat([x, low_features_os8], dim=1)
        x = self._low_level_fusion_os8_conv0_bn_act(x)
        x = self._low_level_fusion_os8_conv1_bn_act(x)

        low_features_os4 = self._low_level_projection_os4(low_features_os4)
        x = F.interpolate(x, size=low_features_os4.shape[-2:], mode='bilinear', align_corners=align_corners)
        x = torch.concat([x, low_features_os4], dim=1)
        x = self._low_level_fusion_os4_conv0_bn_act(x)
        x = self._low_level_fusion_os4_conv1_bn_act(x)

        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.final_conv(x)
        return x

