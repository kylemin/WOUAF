import torch
import torch.nn.functional as F
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Dict, List, Optional, Tuple, Union
from ldm.modules.diffusionmodules.model import ResnetBlock as ResnetBlock2D
from ldm.modules.diffusionmodules.model import AttnBlock as AttentionBlock
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D, CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D, CrossAttnUpBlock2D
# from diffusers.models.resnet import ResnetBlock2D
# from diffusers.models.attention import AttentionBlock
from diffusers.models.cross_attention import CrossAttention
from attribution import FullyConnectedLayer_normal
import math


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
def customize_vae_decoder(vae, phi_dimension, modulation, finetune, weight_offset):
    q = 'q' in modulation
    k = 'k' in modulation
    v = 'v' in modulation
    d = 'd' in modulation
    e = 'e' in modulation

    def add_affine_conv(vaed):
        if not (d or e):
            return

        for layer in vaed.children():
            if type(layer) == ResnetBlock2D:
                if d:
                    layer.affine_d = FullyConnectedLayer_normal(phi_dimension, layer.conv1.weight.shape[1], bias_init=1)
                if e:
                    layer.affine_e = FullyConnectedLayer_normal(phi_dimension, layer.conv2.weight.shape[1], bias_init=1)
            else:
                add_affine_conv(layer)

    def add_affine_attn(vaed):
        if not (q or k or v):
            return

        for layer in vaed.children():
            if type(layer) == AttentionBlock:
                if q:
                    layer.affine_q = FullyConnectedLayer_normal(phi_dimension, layer.q.weight.shape[1], bias_init=1)
                if k:
                    layer.affine_k = FullyConnectedLayer_normal(phi_dimension, layer.k.weight.shape[1], bias_init=1)
                if v:
                    layer.affine_v = FullyConnectedLayer_normal(phi_dimension, layer.v.weight.shape[1], bias_init=1)
            else:
                add_affine_attn(layer)

    def impose_grad_condition(vaed, finetune):
        if finetune == 'all':
            return

        for name, params in vaed.named_parameters():
            requires_grad = False
            if finetune == 'match':
                q_cond = q and (('attn' in name and 'q' in name) or 'affine_q' in name)
                k_cond = k and (('attn' in name and 'k' in name) or 'affine_k' in name)
                v_cond = v and (('attn' in name and 'v' in name) or 'affine_v' in name)
                d_cond = d and (('block' in name and 'conv1' in name) or 'affine_d' in name)
                e_cond = e and (('block' in name and 'conv2' in name) or 'affine_e' in name)
                if q_cond or k_cond or v_cond or d_cond or e_cond:
                    requires_grad = True
                params.requires_grad = requires_grad

    def change_forward(vaed, layer_type, new_forward):
        for layer in vaed.children():
            if type(layer) == layer_type:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer, layer_type, new_forward)

    def change_forward_no_type(vaed, new_forward):
        layer = vaed
        bound_method = new_forward.__get__(layer, layer.__class__)
        setattr(layer, 'forward', bound_method)


    def new_forward_MB(self, hidden_states, encoded_fingerprint, temb=None):
        hidden_states = self.mid.block_1((hidden_states, encoded_fingerprint), temb)
        hidden_states = self.mid.attn_1((hidden_states, encoded_fingerprint))
        hidden_states = self.mid.block_2((hidden_states, encoded_fingerprint), temb)

        return hidden_states

    def new_forward_UDB(self, hidden_states, encoded_fingerprint):
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                hidden_states = self.up[i_level].block[i_block]((hidden_states, encoded_fingerprint), temb=None)
                if len(self.up[i_level].attn) > 0:
                    hidden_states = self.up[i_level].attn[i_block](hidden_states)
            if i_level != 0:
                hidden_states = self.up[i_level].upsample(hidden_states)
        # for resnet in self.resnets:
        #     hidden_states = resnet((hidden_states, encoded_fingerprint), temb=None)
        #
        # if self.upsamplers is not None:
        #     for upsampler in self.upsamplers:
        #         hidden_states = upsampler(hidden_states)

        return hidden_states

    def new_forward_RB(self, input_tensor, temb=None):
        input_tensor, encoded_fingerprint = input_tensor
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = nonlinearity(hidden_states)

        # if self.upsample is not None:
        #     # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        #     if hidden_states.shape[0] >= 64:
        #         input_tensor = input_tensor.contiguous()
        #         hidden_states = hidden_states.contiguous()
        #     input_tensor = self.upsample(input_tensor)
        #     hidden_states = self.upsample(hidden_states)
        # elif self.downsample is not None:
        #     input_tensor = self.downsample(input_tensor)
        #     hidden_states = self.downsample(hidden_states)

        if d:
            phis = self.affine_d(encoded_fingerprint)
            batch_size = phis.shape[0]
            if not weight_offset:
                weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
            else:
                weight = self.conv1.weight
                weight_mod = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
                weight = weight.unsqueeze(0) + weight_mod
            hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv1.bias.view(1, -1, 1, 1)
        else:
            hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.temb_proj(nonlinearity(temb))[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        if e:
            phis = self.affine_e(encoded_fingerprint)
            batch_size = phis.shape[0]
            if not weight_offset:
                weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv2.weight.unsqueeze(0)
            else:
                weight = self.conv2.weight
                weight_mod = phis.view(batch_size, 1, -1, 1, 1) * self.conv2.weight.unsqueeze(0)
                weight = weight.unsqueeze(0) + weight_mod
            hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv2.bias.view(1, -1, 1, 1)
        else:
            hidden_states = self.conv2(hidden_states)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                input_tensor = self.conv_shortcut(input_tensor)
            else:
                input_tensor = self.nin_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states)

        return output_tensor

    def new_forward_AB(self, hidden_states):
        hidden_states, encoded_fingerprint = hidden_states
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        if q:
            phis_q = self.affine_q(encoded_fingerprint)
            if not weight_offset:
                query_proj = torch.bmm(hidden_states, phis_q.unsqueeze(-1) * self.q.weight.t().unsqueeze(0)) + self.q.bias
            else:
                qw = self.q.weight
                qw_mod = phis_q.unsqueeze(-1) * qw.t().unsqueeze(0)
                query_proj = torch.bmm(hidden_states, qw.t().unsqueeze(0) + qw_mod) + self.q.bias
        else:
            query_proj = self.q(hidden_states)

        if k:
            phis_k = self.affine_k(encoded_fingerprint)
            if not weight_offset:
                key_proj = torch.bmm(hidden_states, phis_k.unsqueeze(-1) * self.k.weight.t().unsqueeze(0)) + self.k.bias
            else:
                kw = self.k.weight
                kw_mod = phis_k.unsqueeze(-1) * kw.t().unsqueeze(0)
                key_proj = torch.bmm(hidden_states, kw.t().unsqueeze(0) + kw_mod) + self.k.bias
        else:
            key_proj = self.k(hidden_states)

        if v:
            phis_v = self.affine_v(encoded_fingerprint)
            if not weight_offset:
                value_proj = torch.bmm(hidden_states, phis_v.unsqueeze(-1) * self.v.weight.t().unsqueeze(0)) + self.v.bias
            else:
                vw = self.v.weight
                vw_mod = phis_v.unsqueeze(-1) * vw.t().unsqueeze(0)
                value_proj = torch.bmm(hidden_states, vw.t().unsqueeze(0) + vw_mod) + self.v.bias
        else:
            value_proj = self.v(hidden_states)

        scale = 1 / math.sqrt(self.channels / self.num_heads)

        query_proj = self.reshape_heads_to_batch_dim(query_proj)
        key_proj = self.reshape_heads_to_batch_dim(key_proj)
        value_proj = self.reshape_heads_to_batch_dim(value_proj)

        if self._use_memory_efficient_attention_xformers:
            # Memory efficient attention
            hidden_states = xformers.ops.memory_efficient_attention(
                query_proj, key_proj, value_proj, attn_bias=None, op=self._attention_op
            )
            hidden_states = hidden_states.to(query_proj.dtype)
        else:
            attention_scores = torch.baddbmm(
                torch.empty(
                    query_proj.shape[0],
                    query_proj.shape[1],
                    key_proj.shape[1],
                    dtype=query_proj.dtype,
                    device=query_proj.device,
                ),
                query_proj,
                key_proj.transpose(-1, -2),
                beta=0,
                alpha=scale,
            )
            attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
            hidden_states = torch.bmm(attention_probs, value_proj)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    # Reference: https://github.com/huggingface/diffusers
    def new_forward_vaed(self, z, enconded_fingerprint):
        # sample = z
        # sample = self.conv_in(sample)
        #
        # # middle
        # sample = self.mid_block(sample, enconded_fingerprint)
        #
        # # up
        # for up_block in self.up_blocks:
        #     sample = up_block(sample, enconded_fingerprint)
        #
        # # post-process
        # sample = self.conv_norm_out(sample)
        # sample = self.conv_act(sample)
        # sample = self.conv_out(sample)

        sample = self.conv_in(z)

        # middle
        sample = self.mid.block_1((sample, enconded_fingerprint))
        sample = self.mid.attn_1(sample)
        sample = self.mid.block_2((sample, enconded_fingerprint))

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                sample = self.up[i_level].block[i_block]((sample, enconded_fingerprint))
                if len(self.up[i_level].attn) > 0:
                    sample = self.up[i_level].attn[i_block](sample)
            if i_level != 0:
                sample = self.up[i_level].upsample(sample)

        # end
        if self.give_pre_end:
            return sample

        sample = self.norm_out(sample)
        sample = nonlinearity(sample)
        sample = self.conv_out(sample)

        return sample

    @dataclass
    class DecoderOutput(BaseOutput):
        """
        Output of decoding method.
        Args:
            sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Decoded output sample of the model. Output of the last layer of the model.
        """

        sample: torch.FloatTensor

    def new__decode(self, z: torch.FloatTensor, encoded_fingerprint: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, encoded_fingerprint)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def new_decode(self, z: torch.FloatTensor, encoded_fingerprint: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        # if self.use_slicing and z.shape[0] > 1:
        #     decoded_slices = [self._decode(z_slice, encoded_fingerprint) for z_slice in z.split(1)]
        #     decoded = torch.cat(decoded_slices)
        # else:
        decoded = self._decode(z, encoded_fingerprint)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    add_affine_conv(vae.decoder)
    add_affine_attn(vae.decoder)
    impose_grad_condition(vae.decoder, finetune)
    # change_forward_no_type(vae.decoder, new_forward_MB)
    # change_forward_no_type(vae.decoder, new_forward_UDB)
    change_forward(vae.decoder, ResnetBlock2D, new_forward_RB)
    # change_forward(vae.decoder, AttentionBlock, new_forward_AB)
    setattr(vae.decoder, 'forward', new_forward_vaed.__get__(vae.decoder, vae.decoder.__class__))
    setattr(vae, '_decode', new__decode.__get__(vae, vae.__class__))
    setattr(vae, 'decode', new_decode.__get__(vae, vae.__class__))

    return vae

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from notebook_helpers import instantiate_from_config
    config = OmegaConf.load("configs/autoencoder/autoencoder_kl_8x8x64.yaml")
    model = instantiate_from_config(config.model)
    customize_vae_decoder(model, 32, ['d', 'q'], 'match', True)


# def customize_unet(unet, phi_dimension, modulation, finetune, weight_offset):
#     c = 'c' in modulation
#     q = 'q' in modulation
#     k = 'k' in modulation
#     v = 'v' in modulation
#
#     def add_affine_conv(unet):
#         if not c:
#             return
#
#         for layer in unet.children():
#             if type(layer) == ResnetBlock2D:
#                 layer.affine_c = FullyConnectedLayer_normal(phi_dimension, layer.conv1.weight.shape[1], bias_init=1)
#             else:
#                 add_affine_conv(layer)
#
#     def add_affine_crossattn(unet):
#         if not (q or k or v):
#             return
#
#         for layer in unet.children():
#             if type(layer) == CrossAttention:
#                 if q:
#                     layer.affine_q = FullyConnectedLayer_normal(phi_dimension, layer.to_q.weight.shape[1], bias_init=1)
#                 if k:
#                     layer.affine_k = FullyConnectedLayer_normal(phi_dimension, layer.to_k.weight.shape[1], bias_init=1)
#                 if v:
#                     layer.affine_v = FullyConnectedLayer_normal(phi_dimension, layer.to_v.weight.shape[1], bias_init=1)
#             else:
#                 add_affine_crossattn(layer)
#
#     def impose_grad_condition(unet, finetune):
#         if finetune == 'all':
#             return
#
#         for name, params in unet.named_parameters():
#             requires_grad = False
#             if finetune == 'match':
#                 q_cond = q and (('attn' in name and 'to_q' in name) or 'affine_q' in name)
#                 k_cond = k and (('attn' in name and 'to_k' in name) or 'affine_k' in name)
#                 v_cond = v and (('attn' in name and 'to_v' in name) or 'affine_v' in name)
#                 c_cond = c and (('resnets' in name and 'conv1' in name) or 'affine_c' in name)
#                 if q_cond or k_cond or v_cond or c_cond:
#                     requires_grad = True
#                 params.requires_grad = requires_grad
#
#     def change_forward(unet, layer_type, new_forward):
#         for layer in unet.children():
#             if type(layer) == layer_type:
#                 bound_method = new_forward.__get__(layer, layer.__class__)
#                 setattr(layer, 'forward', bound_method)
#             else:
#                 change_forward(layer, layer_type, new_forward)
#
#     def new_forward_CADB(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs=None):
#         hidden_states, encoded_fingerprint = hidden_states
#         output_states = ()
#
#         for resnet, attn in zip(self.resnets, self.attentions):
#             if self.training and self.gradient_checkpointing:
#
#                 def create_custom_forward(module, return_dict=None):
#                     def custom_forward(*inputs):
#                         if return_dict is not None:
#                             return module(*inputs, return_dict=return_dict)
#                         else:
#                             return module(*inputs)
#
#                     return custom_forward
#
#                 hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), (hidden_states, encoded_fingerprint), temb)
#                 hidden_states = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(attn, return_dict=False),
#                     hidden_states,
#                     (encoder_hidden_states, encoded_fingerprint),
#                     cross_attention_kwargs,
#                 )[0]
#             else:
#                 hidden_states = resnet((hidden_states, encoded_fingerprint), temb)
#                 hidden_states = attn(
#                     hidden_states,
#                     encoder_hidden_states=(encoder_hidden_states, encoded_fingerprint),
#                     cross_attention_kwargs=cross_attention_kwargs,
#                 ).sample
#
#             output_states += (hidden_states,)
#
#         if self.downsamplers is not None:
#             for downsampler in self.downsamplers:
#                 hidden_states = downsampler(hidden_states)
#
#             output_states += (hidden_states,)
#
#         return hidden_states, output_states
#
#     def new_forward_DB(self, hidden_states, temb=None):
#         hidden_states, encoded_fingerprint = hidden_states
#         output_states = ()
#
#         for resnet in self.resnets:
#             if self.training and self.gradient_checkpointing:
#
#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         return module(*inputs)
#
#                     return custom_forward
#
#                 hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), (hidden_states, encoded_fingerprint), temb)
#             else:
#                 hidden_states = resnet((hidden_states, encoded_fingerprint), temb)
#
#             output_states += (hidden_states,)
#
#         if self.downsamplers is not None:
#             for downsampler in self.downsamplers:
#                 hidden_states = downsampler(hidden_states)
#
#             output_states += (hidden_states,)
#
#         return hidden_states, output_states
#
#     def new_forward_CAMB(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs=None):
#         hidden_states, encoded_fingerprint = hidden_states
#         hidden_states = self.resnets[0]((hidden_states, encoded_fingerprint), temb)
#         for attn, resnet in zip(self.attentions, self.resnets[1:]):
#             hidden_states = attn(
#                 hidden_states,
#                 encoder_hidden_states=(encoder_hidden_states, encoded_fingerprint),
#                 cross_attention_kwargs=cross_attention_kwargs,
#             ).sample
#             hidden_states = resnet((hidden_states, encoded_fingerprint), temb)
#
#         return hidden_states
#
#     def new_forward_UB(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
#         hidden_states, encoded_fingerprint = hidden_states
#         for resnet in self.resnets:
#             # pop res hidden states
#             res_hidden_states = res_hidden_states_tuple[-1]
#             res_hidden_states_tuple = res_hidden_states_tuple[:-1]
#             hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
#
#             if self.training and self.gradient_checkpointing:
#
#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         return module(*inputs)
#
#                     return custom_forward
#
#                 hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), (hidden_states, encoded_fingerprint), temb)
#             else:
#                 hidden_states = resnet((hidden_states, encoded_fingerprint), temb)
#
#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 hidden_states = upsampler(hidden_states, upsample_size)
#
#         return hidden_states
#
#     def new_forward_CAUB(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None, cross_attention_kwargs=None, upsample_size=None, attention_mask=None):
#         hidden_states, encoded_fingerprint = hidden_states
#         for resnet, attn in zip(self.resnets, self.attentions):
#             # pop res hidden states
#             res_hidden_states = res_hidden_states_tuple[-1]
#             res_hidden_states_tuple = res_hidden_states_tuple[:-1]
#             hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
#
#             if self.training and self.gradient_checkpointing:
#
#                 def create_custom_forward(module, return_dict=None):
#                     def custom_forward(*inputs):
#                         if return_dict is not None:
#                             return module(*inputs, return_dict=return_dict)
#                         else:
#                             return module(*inputs)
#
#                     return custom_forward
#
#                 hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), (hidden_states, encoded_fingerprint), temb)
#                 hidden_states = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(attn, return_dict=False),
#                     hidden_states,
#                     (encoder_hidden_states, encoded_fingerprint),
#                     cross_attention_kwargs,
#                 )[0]
#             else:
#                 hidden_states = resnet((hidden_states, encoded_fingerprint), temb)
#                 hidden_states = attn(
#                     hidden_states,
#                     encoder_hidden_states=(encoder_hidden_states, encoded_fingerprint),
#                     cross_attention_kwargs=cross_attention_kwargs,
#                 ).sample
#
#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 hidden_states = upsampler(hidden_states, upsample_size)
#
#         return hidden_states
#
#     def new_forward_RB(self, input_tensor, temb):
#         input_tensor, encoded_fingerprint = input_tensor
#         hidden_states = input_tensor
#
#         hidden_states = self.norm1(hidden_states)
#         hidden_states = self.nonlinearity(hidden_states)
#
#         if self.upsample is not None:
#             # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
#             if hidden_states.shape[0] >= 64:
#                 input_tensor = input_tensor.contiguous()
#                 hidden_states = hidden_states.contiguous()
#             input_tensor = self.upsample(input_tensor)
#             hidden_states = self.upsample(hidden_states)
#         elif self.downsample is not None:
#             input_tensor = self.downsample(input_tensor)
#             hidden_states = self.downsample(hidden_states)
#
#         if c:
#             phis = self.affine_c(encoded_fingerprint)
#             batch_size = phis.shape[0]
#             if not weight_offset:
#                 weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
#             else:
#                 weight = self.conv1.weight
#                 weight_mod = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
#                 weight = weight.unsqueeze(0) + weight_mod
#             hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv1.bias.view(1, -1, 1, 1)
#         else:
#             hidden_states = self.conv1(hidden_states)
#
#         if temb is not None:
#             temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
#
#         if temb is not None and self.time_embedding_norm == "default":
#             hidden_states = hidden_states + temb
#
#         hidden_states = self.norm2(hidden_states)
#
#         if temb is not None and self.time_embedding_norm == "scale_shift":
#             scale, shift = torch.chunk(temb, 2, dim=1)
#             hidden_states = hidden_states * (1 + scale) + shift
#
#         hidden_states = self.nonlinearity(hidden_states)
#
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.conv2(hidden_states)
#
#         if self.conv_shortcut is not None:
#             input_tensor = self.conv_shortcut(input_tensor)
#
#         output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
#
#         return output_tensor
#
#     def new_forward_CA(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
#         batch_size, sequence_length, _ = hidden_states.shape
#         attention_mask = self.prepare_attention_mask(attention_mask, sequence_length)
#
#         crossattn = False
#         if encoder_hidden_states is not None:
#             encoder_hidden_states, encoded_fingerprint = encoder_hidden_states
#             if encoded_fingerprint is not None and (q or k or v):
#                 crossattn = True
#
#         encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
#
#         if not crossattn:
#             query = self.to_q(hidden_states)
#             key = self.to_k(encoder_hidden_states)
#             value = self.to_v(encoder_hidden_states)
#         else:
#             if q:
#                 phis_q = self.affine_q(encoded_fingerprint)
#                 if not weight_offset:
#                     query = torch.bmm(hidden_states, phis_q.unsqueeze(-1) * self.to_q.weight.t().unsqueeze(0))
#                 else:
#                     qw = self.to_q.weight
#                     qw_mod = phis_q.unsqueeze(-1) * qw.t().unsqueeze(0)
#                     query = torch.bmm(hidden_states, qw.t().unsqueeze(0) + qw_mod)
#             else:
#                 query = self.to_q(hidden_states)
#
#             if k:
#                 phis_k = self.affine_k(encoded_fingerprint)
#                 if not weight_offset:
#                     key = torch.bmm(encoder_hidden_states, phis_k.unsqueeze(-1) * self.to_k.weight.t().unsqueeze(0))
#                 else:
#                     kw = self.to_k.weight
#                     kw_mod = phis_k.unsqueeze(-1) * kw.t().unsqueeze(0)
#                     key = torch.bmm(encoder_hidden_states, kw.t().unsqueeze(0) + kw_mod)
#             else:
#                 key = self.to_k(encoder_hidden_states)
#
#             if v:
#                 phis_v = self.affine_v(encoded_fingerprint)
#                 if not weight_offset:
#                     value = torch.bmm(encoder_hidden_states, phis_v.unsqueeze(-1) * self.to_v.weight.t().unsqueeze(0))
#                 else:
#                     vw = self.to_v.weight
#                     vw_mod = phis_v.unsqueeze(-1) * vw.t().unsqueeze(0)
#                     value = torch.bmm(encoder_hidden_states, vw.t().unsqueeze(0) + vw_mod)
#             else:
#                 value = self.to_v(encoder_hidden_states)
#
#         query = self.head_to_batch_dim(query)
#         key = self.head_to_batch_dim(key)
#         value = self.head_to_batch_dim(value)
#
#         attention_probs = self.get_attention_scores(query, key, attention_mask)
#         hidden_states = torch.bmm(attention_probs, value)
#         hidden_states = self.batch_to_head_dim(hidden_states)
#
#         # linear proj
#         hidden_states = self.to_out[0](hidden_states)
#         # dropout
#         hidden_states = self.to_out[1](hidden_states)
#
#         return hidden_states
#
#     # Reference: https://github.com/huggingface/diffusers
#     @dataclass
#     class UNet2DConditionOutput(BaseOutput):
#         """
#         Args:
#             sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
#                 Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
#         """
#
#         sample: torch.FloatTensor
#
#     # Reference: https://github.com/huggingface/diffusers
#     def new_forward_unet(
#         self,
#         sample: torch.FloatTensor,
#         timestep: Union[torch.Tensor, float, int],
#         encoder_hidden_states: torch.Tensor,
#         encoded_fingerprint: torch.Tensor,
#         class_labels: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#         return_dict: bool = True,
#     ) -> Union[UNet2DConditionOutput, Tuple]:
#         r"""
#         Args:
#             sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
#             timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
#             encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
#             return_dict (`bool`, *optional*, defaults to `True`):
#                 Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
#         Returns:
#             [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
#             [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
#             returning a tuple, the first element is the sample tensor.
#         """
#         # By default samples have to be AT least a multiple of the overall upsampling factor.
#         # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
#         # However, the upsampling interpolation output size can be forced to fit any upsampling size
#         # on the fly if necessary.
#         default_overall_up_factor = 2**self.num_upsamplers
#
#         # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
#         forward_upsample_size = False
#         upsample_size = None
#
#         if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
#             logger.info("Forward upsample size to force interpolation output size.")
#             forward_upsample_size = True
#
#         # prepare attention_mask
#         if attention_mask is not None:
#             attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
#             attention_mask = attention_mask.unsqueeze(1)
#
#         # 0. center input if necessary
#         if self.config.center_input_sample:
#             sample = 2 * sample - 1.0
#
#         # 1. time
#         timesteps = timestep
#         if not torch.is_tensor(timesteps):
#             # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
#             # This would be a good case for the `match` statement (Python 3.10+)
#             is_mps = sample.device.type == "mps"
#             if isinstance(timestep, float):
#                 dtype = torch.float32 if is_mps else torch.float64
#             else:
#                 dtype = torch.int32 if is_mps else torch.int64
#             timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
#         elif len(timesteps.shape) == 0:
#             timesteps = timesteps[None].to(sample.device)
#
#         # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#         timesteps = timesteps.expand(sample.shape[0])
#
#         t_emb = self.time_proj(timesteps)
#
#         # timesteps does not contain any weights and will always return f32 tensors
#         # but time_embedding might actually be running in fp16. so we need to cast here.
#         # there might be better ways to encapsulate this.
#         t_emb = t_emb.to(dtype=self.dtype)
#         emb = self.time_embedding(t_emb)
#
#         if self.class_embedding is not None:
#             if class_labels is None:
#                 raise ValueError("class_labels should be provided when num_class_embeds > 0")
#
#             if self.config.class_embed_type == "timestep":
#                 class_labels = self.time_proj(class_labels)
#
#             class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
#             emb = emb + class_emb
#
#         # 2. pre-process
#         sample = self.conv_in(sample)
#
#         # 3. down
#         down_block_res_samples = (sample,)
#         for downsample_block in self.down_blocks:
#             if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
#                 sample, res_samples = downsample_block(
#                     hidden_states=(sample, encoded_fingerprint),
#                     temb=emb,
#                     encoder_hidden_states=encoder_hidden_states,
#                     attention_mask=attention_mask,
#                     cross_attention_kwargs=cross_attention_kwargs,
#                 )
#             else:
#                 sample, res_samples = downsample_block(hidden_states=(sample, encoded_fingerprint), temb=emb)
#
#             down_block_res_samples += res_samples
#
#         # 4. mid
#         sample = self.mid_block(
#             (sample, encoded_fingerprint),
#             emb,
#             encoder_hidden_states=encoder_hidden_states,
#             attention_mask=attention_mask,
#             cross_attention_kwargs=cross_attention_kwargs,
#         )
#
#         # 5. up
#         for i, upsample_block in enumerate(self.up_blocks):
#             is_final_block = i == len(self.up_blocks) - 1
#
#             res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
#             down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
#
#             # if we have not reached the final block and need to forward the
#             # upsample size, we do it here
#             if not is_final_block and forward_upsample_size:
#                 upsample_size = down_block_res_samples[-1].shape[2:]
#
#             if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
#                 sample = upsample_block(
#                     hidden_states=(sample, encoded_fingerprint),
#                     temb=emb,
#                     res_hidden_states_tuple=res_samples,
#                     encoder_hidden_states=encoder_hidden_states,
#                     cross_attention_kwargs=cross_attention_kwargs,
#                     upsample_size=upsample_size,
#                     attention_mask=attention_mask,
#                 )
#             else:
#                 sample = upsample_block(
#                     hidden_states=(sample, encoded_fingerprint), temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
#                 )
#         # 6. post-process
#         sample = self.conv_norm_out(sample)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)
#
#         if not return_dict:
#             return (sample,)
#
#         return UNet2DConditionOutput(sample=sample)
#
#
#     add_affine_conv(unet)
#     add_affine_crossattn(unet)
#     impose_grad_condition(unet, finetune)
#     change_forward(unet, CrossAttnDownBlock2D, new_forward_CADB)
#     change_forward(unet, DownBlock2D, new_forward_DB)
#     change_forward(unet, UNetMidBlock2DCrossAttn, new_forward_CAMB)
#     change_forward(unet, UpBlock2D, new_forward_UB)
#     change_forward(unet, CrossAttnUpBlock2D, new_forward_CAUB)
#     change_forward(unet, ResnetBlock2D, new_forward_RB)
#     change_forward(unet, CrossAttention, new_forward_CA)
#     bound_method = new_forward_unet.__get__(unet, unet.__class__)
#     setattr(unet, 'forward', bound_method)
#
#     return unet
