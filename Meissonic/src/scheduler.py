# Copyright 2024 The HuggingFace Team and The MeissonFlow Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torchvision.transforms.functional as TF
import os

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def gumbel_noise(t, generator=None):
    device = generator.device if generator is not None else t.device
    noise = torch.zeros_like(t, device=device).uniform_(0, 1, generator=generator).to(t.device)
    return -torch.log((-torch.log(noise.clamp(1e-20))).clamp(1e-20))


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None, cur_entropy=None, logit_temperature=None, cur_step=None):
    if not logit_temperature==None:
        confidence = torch.log(probs.clamp(1e-20)) + temperature * logit_temperature * gumbel_noise(probs, generator=generator)
    else:
        confidence = torch.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    # # reshape and interpolate to 256x256 using nearest-neighbor
    # masking_map = masking.view(1, 1, 64, 64).float()  # shape: [1, 1, 64, 64]
    # upsampled = F.interpolate(masking_map, size=(256, 256), mode='nearest')  # still float in [0,1]
    # upsampled = upsampled.to(torch.uint8) * 255  # binary mask: 0 or 255

    # if not os.path.exists('./masking_map'): os.makedirs('./masking_map')
    # img = TF.to_pil_image(upsampled.squeeze(0))  # [1, 256, 256] -> [256, 256]
    # img.save("./masking_map/%d_th.png" % cur_step)
    return masking

def mask_by_random_topk_multinomial(mask_len, probs, generator=None, cur_step=None):
    """
    Args:
        mask_len: torch.Tensor, shape (B, 1)，每个样本要 mask 的数量
        probs: torch.Tensor, shape (B, N)，每个样本上 token 的概率分布
        generator: optional, torch.Generator

    Returns:
        masking: torch.BoolTensor, shape (B, N)
    """
    B, N = probs.shape
    masking = torch.zeros_like(probs, dtype=torch.bool)

    for b in range(B):
        k = int(mask_len[b].item())
        sampled_idx = torch.multinomial(probs[b], num_samples=k, replacement=False, generator=generator)
        masking[b, sampled_idx] = True
    masking_map = masking.view(64, 64).to(torch.uint8) * 255
    if not os.path.exists('./masking_map'):os.makedirs('./masking_map')
    img = TF.to_pil_image(masking_map)
    img.save("./masking_map/%d_th.png"%cur_step)
    return masking


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor = None


def top_k_filtering(logits,top_k,filter_value: float = -float("Inf")):
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value
    return logits


class Scheduler(SchedulerMixin, ConfigMixin):
    order = 1

    temperatures: torch.Tensor

    @register_to_config
    def __init__(
        self,
        mask_token_id: int,
        masking_schedule: str = "cosine",
    ):
        self.temperatures = None
        self.top_k = 0
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
        device: Union[str, torch.device] = None,
    ):
        self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)

        if isinstance(temperature, (tuple, list)):
            self.temperatures = torch.linspace(temperature[0], temperature[1], num_inference_steps, device=device)
        else:
            self.temperatures = torch.linspace(temperature, 0.01, num_inference_steps, device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.long,
        sample: torch.LongTensor,
        enable_entropy_filtering: bool = False,
        entropy_range: List = [], 
        temperature_value: List = [],
        starting_mask_ratio: int = 1,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        cur_step: int = 1,
    ) -> Union[SchedulerOutput, Tuple]:
        two_dim_input = sample.ndim == 3 and model_output.ndim == 4

        if two_dim_input:
            batch_size, codebook_size, height, width = model_output.shape
            sample = sample.reshape(batch_size, height * width)
            model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)#b,l,v

        unknown_map = sample == self.config.mask_token_id

        # top-k filtering
        if self.top_k>0:
            model_output=top_k_filtering(model_output,self.top_k)
        probs = model_output.softmax(dim=-1)
        if enable_entropy_filtering:
            cur_entropy = - (probs * torch.log(probs + 1e-12)).sum(dim=-1)#b,l
            logit_temperature=2.5*torch.exp(-cur_entropy/3)+0.7
            logits=model_output/logit_temperature[...,None]
            probs = logits.softmax(dim=-1)
        else:
            if len(entropy_range)>0:
                cur_entropy = - (probs * torch.log(probs + 1e-12)).sum(dim=-1)#b,l
                # logit_temperature=2.5*torch.exp(-cur_entropy/3)+0.7
                # logits=model_output/logit_temperature[...,None]
                # probs = logits.softmax(dim=-1)
                # print(cur_entropy.shape,model_output.shape)
                
                logit_temperature=torch.ones_like(cur_entropy)
                for (r, t) in zip(entropy_range, temperature_value):
                    mask = (cur_entropy >= r[0]) & (cur_entropy < r[1])#左闭右开
                    logit_temperature[mask] = t
                    logits=model_output/logit_temperature[...,None]
                    probs = logits.softmax(dim=-1)
            else:
                cur_entropy=None
                logit_temperature=None

        device = probs.device
        probs_ = probs.to(generator.device) if generator is not None else probs  # handles when generator is on CPU
        if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
            probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
        probs_ = probs_.reshape(-1, probs.size(-1))
        pred_original_sample = torch.multinomial(probs_, 1, generator=generator).to(device=device)
        pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])
        pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)

        if timestep == 0:
            prev_sample = pred_original_sample
            masking=torch.zeros(batch_size, height * width).bool()
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)

            if self.config.masking_schedule == "cosine":
                mask_ratio = torch.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "linear":
                mask_ratio = 1 - ratio
            else:
                raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

            mask_ratio = starting_mask_ratio * mask_ratio

            mask_len = (seq_len * mask_ratio).floor()
            # do not mask more than amount previously masked
            mask_len = torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            # mask at least one
            mask_len = torch.max(torch.tensor([1], device=model_output.device), mask_len)

            selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

            masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator, 
                                          cur_entropy=cur_entropy, logit_temperature=logit_temperature, cur_step=cur_step)
            # masking = mask_by_random_topk_multinomial(mask_len, selected_probs, generator, cur_step=cur_step)

            # Masks tokens with lower confidence.
            prev_sample = torch.where(masking, self.config.mask_token_id, pred_original_sample)

        if two_dim_input:
            prev_sample = prev_sample.reshape(batch_size, height, width)
            pred_original_sample = pred_original_sample.reshape(batch_size, height, width)

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return SchedulerOutput(prev_sample, pred_original_sample),masking

    def add_noise(self, sample, timesteps, generator=None):
        step_idx = (self.timesteps == timesteps).nonzero()
        ratio = (step_idx + 1) / len(self.timesteps)

        if self.config.masking_schedule == "cosine":
            mask_ratio = torch.cos(ratio * math.pi / 2)
        elif self.config.masking_schedule == "linear":
            mask_ratio = 1 - ratio
        else:
            raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

        mask_indices = (
            torch.rand(
                sample.shape, device=generator.device if generator is not None else sample.device, generator=generator
            ).to(sample.device)
            < mask_ratio
        )

        masked_sample = sample.clone()

        masked_sample[mask_indices] = self.config.mask_token_id

        return masked_sample
