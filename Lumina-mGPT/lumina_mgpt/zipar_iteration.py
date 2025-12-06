import math
from typing import Optional, Tuple, Union, List, Dict, Sequence, Any
import random
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import pdb

from transformers.cache_utils import Cache
import json
import copy
import numpy as np

from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from transformers.utils import ModelOutput, is_torchdynamo_compiling
from transformers.generation.utils import (
    GenerateNonBeamOutput, 
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LogitsWarper
import transformers

logger = logging.get_logger(__name__)

from absl import logging
import time
_CONFIG_FOR_DOC = "ChameleonConfig"
_CHECKPOINT_FOR_DOC = "meta/chameleon-7b"
_EXPECTED_OUTPUT_SHAPE = [1, 7, 4096]
_SEQ_CLASS_EXPECTED_LOSS = 1.03
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"


class LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        image_start_token_id,
        image_end_token_id,
        image_next_line_token_id,
        patch_size,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context_backup = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": transformers.DynamicCache() if use_cache else None,
            "first_pass": True,
        }
        self.unconditional_context = None

        self.nums_image_start_tokens = None

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

    def get_unconditional_logits(self, input_ids, image_start_token_id_index):

        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, image_start_token_id_index:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if num_image_start_tokens == num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            self.unconditional_context = None
            return scores

        elif num_image_start_tokens == num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2

                if self.unconditional_context is None:
                    self.unconditional_context = copy.deepcopy(self.unconditional_context_backup)

                if self.guidance_scale == 1.0:
                    return scores

                unconditional_logits = self.get_unconditional_logits(input_ids, self.image_start_token_id_index)[:, -1]

                scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
                return scores_processed

        else:
            print("Something wrong in the decoding process.")

        return scores


class MultiModalLogitsProcessor_ZIPAR(LogitsProcessor):

    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
    ):
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

        self.vocab_list = [i for i in range(voc_size)]
        self.image_token_list = [i for i in range(4, 8195 + 1)]
        self.suppress_tokens = torch.tensor(
            [x for x in self.vocab_list if x not in self.image_token_list], device="cuda"
        )

        self.vocab_tensor = torch.arange(voc_size, device="cuda")
        self.suppress_token_mask = torch.isin(self.vocab_tensor, self.suppress_tokens)
        self.new_line_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_next_line_token_id], device="cuda")
        )
        self.eos_image_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_end_token_id], device="cuda")
        )

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # print(self.num_image_start_tokens, self.num_image_end_tokens)

        if self.num_image_start_tokens == self.num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            return scores

        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
                print(self.image_start_token_id_index)
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()

            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            # print(f"num new tokens: {new_token_num}")
            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    # print(f"h_grids: {h_grids}, w_grids: {w_grids}")
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2
                    print(f"h_latent_dim: {self.h_latent_dim}, w_latent_dim: {self.w_latent_dim}")

                tokens = input_ids[0][self.image_start_token_id_index + 3 :]
                # if (len(tokens) + 1) % (self.w_latent_dim + 1) == 0: ## we do not need it for IPD decoding
                #     new_line_constrained_scores = torch.full_like(scores, -math.inf)
                #     new_line_constrained_scores[:, self.image_next_line_token_id] = 0
                #     print(f"new line: {len(tokens)+1}")
                #     return new_line_constrained_scores
                if (len(tokens) + 1) == (self.w_latent_dim + 1) * self.h_latent_dim + 1:
                    eos_image_constrained_scores = torch.full_like(scores, -math.inf)
                    eos_image_constrained_scores[:, self.image_end_token_id] = 0
                    print(f"eos image: {len(tokens)+1}")
                    return eos_image_constrained_scores
                # elif (len(tokens) + 1) % (self.w_latent_dim + 1) != 0:
                else:
                    image_constrained_scores = torch.where(self.suppress_token_mask, -float("inf"), scores)
                    return image_constrained_scores
        else:
            print("Something wrong in the decoding process.")

        return scores

class MultiModalLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
    ):
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

        self.vocab_list = [i for i in range(voc_size)]
        self.image_token_list = [i for i in range(4, 8195 + 1)]
        self.suppress_tokens = torch.tensor(
            [x for x in self.vocab_list if x not in self.image_token_list], device="cuda"
        )

        self.vocab_tensor = torch.arange(voc_size, device="cuda")
        self.suppress_token_mask = torch.isin(self.vocab_tensor, self.suppress_tokens)
        self.new_line_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_next_line_token_id], device="cuda")
        )
        self.eos_image_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_end_token_id], device="cuda")
        )

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # print(self.num_image_start_tokens, self.num_image_end_tokens)

        if self.num_image_start_tokens == self.num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            return scores

        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
                print(self.image_start_token_id_index)
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()

            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            # print(f"num new tokens: {new_token_num}")
            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    # print(f"h_grids: {h_grids}, w_grids: {w_grids}")
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2
                    print(f"h_latent_dim: {self.h_latent_dim}, w_latent_dim: {self.w_latent_dim}")

                tokens = input_ids[0][self.image_start_token_id_index + 3 :]
                if (len(tokens) + 1) % (self.w_latent_dim + 1) == 0:
                    new_line_constrained_scores = torch.full_like(scores, -math.inf)
                    new_line_constrained_scores[:, self.image_next_line_token_id] = 0
                    print(f"new line: {len(tokens)+1}")
                    return new_line_constrained_scores
                elif (len(tokens) + 1) == (self.w_latent_dim + 1) * self.h_latent_dim + 1:
                    eos_image_constrained_scores = torch.full_like(scores, -math.inf)
                    eos_image_constrained_scores[:, self.image_end_token_id] = 0
                    print(f"eos image: {len(tokens)+1}")
                    return eos_image_constrained_scores
                elif (len(tokens) + 1) % (self.w_latent_dim + 1) != 0:
                    image_constrained_scores = torch.where(self.suppress_token_mask, -float("inf"), scores)
                    return image_constrained_scores
        else:
            print("Something wrong in the decoding process.")

        return scores

class InterleavedTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements. Often used together
    with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].
    """

    def __init__(
        self,
        image_top_k: int,
        text_top_k: int,
        image_start_token_id=None,
        image_end_token_id=None,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(text_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`text_top_k` has to be a strictly positive integer, but is {text_top_k}")
        if not isinstance(image_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`image_top_k` has to be a strictly positive integer, but is {image_top_k}")

        self.image_top_k = max(image_top_k, min_tokens_to_keep)
        self.text_top_k = max(text_top_k, min_tokens_to_keep)
        self.filter_value = filter_value

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if self.num_image_start_tokens == self.num_image_end_tokens + 1:
            top_k = min(self.image_top_k, scores.size(-1))
        else:
            top_k = min(self.text_top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def delete_false_key_value(
    self,
    num_of_false_tokens,
) -> Tuple[torch.Tensor, torch.Tensor]:

    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

def postprocess_cfg_decode(
    model_inputs,
    cfg_half_name_list=['inputs_embeds', 'input_ids', 'pixel_values', ],
):
    cfg_half_name_list = cfg_half_name_list
    def cfg_half(x):
        return x[:x.shape[0]//2]
    
    for name in cfg_half_name_list:
        if (name in model_inputs) and (model_inputs[name] is not None):
            model_inputs[name] = cfg_half(model_inputs[name])
    
    return model_inputs

def check_is_force_no_cfg(input_ids, image_start_token_id=None, image_end_token_id=None, guidance_scale=3., do_cfg=True):
    if (image_start_token_id is None) or (image_end_token_id is None):
        return False
    
    num_image_start_tokens = (input_ids[0] == image_start_token_id).sum()
    num_image_end_tokens = (input_ids[0] == image_end_token_id).sum()

    if num_image_start_tokens == num_image_end_tokens:
        return True
    else:
        return False

def sampling_logits2tokens(
    logits,
    all_collected_input_ids,
    unfinished_sequences, pad_token_id,
    output_token_num = 1,
    logits_processor=None, logits_warper=None,
    do_sample=True,
    has_eos_stopping_criteria=True,
    do_cfg=False,
    guidance_scale=3.,
    generator=None, #token_sampler = None,
    is_force_no_cfg = False,
):
    # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
    # (the clone itself is always small)
    next_token_logits = logits[ :, -output_token_num:, : ].clone()

    if do_cfg:
        conditional_logits, unconditional_logits = next_token_logits.chunk(2, dim=0)
        if is_force_no_cfg:
            next_token_logits = conditional_logits
        else:
            next_token_logits = guidance_scale * (conditional_logits - unconditional_logits) + unconditional_logits
    
    next_token_scores = logits_processor(all_collected_input_ids, next_token_logits)
    if do_sample and (logits_warper is not None):
        next_token_scores = logits_warper(all_collected_input_ids, next_token_scores)

    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
        probs_shape = None
        if len(probs.shape) >= 3:
            probs_shape = probs.shape
            probs = probs.flatten(0, len(probs_shape)-2)

        next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        if probs_shape is not None:
            next_tokens = next_tokens.reshape(probs_shape[:-1])
            probs = probs.reshape(probs_shape)

        next_token_scores = probs
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        next_token_scores = nn.functional.softmax(next_token_scores, dim=-1)

    # finished sentences should have their next token be a padding token
    if has_eos_stopping_criteria:
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
    
    return next_tokens, next_token_scores

class SpeculativeSampler:

    def __init__(
        self, 
        collected_draft_logits=[], 
        collected_advanced_logits=[], 
        max_num_collected_logits=2,
        generator=None,
        draft_type = 'jacobian_states',
        reject_sampling_relative_ids = None,
        reject_sampling_draft_token_logits = None,
        sampling_last_draft_token = None,
    ):

        self.max_num_collected_logits = max_num_collected_logits
        self.collected_draft_logits = collected_draft_logits
        self.collected_advanced_logits = collected_advanced_logits

        self.draft_token_index_selector = lambda x: x
        if draft_type == 'jacobian_states':
            # for jacobi iteration (predict next token)
            self.advanced_token_index_selector = lambda x: x - 1
        else:
            self.advanced_token_index_selector = lambda x: x

        self.generator = generator

        self.image_token_list = [i for i in range(4, 8195 + 1)]

        self.reject_sampling_relative_ids = reject_sampling_relative_ids
        self.reject_sampling_draft_token_logits = reject_sampling_draft_token_logits
        self.sampling_last_draft_token = sampling_last_draft_token

        self._init_reject_sampling_params()
    
    def collect_logits(self, logits, collection_type='draft'):
        if collection_type == 'draft': 
            collected_logits = self.collected_draft_logits
        elif collection_type == 'advanced':
            collected_logits = self.collected_advanced_logits
        else:
            assert False, f"collection_type should be 'draft' or 'advanced', but got {collection_type}"

        if logits is not None:
            collected_logits.append(logits)
        
        if len(collected_logits) > self.max_num_collected_logits:
            return collected_logits.pop(0)
        else:
            return None
    
    def logits_calibrating(self, advanced_prob,):

        calibrated_logits = advanced_prob.log()

        B, L = advanced_prob.shape[:2]
        for b in range(B):
            reject_sampling_relative_index = self.reject_sampling_relative_ids[b]
            reject_sampling_draft_token_logits = self.reject_sampling_draft_token_logits[b]
            if reject_sampling_relative_index >= 0:
                token_advanced_prob = advanced_prob[b, reject_sampling_relative_index]

                calibrated_logits[b, reject_sampling_relative_index] = self.get_reject_sampling_logits(
                    token_advanced_prob, reject_sampling_draft_token_logits)
        
        self._init_reject_sampling_params()

        return calibrated_logits
    
    def get_reject_sampling_logits(self, token_advanced_prob, token_draft_prob):
        pos_delta_logits = (
            token_advanced_prob - token_draft_prob
        ).clamp(min=0).log()
        return pos_delta_logits
    
    def reject_sampling_single_token(
        self, token_advanced_prob, token_draft_prob,
        logits_processor=None, logits_warper=None,
        all_collected_input_ids=None,
    ):

        pos_delta_logits = self.get_reject_sampling_logits(token_advanced_prob, token_draft_prob)
        shape_pos_delta_logits = pos_delta_logits.shape

        if (logits_processor is not None) or (logits_warper is not None):
            while len(all_collected_input_ids.shape) < 2:
                all_collected_input_ids = all_collected_input_ids.unsqueeze(0)
            
            while len(pos_delta_logits.shape) < 3:
                pos_delta_logits = pos_delta_logits.unsqueeze(0)
        
        if logits_processor is not None:
            pos_delta_logits = logits_processor(all_collected_input_ids, pos_delta_logits)
        
        if logits_warper is not None:
            pos_delta_logits = logits_warper(all_collected_input_ids, pos_delta_logits)

        pos_delta_logits = pos_delta_logits.view(shape_pos_delta_logits)
        probs = F.softmax(pos_delta_logits, dim=-1)
        resampled_scores = probs

        probs = probs.unsqueeze(0) if len(probs.shape) <= 1 else probs

        resampled_tokens = torch.multinomial(
            probs, num_samples=1, #len(probs.shape)-1,
            generator=self.generator,
        ).squeeze(-1)
        return resampled_tokens, resampled_scores
        
    def _init_reject_sampling_params(self,):
        self.reject_sampling_relative_ids.fill_(-1)
        self.reject_sampling_draft_token_logits.fill_(0)
    
    def __call__(
        self, draft_tokens, advanced_tokens, draft_prob, advanced_prob,
        logits_processor = None, logits_warper = None,
        all_collected_input_ids = None,
        **kwargs,
    ): 
        # draft_tokens: [B, L], advanced_tokens: [B, L], draft_prob: [B, L, V], advanced_prob: [B, L, V]

        # reinitalize self.reject_sampling_relative_ids
        self._init_reject_sampling_params()

        B, L = draft_tokens.shape

        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator)

        draft_token_index_selector = self.draft_token_index_selector
        advanced_token_index_selector = self.advanced_token_index_selector

        resampled_target_tokens = advanced_tokens.clone()
        resampled_target_scores = advanced_prob.clone()

        first_misaligned_token_inds = []
        for b in range(B):
            first_misaligned_token_index = L # keep at least one token left
            for i in range(1, L):

                draft_token_index = draft_token_index_selector(i)
                target_token_index = advanced_token_index_selector(i)

                cls_idx = draft_tokens[b, draft_token_index]

                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]

                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]

                r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                if r < (sampled_advanced_prob / sampled_draft_prob).clamp(max=1): 
                    # accept sampling
                    resampled_target_tokens[b, target_token_index] = cls_idx
                    resampled_target_scores[b, target_token_index, :] = draft_prob[b, draft_token_index, :]
                else:

                    first_misaligned_token_index = i
                    self.reject_sampling_relative_ids[b] = 0
                    self.reject_sampling_draft_token_logits[b] = draft_prob[b, draft_token_index]

                    # we perform reject sampling in the backbone model's prediction loop out of the this sampler
                    resampled_tokens, resampled_scores = self.reject_sampling_single_token(
                        token_advanced_prob = advanced_prob[b, target_token_index, :], 
                        token_draft_prob = draft_prob[b, draft_token_index, :],
                        logits_processor = logits_processor,
                        logits_warper = logits_warper,
                        all_collected_input_ids = torch.cat([
                            all_collected_input_ids[b, :],
                            resampled_target_tokens[b, :target_token_index],
                        ], dim=-1),
                    )
                    resampled_target_tokens[b, target_token_index] = resampled_tokens
                    # resampled_target_scores[b, target_token_index, :] = resampled_scores # the score is kept, so not to update this.
                    first_misaligned_token_index = i

                    break
        
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

def find_first_misaligned_token_inds(
    input_ids, next_tokens,
):
    # input_ids: [B, L], next_tokens: [B, L]
    first_misaligned_token_inds = []
    for b in range(input_ids.shape[0]):
        first_misaligned_token_index = input_ids.shape[1] #- 1 # keep at least one token left
        for i in range(1, input_ids.shape[1]):
            if input_ids[b, i] == next_tokens[b, i-1]:
                pass
            else:
                first_misaligned_token_index = i
                break
    
        first_misaligned_token_inds.append(first_misaligned_token_index)
    
    return first_misaligned_token_inds

def prefix_matching_next_tokens(
    model_input_ids, next_tokens, next_token_scores,
    is_prefilling_phase=False,
    input_token_scores = None,
    prefix_token_sampler=None,
    **kwargs,
):
    # all_collected_input_ids should only include the first element of model_inputs['input_ids']

    if is_prefilling_phase:
        matched_num = model_input_ids.shape[1]
        matched_next_tokens = next_tokens[:, -1:]
        unmatched_next_tokens = next_tokens[:, next_tokens.shape[1]:]

        matched_next_scores = next_token_scores[:, -1:]
        unmatched_next_scores = next_token_scores[:, next_token_scores.shape[1]:]
    else:

        if prefix_token_sampler is not None:
            # SpeculativeSampler.__call__
            first_misaligned_input_token_inds, next_tokens, next_token_scores = prefix_token_sampler(
                draft_tokens = model_input_ids,
                advanced_tokens = next_tokens,
                draft_prob = input_token_scores,
                advanced_prob = next_token_scores,
                **kwargs,
            )
            min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)
        else:
            first_misaligned_input_token_inds = find_first_misaligned_token_inds(
                model_input_ids, next_tokens,
            )
            min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)

        matched_num = min_first_misaligned_input_token_index
        matched_next_tokens = next_tokens[:, :matched_num]
        unmatched_next_tokens = next_tokens[:, matched_num:]

        matched_next_scores = next_token_scores[:, :matched_num]
        unmatched_next_scores = next_token_scores[:, matched_num:]

    return matched_num, matched_next_tokens, unmatched_next_tokens, matched_next_scores, unmatched_next_scores

def push_forward_model_kwargs_and_inputs(
    model_kwargs,
    all_collected_input_ids, 
    model_input_ids, output_token_num,
    num_matched_tokens, matched_next_tokens, unmatched_next_tokens,
    temporary_collected_scores=None,
    matched_next_scores=None,
    unmatched_next_scores=None,
):

    updated_input_ids = torch.cat([all_collected_input_ids, matched_next_tokens], dim=-1)
    if temporary_collected_scores is not None:
        temporary_collected_scores = torch.cat([
            temporary_collected_scores[:, -1:], 
            matched_next_scores,
        ], dim=-2)
    
    past_key_values = model_kwargs["past_key_values"]
    attention_mask = model_kwargs["attention_mask"]
    cache_position = model_kwargs["cache_position"]

    new_model_inputs = None

    seq_len = cache_position.shape[-1]
    remaining_tokens_num = seq_len - num_matched_tokens
    if remaining_tokens_num > 0:
        # roll back

        delete_false_key_value(past_key_values, remaining_tokens_num)

        attention_mask = attention_mask[..., :-remaining_tokens_num, :-remaining_tokens_num]
        cache_position = cache_position[..., :-remaining_tokens_num]

        new_model_inputs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        }

        model_kwargs.update(new_model_inputs)

        additional_tokens = unmatched_next_tokens
        additional_scores = unmatched_next_scores

        nonoverlap_output_token_num = output_token_num 
    else:

        # attention_mask is all useful
        nonoverlap_output_token_num = output_token_num
        additional_tokens = None
        additional_scores = None
    
    return model_kwargs, updated_input_ids, nonoverlap_output_token_num, additional_tokens, additional_scores, temporary_collected_scores

def renew_pipeline(model_class):
    class ZipARPipeline(model_class):

        def _init_new_params(self, guidance_scale=3.0, image_top_k=2000, text_top_k=10, **kwargs):
            self.cfg = guidance_scale
            self.image_top_k = image_top_k
            self.text_top_k = text_top_k

    def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10):
        logits_processor = LogitsProcessorList()

        cfg_processor = LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(
            guidance_scale=cfg,
            model=self.model,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
        )

        candidate_processor = MultiModalLogitsProcessor_ZIPAR(
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
            voc_size=self.model.config.vocab_size,
        )

        topk_processor = InterleavedTopKLogitsWarper(
            image_top_k=image_top_k,
            text_top_k=text_top_k,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        )

        # logits_processor.append(cfg_processor)  ## this is slow! we use batchify cfg
        logits_processor.append(candidate_processor)
        logits_processor.append(topk_processor)

        return logits_processor
    
    return ZipARPipeline

def get_multi_token_for_preparation(
    img_vocab, 
    rand_token_num, 
    input_ids, temporary_collected_scores, device, 
    multi_token_init_scheme=None,
    last_input_tokens=None, last_input_scores=None,
    generator = None,
    extrapolation_guidance_weight = 1.5,
    eps = 1e-7,
    prefill_num = 3,
    additional_tokens_len = 0,
    **kwargs,
):

    def random_multinomial_sample_from_logits(rand_logits):
        logits = rand_logits
        probs_shape = None
        if len(logits.shape) >= 3:
            probs_shape = logits.shape
            logits = logits.flatten(0, len(probs_shape)-2)

        topk_logits, topk_cls_indices = torch.topk(logits, k=1, dim=-1)
        logits = torch.full_like(logits, -float("Inf")).scatter(-1, topk_cls_indices, topk_logits)
        probs = F.softmax(logits, dim=-1)

        rand_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        if probs_shape is not None:
            rand_tokens = rand_tokens.reshape(probs_shape[:-1])
            probs = probs.reshape(probs_shape)
        
        return rand_tokens, probs

    if multi_token_init_scheme == 'random':
        img_vocab = img_vocab.to(device)
        img_vocab_size = len(img_vocab)
        rand_tokens = torch.randint(
            0, img_vocab_size, 
            (*input_ids.shape[:-1], rand_token_num)
        ).to(device)
        rand_tokens = img_vocab[rand_tokens]

        scores_of_rand_tokens = temporary_collected_scores.new_zeros(
            (*temporary_collected_scores.shape[:-2], rand_token_num, temporary_collected_scores.shape[-1])
        )
        scores_of_rand_tokens = torch.scatter(scores_of_rand_tokens, -1, rand_tokens.unsqueeze(-1), 1.0)
    
    else:
        img_vocab = img_vocab.to(device)
        img_vocab_size = len(img_vocab)
        rand_tokens = torch.randint(
            0, img_vocab_size, 
            (*input_ids.shape[:-1], rand_token_num)
        ).to(device)
        rand_tokens = img_vocab[rand_tokens]

        scores_of_rand_tokens = temporary_collected_scores.new_zeros(
            (*temporary_collected_scores.shape[:-2], rand_token_num, temporary_collected_scores.shape[-1])
        )
        scores_of_rand_tokens = torch.scatter(scores_of_rand_tokens, -1, rand_tokens.unsqueeze(-1), 1.0)


        img_width = kwargs.get("img_width", None)
        pad_len = 1
        img_width = img_width + pad_len if img_width is not None else 0
        input_ids_len = input_ids.shape[1]
        
        prefill_num = prefill_num + 3 # TODO: this is for mgpt, the first 3 tokens <start, h, w>
        if (img_width > 0) and (input_ids_len + additional_tokens_len >= prefill_num) and (rand_token_num > 0):
            
            horizon_indices = (torch.arange(
                input_ids_len + additional_tokens_len, 
                input_ids_len + additional_tokens_len + rand_token_num, 
                device=device, dtype=torch.long
            ) - prefill_num) % img_width
            vertical_indices = (torch.arange(
                input_ids_len + additional_tokens_len, 
                input_ids_len + additional_tokens_len + rand_token_num, 
                device=device, dtype=torch.long
            ) - prefill_num) // img_width

            if 'horizon' in multi_token_init_scheme:
                valid_indices = (horizon_indices - 1 >= 0)
                last_vertical_indices = vertical_indices
                last_horizon_indices = horizon_indices - 1
            else:
                # # vertical consumes more memory to store the previous logits, so we use horizon in practice
                # if 'vertical' in multi_token_init_scheme:
                #     valid_indices = (vertical_indices - 1 >= 0)
                #     last_vertical_indices = vertical_indices - 1
                #     last_horizon_indices = horizon_indices
                assert False, f"multi_token_init_scheme should be 'horizon' or 'vertical', but got {multi_token_init_scheme}"
            
            last_input_tokens = torch.cat(
                [input_ids, last_input_tokens], dim=1
            ) if last_input_tokens is not None else input_ids
            last_input_scores = torch.cat(
                [temporary_collected_scores, last_input_scores], dim=1
            ) if last_input_scores is not None else temporary_collected_scores
            last_input_logits = (last_input_scores.float() + eps).log()

            last_flatten_indices = last_vertical_indices[valid_indices] * img_width + last_horizon_indices[valid_indices] + prefill_num
            
            # e.g., last indices [100, 101, 102], but the current indices up to 100, 
            # and 101, 102 depends on the values from 100 (but 100 has not been appended to input-ids yet)
            last_flatten_indices = last_flatten_indices.clamp(min=0, max = last_input_tokens.shape[1]-1) 
            
            last_resampled_input_tokens = last_input_tokens[:, last_flatten_indices]
            last_resampled_input_logits = last_input_logits[:, last_flatten_indices]

            if 'sample' in multi_token_init_scheme:
                resampled_rand_tokens, resampled_scores_of_rand_tokens = random_multinomial_sample_from_logits(
                    last_resampled_input_logits
                )
                scores_of_rand_tokens[:, valid_indices] = 0
                scores_of_rand_tokens[:, valid_indices] = torch.scatter(
                    scores_of_rand_tokens[:, valid_indices], -1, resampled_rand_tokens.unsqueeze(-1), 1.0)
            elif 'repeat' in multi_token_init_scheme:
                resampled_rand_tokens = last_resampled_input_tokens
                scores_of_rand_tokens[:, valid_indices] = 0
                scores_of_rand_tokens[:, valid_indices] = torch.scatter(
                    scores_of_rand_tokens[:, valid_indices], -1, resampled_rand_tokens.unsqueeze(-1), 1.0)
            else:
                assert False, f"multi_token_init_scheme should be 'sample' or 'repeat', but got {multi_token_init_scheme}"
            
            rand_tokens[:, valid_indices] = resampled_rand_tokens
    
    return rand_tokens, scores_of_rand_tokens

def renew_sampler(model_class):
    #重写ChameleonForConditionalGeneration
    class ZipARSampler(model_class, nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_new_params()

        def _init_new_params_(self,cfg=4.0,zipar_window_size=16):
            self.cfg = cfg
            self.zipar_window_size = zipar_window_size
            self.prompt_len = 0
            self.zipar_window_size = 16
            self.tokens_per_row = 49
            self.eol_token = torch.tensor([8803], device='cuda')
            self.eoi_token = torch.tensor([8196], device='cuda')
            self.end_token = torch.tensor([8710], device='cuda')
            self.pad_token = torch.tensor([1], device='cuda')
            self.ongoing_row_list = []
            self.row_token_num = torch.zeros((self.tokens_per_row-1,), dtype=torch.long)
        
        def prepare_ZIPAR_inputs_for_generation(
            self,
            input_ids,
            pixel_values=None,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            **kwargs,
        ):
            attention_mask = torch.ones((1, input_ids.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)

            if past_key_values is not None:
                new_input_ids = []
                new_position_ids = [] ## the global index of tokens
                local_position_ids = [] ## the index in input_ids
                if kwargs['last_token_in_row_idx'] is not None:
                    idx_in_input_ids = kwargs['last_token_in_row_idx']
                    assert idx_in_input_ids == self.prompt_len + 2 + torch.sum(self.row_token_num[:self.ongoing_row_list[0]], dim=0) - 1
                    global_idx = self.prompt_len + 2 + (self.ongoing_row_list[0]-1) * self.tokens_per_row + self.row_token_num[(self.ongoing_row_list[0]-1)] - 1
                    new_input_ids.append(input_ids[:, idx_in_input_ids].unsqueeze(-1))
                    new_position_ids.append(global_idx)
                    local_position_ids.append(idx_in_input_ids)

                for i in range(len(self.ongoing_row_list)):
                    idx_in_input_ids = self.prompt_len + 2 + torch.sum(self.row_token_num[:(self.ongoing_row_list[i] + 1)], dim=0) - 1
                    global_idx = self.prompt_len + 2 + self.ongoing_row_list[i] * self.tokens_per_row + self.row_token_num[(self.ongoing_row_list[i])] - 1
                    new_input_ids.append(input_ids[:, idx_in_input_ids].unsqueeze(-1))
                    new_position_ids.append(global_idx)
                    local_position_ids.append(idx_in_input_ids)
                # input_ids = torch.tensor(new_input_ids, device=input_ids.device).unsqueeze_(0)
                input_ids = torch.cat(new_input_ids, dim=1)
                position_ids = torch.tensor(new_position_ids, device=input_ids.device).unsqueeze_(0)
                local_position_ids = torch.tensor(local_position_ids, device=input_ids.device).unsqueeze_(0)

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

            if cache_position[0] == 0:
                # If we're in cached decoding stage, pixel values should be `None` because input ids do not contain special image token anymore
                # Otherwise we need pixel values to be passed to model
                model_inputs["pixel_values"] = pixel_values

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                    "local_position_ids": local_position_ids
                }
            )
            return model_inputs
        
        def position_insert(self, input_ids, next_tokens, position):
            return torch.cat((input_ids[:, :position], next_tokens[:, None], input_ids[:, position:]), dim=1)

        @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            local_position_ids: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            r"""
            Args:
                labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                    (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            Returns:

            Example:

            ```python
            >>> from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
            >>> import torch
            >>> import requests
            >>> from PIL import Image

            >>> model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16)
            >>> processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

            >>> prompt = "I used to know a lot about constellations when I was younger, but as I grew older, I forgot most of what I knew. These are the only two constellations that I really remember now.<image><image>I would like for you to tell me about 3 more constellations and give me a little bit of history about the constellation."
            >>> image = Image.open(requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw)
            >>> image_2 = Image.open(requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw)

            >>> inputs = processor(prompt, images=[image, image_2], return_tensors="pt").to(model.device, torch.bfloat16)

            >>> generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            >>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            ```"""
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                local_position_ids=local_position_ids
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            if self.config.mask_image_logits:
                # Disallow image tokens which does not include special begin-image and end-image tokens
                image_tokens = self.model.vocabulary_mapping.image_tokens
                logits[:, :, image_tokens] = torch.finfo(logits.dtype).min

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def _sample(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus,
            streamer,
            **model_kwargs,
        ):
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                generation_config ([`~generation.GenerationConfig`]):
                    The generation configuration to be used as parametrization of the decoding method.
                synced_gpus (`bool`):
                    Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                streamer (`BaseStreamer`, *optional*):
                    Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                    through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            """
            # init values
            pad_token_id = generation_config._pad_token_tensor
            output_attentions = generation_config.output_attentions
            output_hidden_states = generation_config.output_hidden_states
            output_scores = generation_config.output_scores
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
            max_length = generation_config.max_length
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
            model_kwargs['last_token_in_row_idx'] = None ## special case, we need to input last token in row to model to generate KV cache
            first_run = True

            while self._has_unfinished_sequences(
                this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
            ):
                num_image_start_tokens = (input_ids[0] == 8197).sum()
                if num_image_start_tokens >= 1:
                    image_start_token_id_index = torch.where(input_ids[0] == 8197)[0][-1].item()
                    image_token_num = len(input_ids[0][image_start_token_id_index + 1 :])
                else:
                    image_token_num = 0

                if image_token_num == 2:
                    self.ongoing_row_list.append(0)
                    self.row_token_num[0] += 1

                # prepare model inputs
                if image_token_num > 2:
                    model_inputs = self.prepare_ZIPAR_inputs_for_generation(input_ids, **model_kwargs)
                else:
                    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                num_new_tokens = model_inputs['input_ids'].shape[1] if len(self.ongoing_row_list) > 0 else 1
                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                if self.cfg > 1.0 and first_run:
                    first_run = False
                    uncond_input_ids = torch.ones_like(model_inputs['input_ids'])
                    uncond_input_ids[:,0] = 0
                    model_inputs['input_ids'] = torch.cat([model_inputs['input_ids'], uncond_input_ids], dim=0)
                elif self.cfg > 1.0:
                    model_inputs['input_ids'] = model_inputs['input_ids'].repeat(2,1) ## repeat along the batch dim

                # forward pass to get next token
                # torch.cuda.synchronize()
                # start = time.perf_counter()
                outputs = self(**model_inputs, return_dict=True)
                # torch.cuda.synchronize()
                # end = time.perf_counter()
                # print('Inference time: {:.2f}'.format((end-start)*1000), 'ms')

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                if len(self.ongoing_row_list) == 0:
                    next_token_logits = outputs.logits.clone()[:, -1, :].float()

                    if image_token_num > 1 and self.cfg > 1.0:
                        cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                        next_token_logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg
                    elif self.cfg > 1.0:
                        next_token_logits, _ = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)

                    # pre-process distribution
                    next_token_scores = logits_processor(input_ids, next_token_logits)

                    # Store scores, attentions and hidden_states when required
                    if return_dict_in_generate:
                        if output_scores:
                            scores += (next_token_scores,)
                        if output_logits:
                            raw_logits += (next_token_logits,)
                        if output_attentions:
                            decoder_attentions += (
                                (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                            )
                            if self.config.is_encoder_decoder:
                                cross_attentions += (outputs.cross_attentions,)

                        if output_hidden_states:
                            decoder_hidden_states += (
                                (outputs.decoder_hidden_states,)
                                if self.config.is_encoder_decoder
                                else (outputs.hidden_states,)
                            )

                    # token selection
                    if do_sample:
                        probs = nn.functional.softmax(next_token_scores, dim=-1)
                        # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        next_tokens = torch.argmax(next_token_scores, dim=-1)

                    # finished sentences should have their next token be a padding token
                    if has_eos_stopping_criteria:
                        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                    # update generated ids, model inputs, and length for next step
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    if streamer is not None:
                        streamer.put(next_tokens.cpu())                
                else:
                    need_remove_row = []
                    if model_kwargs['last_token_in_row_idx'] is not None:
                        num_skip_tokens = 1
                        model_kwargs['last_token_in_row_idx'] = None
                    else:
                        num_skip_tokens = 0
                    for i in range(0, num_new_tokens-num_skip_tokens):
                        next_token_logits = outputs.logits.clone()[:, i+num_skip_tokens, :].float() ## deal with tokens one by one

                        if image_token_num > 1 and self.cfg > 1.0:
                            cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                            next_token_logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg
                        elif self.cfg > 1.0:
                            next_token_logits, _ = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)

                        # pre-process distribution
                        next_token_scores = logits_processor(input_ids, next_token_logits)

                        # Store scores, attentions and hidden_states when required
                        if return_dict_in_generate:
                            if output_scores:
                                scores += (next_token_scores,)
                            if output_logits:
                                raw_logits += (next_token_logits,)
                            if output_attentions:
                                decoder_attentions += (
                                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                                )
                                if self.config.is_encoder_decoder:
                                    cross_attentions += (outputs.cross_attentions,)

                            if output_hidden_states:
                                decoder_hidden_states += (
                                    (outputs.decoder_hidden_states,)
                                    if self.config.is_encoder_decoder
                                    else (outputs.hidden_states,)
                                )

                        # token selection
                        if do_sample:
                            probs = nn.functional.softmax(next_token_scores, dim=-1)
                            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                        else:
                            next_tokens = torch.argmax(next_token_scores, dim=-1)

                        # finished sentences should have their next token be a padding token
                        if has_eos_stopping_criteria:
                            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                        # concat input_ids with next_tokens accordingly
                        position = torch.sum(self.row_token_num[:(self.ongoing_row_list[i] + 1)], dim=0) + self.prompt_len + 2 ## this is position in input_ids
                        input_ids = self.position_insert(input_ids, next_tokens, position) ## input_ids[:, position] = next_tokens

                        self.row_token_num[self.ongoing_row_list[i]] += 1

                        if self.row_token_num[self.ongoing_row_list[i]] == self.zipar_window_size and self.ongoing_row_list[i] < self.tokens_per_row - 2\
                            and self.ongoing_row_list[i]+1 not in self.ongoing_row_list: ## append a eol token in this line to start next line
                            # if self.zipar_window_size < 40 and self.ongoing_row_list[0] % 3 == 0:
                            #     self.zipar_window_size += 1
                            input_ids = torch.cat([input_ids, self.eol_token[:, None]], dim=1)
                            self.ongoing_row_list.append(self.ongoing_row_list[i]+1)
                            self.row_token_num[self.ongoing_row_list[i]+1] += 1
                        elif self.ongoing_row_list[i] == self.tokens_per_row - 2 and self.row_token_num[self.ongoing_row_list[i]] == self.tokens_per_row:
                            input_ids = torch.cat([input_ids, self.eol_token[:, None]], dim=1)
                            input_ids = torch.cat([input_ids, self.eoi_token[:, None]], dim=1)
                            input_ids = torch.cat([input_ids, self.end_token[:, None]], dim=1)
                            break

                        if self.row_token_num[self.ongoing_row_list[i]] == self.tokens_per_row: ## this row is done
                            model_kwargs['last_token_in_row_idx'] = position
                            need_remove_row.append(self.ongoing_row_list[i])

                    for row in need_remove_row:
                        self.ongoing_row_list.remove(row)
                    need_remove_row.clear()

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    num_new_tokens=num_new_tokens
                )

                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0

                if (input_ids[0] == 8710).sum() >= 2:
                    this_peer_finished = True
                    
                cur_len += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs
            print('Model forward times: ', cur_len-self.prompt_len)
            # torch.save(input_ids.cpu(), '/mnt/workspace/Lumina-mGPT/generated_ids.pth')

            if streamer is not None:
                streamer.end()

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                return input_ids
    
    return ZipARSampler

def renew_backbone(model_class):
    class ZipARBackbone(model_class):

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            local_position_ids: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, BaseModelOutputWithPast]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if self.gradient_checkpointing and self.training and use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if pixel_values is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
                )

            if pixel_values is not None:
                image_tokens = self.get_image_tokens(pixel_values)
                special_image_mask = input_ids == self.vocabulary_mapping.image_token_id
                image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
                input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            if cache_position[0] != 0 and input_ids.shape[1] != 1: ## using ZIPAR
                causal_mask = self._update_causal_mask_ZIPAR(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions, local_position_ids
                )
            else:
                causal_mask = self._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
                )

            # embed positions
            hidden_states = inputs_embeds

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        local_position_ids=local_position_ids
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = None
            if use_cache:
                next_cache = next_decoder_cache

            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        def _update_causal_mask_ZIPAR(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
            local_position_ids
        ):
            # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
            # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
            # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
            # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_static_cache = isinstance(past_key_values, StaticCache)

            # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
            if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            if using_static_cache:
                target_length = past_key_values.get_max_length()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            if attention_mask is not None and attention_mask.dim() == 4:
                # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
                if attention_mask.max() != 0:
                    raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
                causal_mask = attention_mask
            else:
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
                indices = torch.arange(target_length, device=causal_mask.device).unsqueeze(0).expand(sequence_length, -1)
                before_mask = indices <= local_position_ids.squeeze_(0).unsqueeze_(1)
                causal_mask[before_mask] = 0

                causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            
            return causal_mask
    return ZipARBackbone

def renew_pipeline_sampler(pipe_line, **kwargs):
    pipe_line.__class__ = renew_pipeline(pipe_line.__class__)#FlexARInferenceSolver
    pipe_line._init_new_params(**kwargs)
    pipe_line.model.__class__ = renew_sampler(pipe_line.model.__class__)#ChameleonForConditionalGeneration
    pipe_line.model._init_new_params_(**kwargs)
    pipe_line.model.model.__class__ = renew_backbone(pipe_line.model.model.__class__)#ChameleonModel
    # sdpa attention暂时还有问题，先算了吧，到时候再优化代码
    # pipe_line.model.model.
    return pipe_line