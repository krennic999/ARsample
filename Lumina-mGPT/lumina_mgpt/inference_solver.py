import argparse
import copy
import math
from typing import List, Optional, Union

from PIL import Image
import time
import torch
import transformers
from transformers import GenerationConfig, TextStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LogitsWarper
import pdb

from data.item_processor import FlexARItemProcessor
from model.chameleon import ChameleonForConditionalGeneration
import torch.nn.functional as F


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
        enable_entropy_filtering,
        model,
        image_start_token_id,#这应该是加的boi
        image_end_token_id,#这是eoi
        image_next_line_token_id,#这是end of line
        patch_size,
        unconditional_ids: Optional[torch.LongTensor] = None,#下面这两个参数都没有用到
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,#不是kv cache，就是把之前的input_ids存到context里
    ):
        self.guidance_scale = guidance_scale
        self.enable_entropy_filtering=enable_entropy_filtering
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
        self.image_next_line_token_id = image_next_line_token_id#这个processor里没有用到关于line的部分
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

    def get_unconditional_logits(self, input_ids, image_start_token_id_index):
        # 这个函数看起来是为了更新context，并调用模型预测logit的
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                # 原本是下面这样
                # self.unconditional_context["input_ids"] = input_ids[:, -1:]
                # input_ids[:, -1:]是当前的token
                self.unconditional_context["input_ids"] = input_ids[:, image_start_token_id_index:]#从soi开始（包括soi）的所有token
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

        #似乎删掉了关于forward pass和cache的一些代码
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        # input_ids是整个自回归序列中所有token的index（包括文本），BL,input_ids[0]表示第一个batch的seq
        # self.image_start_token_id:8197,image_end_token_id:8196（我记得chameleon的codebook应该是8192的）
        # 计算sequence中包含SOI和EOI的个数
        # 啊所以score是_sample中自己包含的生成结果，在这里又调用get_unconditional_logits生成uncondition的结果？
        num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()#统计seq中idx=8197（start img）的token个数
        num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()#统计seq中idx=8196（end img）的token个数

        # 都等0、都等1时不执行操作，因为在执行文本序列的预测
        if num_image_start_tokens == num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            self.unconditional_context = None
            return scores

        # SOI=EOI+1:在预测图像序列
        elif num_image_start_tokens == num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:#记录下当前图像SOI的位置
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])#有效的image token的个数

            # input_ids[0][self.image_start_token_id_index]:SOI；self.image_start_token_id_index+1和+2是h/w indicator
            # print(f"num new tokens: {new_token_num}")
            if new_token_num >= 2:
                #前两个是h/w indicator（如果是1024的话得到的id是8836）；怎么训练能让两个token分别表示h/w呢？
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    # （8836-8804）*2=64
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2

                if self.unconditional_context is None:
                    # {'input_ids': None, 'attention_mask': None, 'use_cache': True, 'past_key_values': DynamicCache(), 'first_pass': True}
                    self.unconditional_context = copy.deepcopy(self.unconditional_context_backup)

                if self.guidance_scale == 1.0:
                    return scores

                unconditional_logits = self.get_unconditional_logits(input_ids, self.image_start_token_id_index)[:, -1]

                # if self.enable_entropy_filtering:
                #     scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
                scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
                return (scores_processed,scores,unconditional_logits)

        else:
            print("Something wrong in the decoding process.")
        return scores


class MultiModalLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        enable_entropy_filtering,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
    ):
        self.enable_entropy_filtering=enable_entropy_filtering
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


    # # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # def __call__(self, input_ids: torch.LongTensor, scores_: torch.FloatTensor) -> torch.FloatTensor:
    #     if torch.is_tensor(scores_):
    #         scores=scores_
    #     else:
    #         scores,cond_logits,uncond_logits=scores_
    #     # 前置部分的处理和上一个logitprocessor一样，筛选出来有效的image token处理
    #     self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
    #     self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

    #     # print(self.num_image_start_tokens, self.num_image_end_tokens)

    #     if self.num_image_start_tokens == self.num_image_end_tokens:#start和end tokens的count都是0时，即在生成文本或是图像已经生成完
    #         self.h_latent_dim, self.w_latent_dim = None, None
    #         self.image_start_token_id_index = None
    #         return scores

    #     elif self.num_image_start_tokens == self.num_image_end_tokens + 1:#在生成图像
    #         if self.image_start_token_id_index is None:#self.image_start_token_id_index和上一个processor一样
    #             self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
    #             print(self.image_start_token_id_index)
    #             self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()

    #         new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])#生成图像的前2个token是h/w indicator

    #         if new_token_num >= 2:
    #             if self.h_latent_dim is None or self.w_latent_dim is None:
    #                 h_grids, w_grids = (
    #                     input_ids[0][self.image_start_token_id_index + 1] - 8804,
    #                     input_ids[0][self.image_start_token_id_index + 2] - 8804,
    #                 )
    #                 # print(f"h_grids: {h_grids}, w_grids: {w_grids}")
    #                 self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2
    #                 print(f"h_latent_dim: {self.h_latent_dim}, w_latent_dim: {self.w_latent_dim}")

    #             tokens = input_ids[0][self.image_start_token_id_index + 3 :]#拆分出来有效的token
    #             if (len(tokens) + 1) % (self.w_latent_dim + 1) == 0:#预测到整行的最后一个token
    #                 new_line_constrained_scores = torch.full_like(scores, -math.inf)#强制把这个特殊token替换成eol
    #                 new_line_constrained_scores[:, self.image_next_line_token_id] = 0
    #                 #这些token_id=0的操作应该是为了避免特殊token对最终预测产生影响，即从score采样的过程不考虑他们
    #                 print(f"new line: {len(tokens)+1}, time: {time.time()}")
    #                 return new_line_constrained_scores
    #             elif (len(tokens) + 1) == (self.w_latent_dim + 1) * self.h_latent_dim + 1:#预测完整个图像
    #                 eos_image_constrained_scores = torch.full_like(scores, -math.inf)
    #                 eos_image_constrained_scores[:, self.image_end_token_id] = 0
    #                 print(f"eos image: {len(tokens)+1}, time: {time.time()}")
    #                 return eos_image_constrained_scores
    #             elif (len(tokens) + 1) % (self.w_latent_dim + 1) != 0:#把一些channel置成0，不考虑这些channel对应的logits（可能是图文区分开）
    #                 image_constrained_scores = torch.where(self.suppress_token_mask, -float("inf"), scores)
    #                 if self.enable_entropy_filtering:
    #                     probs=F.softmax(image_constrained_scores, dim=-1)
    #                     cur_entropy = - (probs * torch.log(probs + 1e-12)).sum(dim=-1)#b,l
    #                     # print(cur_entropy)
    #                     temperature=3.0*torch.exp(-cur_entropy[:,None]/2.5)+0.3
    #                     temperature[cur_entropy[:,None] > 7] = 0.4
    #                     image_constrained_scores=image_constrained_scores/temperature
    #                 return image_constrained_scores
    #     else:
    #         print("Something wrong in the decoding process.")

    #     return scores


    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores_: torch.FloatTensor) -> torch.FloatTensor:
        if torch.is_tensor(scores_):
            scores=scores_
        else:
            scores,cond_logits,uncond_logits=scores_
        # 前置部分的处理和上一个logitprocessor一样，筛选出来有效的image token处理
        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # print(self.num_image_start_tokens, self.num_image_end_tokens)

        if self.num_image_start_tokens == self.num_image_end_tokens:#start和end tokens的count都是0时，即在生成文本或是图像已经生成完
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            return scores

        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:#在生成图像
            if self.image_start_token_id_index is None:#self.image_start_token_id_index和上一个processor一样
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
                print(self.image_start_token_id_index)
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()

            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])#生成图像的前2个token是h/w indicator

            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    # print(f"h_grids: {h_grids}, w_grids: {w_grids}")
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2
                    print(f"h_latent_dim: {self.h_latent_dim}, w_latent_dim: {self.w_latent_dim}")

                tokens = input_ids[0][self.image_start_token_id_index + 3 :]#拆分出来有效的token
                if (len(tokens) + 1) % (self.w_latent_dim + 1) == 0:#预测到整行的最后一个token
                    new_line_constrained_scores = torch.full_like(scores, -math.inf)#强制把这个特殊token替换成eol
                    new_line_constrained_scores[:, self.image_next_line_token_id] = 0
                    #这些token_id=0的操作应该是为了避免特殊token对最终预测产生影响，即从score采样的过程不考虑他们
                    print(f"new line: {len(tokens)+1}, time: {time.time()}")
                    return new_line_constrained_scores
                elif (len(tokens) + 1) == (self.w_latent_dim + 1) * self.h_latent_dim + 1:#预测完整个图像
                    eos_image_constrained_scores = torch.full_like(scores, -math.inf)
                    eos_image_constrained_scores[:, self.image_end_token_id] = 0
                    print(f"eos image: {len(tokens)+1}, time: {time.time()}")
                    return eos_image_constrained_scores
                elif (len(tokens) + 1) % (self.w_latent_dim + 1) != 0:#把一些channel置成0，不考虑这些channel对应的logits（可能是图文区分开）
                    image_constrained_scores = torch.where(self.suppress_token_mask, -float("inf"), scores)
                    if self.enable_entropy_filtering:
                        probs=F.softmax(image_constrained_scores, dim=-1)
                        cur_entropy = - (probs * torch.log(probs + 1e-12)).sum(dim=-1)#b,l
                        # print(cur_entropy)
                        # temperature=3.0*torch.exp(-cur_entropy[:,None]/3)+0.3
                        temperature_cfg=torch.ones_like(cur_entropy[:,None])
                        
                        # entropy0~1.5之间的改cfg
                        mask = ((cur_entropy >= 0) & (cur_entropy < 2)).unsqueeze(-1)#左闭右开
                        temperature_cfg[mask] = 1.5
                        # log_p_uncond = F.log_softmax(uncond_logits, dim=-1)
                        # log_p_cond   = F.log_softmax(cond_logits, dim=-1)

                        image_constrained_scores_new=torch.where(self.suppress_token_mask, -float("inf"), 4.0/temperature_cfg * (cond_logits - uncond_logits) + uncond_logits)
                        
                        # entropy其他区间的改temperature
                        # temperature=(3.5*torch.exp(-cur_entropy[:,None]/2.5)+0.6)#.clamp_(0.6,2)#fid71.88->73.62;clip0.2619->0.2630
                        # temperature=(2.0*torch.exp(-cur_entropy[:,None]/2.5)+0.6)#.clamp_(0.6,2)#fid71.88->69.98;clip0.2619->0.2641
                        temperature=(1.5*torch.exp(-cur_entropy[:,None]/2.5)+0.7).clamp_(0.7,2)#fid71.88->68.85;clip0.2619->0.2643
                        # temperature[cur_entropy[:,None] > 7] = 0.4
                        mask_=mask.expand(-1, 65536)
                        mask_=torch.zeros_like(mask_)
                        # image_constrained_scores=torch.where(mask_,image_constrained_scores_new,image_constrained_scores/temperature)
                        image_constrained_scores=image_constrained_scores/temperature
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

        if self.num_image_start_tokens == self.num_image_end_tokens + 1:#生成文本用text_topk，图像用image_topk
            top_k = min(self.image_top_k, scores.size(-1))
        else:
            top_k = min(self.text_top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class FlexARInferenceSolver:
    @classmethod
    def get_args_parser(cls):
        parser = argparse.ArgumentParser("xllmx Inference", add_help=False)
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "tf32"], default="bf16")

        return parser

    def __init__(self, model_path, precision, target_size=512):
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        self.model = ChameleonForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="cuda",
        )
        self.item_processor = FlexARItemProcessor(tokenizer=model_path,target_size=target_size)

    def get_streamer(self):
        return TextStreamer(self.item_processor.tokenizer)

    @torch.no_grad()
    def generate(
        self,
        images: Image.Image | str | List[Union[Image.Image, str]],
        qas,
        max_gen_len,
        temperature,
        logits_processor=None,
        streamer=None,
    ):

        conversations = []
        for q, a in qas:
            conversations.append(
                {
                    "from": "human",
                    "value": q,
                }
            )
            conversations.append(
                {
                    "from": "gpt",
                    "value": a,
                }
            )
        item = {"image": images, "conversations": conversations}

        _prompt = self.item_processor.process_item(item)
        prompt = []
        for value in _prompt:
            if isinstance(value, int):
                prompt.append(value)
            else:
                prompt += value["input_ids"]
        prompt_len = len(prompt)
        prompt = torch.tensor(prompt, dtype=torch.int64, device=self.model.device).unsqueeze(0)

        generation_config = GenerationConfig(
            max_new_tokens=max_gen_len,
            max_length=self.model.config.max_position_embeddings,
            temperature=temperature,
            top_k=None,
            do_sample=True,
            eos_token_id=[8710],
        )

        if logits_processor is None:
            logits_processor = self.create_logits_processor()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            generation_result = self.model.generate(
                prompt, generation_config, logits_processor=logits_processor, streamer=streamer
            )[0][prompt_len:].tolist()
            if len(generation_result) > 0 and generation_result[-1] == 8710:
                generation_result = generation_result[:-1]

        return self.decode_ids(generation_result)

    def decode_ids(self, tokens: List[int]):
        generated_images = []
        generation_result_processed = []
        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            if token_id == self.item_processor.token2id(self.item_processor.image_start_token):#image_start_token=8197
                cache = []
                for j in range(i + 1, len(tokens)):
                    if tokens[j] != self.item_processor.token2id(self.item_processor.image_end_token):#image_end_token=8196
                        cache.append(tokens[j])
                        i = j + 1
                    else:
                        image = self.decode_image(cache)
                        generated_images.append(image)
                        generation_result_processed.append(self.item_processor.token2id("<|image|>"))
                        i = j + 1
                        break
            else:
                generation_result_processed.append(token_id)
                i += 1

        generated = self.item_processor.tokenizer.decode(generation_result_processed)

        return generated, generated_images

    def decode_image(self, tokens: List[int]):
        return self.item_processor.decode_image(tokens)

    @staticmethod
    def create_image_grid(images, rows, cols):
        width, height = images[0].size

        grid_img = Image.new("RGB", (cols * width, rows * height))

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid_img.paste(img, (col * width, row * height))

        return grid_img

    def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10, enable_entropy_filtering=False):
        logits_processor = LogitsProcessorList()

        cfg_processor = LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(
            enable_entropy_filtering=enable_entropy_filtering,
            guidance_scale=cfg,
            model=self.model,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
        )

        candidate_processor = MultiModalLogitsProcessor(
            enable_entropy_filtering=enable_entropy_filtering,
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

        logits_processor.append(cfg_processor)
        logits_processor.append(candidate_processor)
        logits_processor.append(topk_processor)

        return logits_processor


if __name__ == "__main__":
    parser = FlexARInferenceSolver.get_args_parser()
    args = parser.parse_args()
    solver = FlexARInferenceSolver(**vars(args))
