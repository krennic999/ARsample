import torch.nn as nn
import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers import GenerationConfig
from typing import Optional, Tuple, Union, List, Dict, Sequence, Any
from transformers.pytorch_utils import isin_mps_friendly
from transformers.generation.utils import (
    GenerateNonBeamOutput, 
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)

def renew_sampler(model_class):
    class RevisedCFGSampler(model_class,nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_new_params()
            
        def _init_new_params_(self,cfg=4.0):
            self.cfg=cfg

        def _prepare_attention_mask_for_generation(
            self,
            inputs: torch.Tensor,
            pad_token_id: Optional[torch.Tensor],
            eos_token_id: Optional[torch.Tensor],
        ) -> torch.LongTensor:
            pad_token_id=torch.tensor(0,device=pad_token_id.device)
            # No information for attention mask inference -> return default attention mask
            default_attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
            if pad_token_id is None:
                return default_attention_mask

            is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
            if not is_input_ids:
                return default_attention_mask

            is_pad_token_in_inputs = (pad_token_id is not None) and (
                isin_mps_friendly(elements=inputs, test_elements=pad_token_id).any()
            )
            is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or ~(
                isin_mps_friendly(elements=eos_token_id, test_elements=pad_token_id).any()
            )
            can_infer_attention_mask = is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
            attention_mask_from_padding = inputs.ne(pad_token_id).long()

            attention_mask = (
                attention_mask_from_padding * can_infer_attention_mask + default_attention_mask * ~can_infer_attention_mask
            )
            # attention_mask[:,0]=1
            if self.cfg>1.0:
                uncond_input_mask = torch.ones_like(attention_mask)
                attention_mask = torch.cat([attention_mask,uncond_input_mask],dim=0)
                
            # 这个加上感觉对性能提升也没啥用，是为了避免infer的时候pos的gap
            # for row in attention_mask:
            #     idx = (row == 1).nonzero(as_tuple=True)[0]
            #     if len(idx) > 0 and idx[0] > 0:
            #         row[idx[0] - 1] = 1
            return attention_mask

        def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
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
                    Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                    `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
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

            first_run = True
            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

            while self._has_unfinished_sequences(
                this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
            ):
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
                
                if self.cfg > 1.0 and first_run:
                    first_run = False
                    uncond_input_ids = torch.ones_like(model_inputs['input_ids'])#给全0的input id作为条件
                    uncond_input_ids[:,0] = 0
                    model_inputs['input_ids'] = torch.cat([model_inputs['input_ids'], uncond_input_ids], dim=0)
                    # uncond_input_mask = torch.zeros_like(model_inputs['attention_mask'])
                    # model_inputs['attention_mask'] = torch.cat([model_inputs['attention_mask'],uncond_input_mask],dim=0)
                elif self.cfg > 1.0:
                    model_inputs['input_ids'] = model_inputs['input_ids'].repeat(2,1) ## repeat along the batch dim
                    
                # forward pass to get next token
                outputs = self(**model_inputs, return_dict=True)

                # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )
                if synced_gpus and this_peer_finished:
                    continue

                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits.clone()[:, -1, :].float()
                next_token_logits = next_token_logits.to(input_ids.device)

                if self.cfg > 1.0:
                    cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                    next_token_logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg

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

                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0
                cur_len += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

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
    return RevisedCFGSampler

def renew_pipeline_sampler(pipe_line, **kwargs):
    # pipe_line.__class__ = renew_pipeline(pipe_line.__class__)#FlexARInferenceSolver
    # pipe_line._init_new_params(**kwargs)
    pipe_line.model.__class__ = renew_sampler(pipe_line.model.__class__)#ChameleonForConditionalGeneration
    pipe_line.model._init_new_params_(**kwargs)
    # pipe_line.model.model.__class__ = renew_backbone(pipe_line.model.model.__class__)#ChameleonModel
    return pipe_line