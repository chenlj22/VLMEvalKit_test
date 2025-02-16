# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.utils import logging as logging_this

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from vlmeval.smp import *
from vlmeval.vlm.internvl.utils import (build_multi_choice_prompt,
                    build_video_prompt,
                    build_mpo_prompt,
                    build_mcq_cot_prompt,
                    build_qa_cot_prompt,
                    mpo_post_processing,
                    reorganize_prompt,
                    split_model, load_image)
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor

logger = logging_this.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message') and config.system_message:
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)


    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            # print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

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

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()


class InternVLRewardModel(InternVLChatModel):
    @staticmethod
    def split_response(response, sep='\n\n', max_steps=None):
        steps = response.split(sep)

        if max_steps is not None:
            step = math.ceil(len(steps) / max_steps)
            new_steps = []
            for i in range(0, len(steps), step):
                new_steps.append(sep.join(steps[i:i+step]))
            return new_steps

        return steps

    @staticmethod
    def join_steps(steps, sep='\n\n'):
        return sep.join(steps)

    def generate_steps_with_hard_score(
        self,
        tokenizer,
        question,
        response,
        pixel_values,
        num_patches_list=None,
        max_steps=None,
        early_stop=False,
        orm=False,
        verbose=False,
        str2score=None,
    ):
        steps_with_score = []
        generation_config = dict(
            max_new_tokens=10,
            do_sample=False,
        )

        if pixel_values is not None:
            question = f'<image>\n{question}'

        if str2score is None:
            str2score = {
                'Good': 2,
                'Slightly Good': 1,
                'Neural': 0,
                'Slightly Bad': -1,
                'Bad': -2,
                #
                '+': 0,
                '-': -100,
            }

        history = None
        steps = [response] if orm else self.split_response(response, max_steps=max_steps)
        for step_idx, step in enumerate(steps):
            if step_idx == 0:
                step = f'### Question:\n{question}\n\n### Solution Process:\n{step}'

            pred, history = self.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=step,
                generation_config=generation_config,
                history=history,
                return_history=True,
                num_patches_list=num_patches_list,
                verbose=verbose,
            )
            if pred not in str2score:
                print(f'Invalid pred or str2score: {pred=}, {str2score=}')
                pred = '-'
            steps_with_score.append({'step': step, 'score': str2score[pred]})

            if early_stop and pred in ['Bad', '-']:
                break

        return steps_with_score

    def generate_steps_with_soft_score(
        self,
        tokenizer,
        question,
        response,
        pixel_values,
        num_patches_list=None,
        max_steps=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        str2score=None,
        orm=False,
        verbose=False,
        # keep compatible with generate_steps_with_hard_score
        early_stop=False,
    ):
        if str2score is None:
            str2score = {'+': 1, '-': 0}

        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        assert pixel_values is None or (len(pixel_values) == sum(num_patches_list) and len(num_patches_list) == question.count('<image>'))

        image_input = pixel_values is not None
        if pixel_values is None:
            pixel_values = torch.zeros(1, 3, self.config.vision_config.image_size, self.config.vision_config.image_size)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        history = []
        candidate_tokens = []
        candidate_weights = []
        steps_with_score = []
        steps = [response] if orm else self.split_response(response, max_steps=max_steps)

        for k, v in str2score.items():
            k_id = tokenizer.convert_tokens_to_ids(k)
            assert k_id != tokenizer.unk_token_id

            candidate_tokens.append(k_id)
            candidate_weights.append(v)

        for step_idx, step in enumerate(steps):
            if step_idx == 0:
                step = f'### Question:\n{question}\n\n### Solution Process:\n{step}'

            template = get_conv_template(self.template)
            template.system_message = self.system_message

            for (old_question, old_answer) in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
            template.append_message(template.roles[0], step)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            model_inputs = tokenizer(query, return_tensors='pt')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_ids = model_inputs['input_ids'].to(device)
            attention_mask = model_inputs['attention_mask'].to(device)
            image_flags = torch.tensor([image_input] * pixel_values.size(0), dtype=torch.long).to(device)

            logits = self(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
            ).logits[:, -1, candidate_tokens]
            soft_scores = logits.softmax(dim=-1)[0].tolist()

            score = 0
            for s, w in zip(soft_scores, candidate_weights):
                score += s * w

            steps_with_score.append({'step': step, 'score': score})
            history.append((step, '+'))

        return steps_with_score

    def find_placeholder_idx(self, tokenizer, input_ids, PLACEHOLDER):
        # TODO: support batch inference
        input_ids = input_ids[0].tolist()
        template = get_conv_template(self.template)

        idx = []
        bos =  tokenizer(template.roles[1], add_special_tokens=False).input_ids
        target = tokenizer(template.roles[1] + PLACEHOLDER + template.sep, add_special_tokens=False).input_ids
        for i in range(len(input_ids)):
            if input_ids[i:i+len(target)] == target:
                assert i + len(bos) - 1 >= 0
                idx.append(i + len(bos) - 1)

        return idx

    def generate_steps_with_soft_score_v2(
        self,
        tokenizer,
        question,
        response,
        pixel_values,
        num_patches_list=None,
        max_steps=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        PLACEHOLDER=None,
        str2score=None,
        orm=False,
        verbose=False,
        # keep compatible with generate_steps_with_hard_score
        early_stop=False,
    ):
        if str2score is None:
            str2score = {'+': 1, '-': 0}

        if PLACEHOLDER is None:
            PLACEHOLDER = '+'

        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        assert pixel_values is None or (len(pixel_values) == sum(num_patches_list) and len(num_patches_list) == question.count('<image>'))

        image_input = pixel_values is not None
        if pixel_values is None:
            pixel_values = torch.zeros(1, 3, self.config.vision_config.image_size, self.config.vision_config.image_size)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        candidate_tokens = []
        candidate_weights = []
        steps = [response] if orm else self.split_response(response, max_steps=max_steps)

        # Prepare Query
        for k, v in str2score.items():
            k_id = tokenizer.convert_tokens_to_ids(k)
            assert k_id != tokenizer.unk_token_id

            candidate_tokens.append(k_id)
            candidate_weights.append(v)

        template = get_conv_template(self.template)
        template.system_message = self.system_message

        for step_idx, step in enumerate(steps):
            if step_idx == 0:
                step = f'### Question:\n{question}\n\n### Solution Process:\n{step}'
            template.append_message(template.roles[0], step)
            template.append_message(template.roles[1], PLACEHOLDER)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        # Prepare inputs
        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        image_flags = torch.tensor([image_input] * pixel_values.size(0), dtype=torch.long).to(device)

        # Forward
        idx = self.find_placeholder_idx(tokenizer, input_ids, PLACEHOLDER=PLACEHOLDER)

        if pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16).to(device)

        logits = self(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
        ).logits
        logits = logits[0][idx, :][:, candidate_tokens]
        soft_scores = logits.softmax(dim=-1).tolist()

        assert len(soft_scores) == len(steps)

        # Gather step scores
        steps_with_score = []
        for soft_score, step in zip(soft_scores, steps):
            score = 0
            for s, w in zip(soft_score, candidate_weights):
                score += s * w
            steps_with_score.append({'step': step, 'score': score})
        return steps_with_score

    def generate_overall_score(self, steps_with_score, version='hard'):
        overall_score = []
        for step in steps_with_score:
            curr_score = step['score']
            overall_score.append(curr_score)

        return sum(overall_score) if version == 'hard' else sum(overall_score) / len(overall_score)

    @torch.inference_mode()
    def select_best_response(
        self,
        tokenizer,
        question,
        response_list,
        pixel_values,
        num_patches_list=None,
        max_steps=None,
        early_stop=False,
        verbose=False,
        version='hard',
        gather_func=None,
    ):
        if version == 'hard':
            orm = False
            generate_steps_with_score = self.generate_steps_with_hard_score
        elif version == 'soft':
            orm = False
            generate_steps_with_score = self.generate_steps_with_soft_score
        elif version == 'soft_v2':
            orm = False
            generate_steps_with_score = self.generate_steps_with_soft_score_v2
        elif version == 'orm_soft':
            orm = True
            generate_steps_with_score = self.generate_steps_with_soft_score
        else:
            raise NotImplementedError(f'Unsupported version: {version}')
        
        scored_response_list=[]
        sorted_response_list = []

        for response in response_list:
            steps_with_score = generate_steps_with_score(
                tokenizer=tokenizer,
                question=question,
                response=response,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                max_steps=max_steps,
                early_stop=early_stop,
                orm=orm,
                verbose=verbose,
            )
            overall_score = self.generate_overall_score(steps_with_score, version=version)
            scored_response_list.append((response, overall_score))

        sorted_response_list = sorted(scored_response_list, key=lambda x:x[1], reverse=True)
        return [item[0] for item in sorted_response_list]
    

    def get_best_idx(self, best_response: str, response_list: List[str]) -> int:
        '''
        获取 best_response 在 response_list 中的索引。

        best_response: 最佳响应字符串
        response_list: 响应字符串列表
        :return: best_response 在 response_list 中的索引，如果未找到则返回 -1
        '''
        try:
            best_response_idx = response_list.index(best_response)
        except ValueError:
            best_response_idx = -1  # 如果 best_response 不在 response_list 中，返回 -1
        return best_response_idx    

    @torch.inference_mode()
    def get_best_response_idx(
        self,
        tokenizer,
        question,
        response_list,
        pixel_values,
        num_patches_list=None,
        max_steps=None,
        early_stop=False,
        verbose=False,
        version='hard',
        gather_func=None,
    ):
        best_response=self.select_best_response(tokenizer=tokenizer, question=str(question), response_list=response_list, pixel_values=pixel_values, num_patches_list=num_patches_list, max_steps=max_steps, early_stop=early_stop, verbose=verbose, version=version, gather_func=gather_func)
        best_response_idx=self.get_best_idx(best_response=best_response[0], response_list=response_list)
        best_response_idx+=1
        return best_response_idx
    
    @torch.no_grad()
    def judge(
        self, 
        message, 
        dataset=None    
    ):
        image_num = len([x for x in message if x['type'] == 'image'])
        # max_num = max(1, min(self.max_num, self.total_max_num // image_num))

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        
        tokenizer=AutoTokenizer.from_pretrained('/mnt/petrelfs/share_data/wangweiyun/share_internvl_preview/InternVL2_5-8B-PRM-v0', trust_remote_code=True, use_fast=False)
        question=[x['value'] for x in message if x['type'] == 'question']
        response_list=[x['value'] for x in message if x['type'] == 'response']
        response_list=response_list[0]
        best_response_idx=self.get_best_response_idx(
            tokenizer=tokenizer,
            question=question,
            response_list=response_list,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            max_steps=12,
            early_stop=True,
            version='soft_v2',
        )
        
        return best_response_idx

    @torch.inference_mode()
    def get_first_idx(self,
        tokenizer,
        problem,
        steps,
        early_stop=False,
        verbose=False,
        version='hard',
        gather_func=None,
        theta=0.5,
    ):
        if version == 'hard':
            orm = False
            generate_steps_with_score = self.generate_steps_with_hard_score
        elif version == 'soft':
            orm = False
            generate_steps_with_score = self.generate_steps_with_soft_score
        elif version == 'soft_v2':
            orm = False
            generate_steps_with_score = self.generate_steps_with_soft_score_v2
        elif version == 'orm_soft':
            orm = True
            generate_steps_with_score = self.generate_steps_with_soft_score
        else:
            raise NotImplementedError(f'Unsupported version: {version}')
        
        scored_response_list=[]

        response=self.join_steps(steps=steps)

        steps_with_score = generate_steps_with_score(
            tokenizer=tokenizer,
            question=problem,
            response=response,
            pixel_values=None,
            early_stop=early_stop,
            orm=orm,
            verbose=verbose,
        )

        for index, step in enumerate(steps_with_score):
            if step['score'] < theta:
                return index
        return -1


    @torch.no_grad()
    def get_first_wrong_idx(
        self, 
        message, 
        dataset=None,
        theta=0.5, 
    ):
        tokenizer=AutoTokenizer.from_pretrained('/mnt/petrelfs/share_data/wangweiyun/share_internvl_preview/InternVL2_5-8B-PRM-v0', trust_remote_code=True, use_fast=False)
        problem=[x['value'] for x in message if x['type'] == 'problem'][0]
        steps=[x['value'] for x in message if x['type'] == 'steps'][0]
        first_wrong_idx=self.get_first_idx(
            tokenizer=tokenizer,
            problem=problem,
            steps=steps,
            early_stop=True,
            version='soft_v2',
            theta=theta,
        )
        
        return first_wrong_idx

    @torch.inference_mode()
    def get_process_correctness(
        self,
        tokenizer,
        question,
        steps,
        pixel_values,
        num_patches_list=None,
        max_steps=None,
        early_stop=False,
        verbose=False,
        version='hard',
        gather_func=None,
        theta=0.5,
    ):
        if version == 'hard':
            orm = False
            generate_steps_with_score = self.generate_steps_with_hard_score
        elif version == 'soft':
            orm = False
            generate_steps_with_score = self.generate_steps_with_soft_score
        elif version == 'soft_v2':
            orm = False
            generate_steps_with_score = self.generate_steps_with_soft_score_v2
        elif version == 'orm_soft':
            orm = True
            generate_steps_with_score = self.generate_steps_with_soft_score
        else:
            raise NotImplementedError(f'Unsupported version: {version}')
        
        scored_response_list=[]

        response=self.join_steps(steps=steps)

        steps_with_score = generate_steps_with_score(
            tokenizer=tokenizer,
            question=question,
            response=response,
            pixel_values=None,
            early_stop=early_stop,
            orm=orm,
            verbose=verbose,
        )

        process_correctness = []

        # 遍历 steps_with_score 列表
        for step in steps_with_score:
            score = step['score']  # 获取 'score' 的值
            if score >= theta:
                process_correctness.append(1)  # 如果 score 大于等于 theta，添加 1
            else:
                process_correctness.append(-1)  # 如果 score 小于 theta，添加 -1

        return process_correctness

    @torch.no_grad()
    def predict_process_correctness(
        self, 
        message, 
        dataset=None,
        theta=0.5, 
    ):
        tokenizer=AutoTokenizer.from_pretrained('/mnt/petrelfs/share_data/wangweiyun/share_internvl_preview/InternVL2_5-8B-PRM-v0', trust_remote_code=True, use_fast=False)
        question=[x['value'] for x in message if x['type'] == 'question'][0]
        steps=[x['value'] for x in message if x['type'] == 'steps'][0]

        image_num = len([x for x in message if x['type'] == 'image_path'][0])
        # max_num = max(1, min(self.max_num, self.total_max_num // image_num))

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image_path'][0]
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        process_correctness=self.get_process_correctness(
            tokenizer=tokenizer,
            question=question,
            steps=steps,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            early_stop=True,
            version='soft_v2',
            theta=theta,
        )
        
        return process_correctness



    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func