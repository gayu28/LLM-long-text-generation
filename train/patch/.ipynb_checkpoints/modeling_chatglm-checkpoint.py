import math
import copy
import warnings
import re
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

from .configuration_chatglm import ChatGLMConfig
from einops import rearrange
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None

if sys.platform != 'darwin':
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/ChatGLM2-6B"
_CONFIG_FOR_DOC = "ChatGLM6BConfig"

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm2-6b",
]

def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)

class TwoPhasePretraining:
    """
    Implements a two-phase pretraining strategy:
    - Phase 1: Foundational linguistic learning on a large, diverse corpus.
    - Phase 2: Curriculum-based domain-specific pretraining on curated datasets.
    """
    def __init__(self, model, optimizer, scheduler, device, distributed_backend='nccl'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.distributed_backend = distributed_backend

    def foundational_linguistic_learning(self, dataset, batch_size):
        """Phase 1: Train the model on a large, diverse corpus."""
        self.model.train()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            inputs, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
            outputs = self.model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def curriculum_domain_specific_training(self, dataset, curriculum, batch_size):
        """Phase 2: Train with curriculum-based domain-specific datasets."""
        self.model.train()
        for phase, data_config in enumerate(curriculum):
            logger.info(f"Phase {phase + 1}/{len(curriculum)}: {data_config['name']}")
            dataloader = torch.utils.data.DataLoader(
                dataset[data_config['name']], batch_size=batch_size, shuffle=True
            )
            for batch in dataloader:
                inputs, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

    def execute(self, phase1_dataset, phase2_dataset, curriculum, batch_size):
        """Executes the two-phase pretraining."""
        logger.info("Starting Phase 1: Foundational Linguistic Learning")
        self.foundational_linguistic_learning(phase1_dataset, batch_size)
        logger.info("Phase 1 completed.")

        logger.info("Starting Phase 2: Curriculum-based Domain-specific Training")
        self.curriculum_domain_specific_training(phase2_dataset, curriculum, batch_size)
        logger.info("Phase 2 completed.")

class SparseAttention(nn.Module):
    """Implements sparse attention for long dependency modeling."""
    def __init__(self, block_size=64, sparsity_ratio=0.5):
        super(SparseAttention, self).__init__()
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio

    def forward(self, query, key, value, mask=None):
        """Compute sparse attention with block sparsity."""
        batch_size, seq_len, dim = query.size()
        num_blocks = seq_len // self.block_size
        attention_scores = torch.zeros(batch_size, seq_len, seq_len, device=query.device)

        for i in range(num_blocks):
            for j in range(max(0, i - int(num_blocks * self.sparsity_ratio)), min(num_blocks, i + int(num_blocks * self.sparsity_ratio))):
                q_block = query[:, i * self.block_size:(i + 1) * self.block_size]
                k_block = key[:, j * self.block_size:(j + 1) * self.block_size]
                v_block = value[:, j * self.block_size:(j + 1) * self.block_size]
                scores = torch.einsum('bqd,bkd->bqk', q_block, k_block)
                attention_scores[:, i * self.block_size:(i + 1) * self.block_size, j * self.block_size:(j + 1) * self.block_size] = scores

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)
        return output

# Incorporate the TwoPhasePretraining and SparseAttention modules into the ChatGLM model
class ChatGLMModelWithSparseAttention(ChatGLMModel):
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config, device=device, empty_init=empty_init)
        self.sparse_attention = SparseAttention(block_size=config.block_size, sparsity_ratio=config.sparsity_ratio)

    def forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Apply sparse attention for enhanced long-context dependency modeling
        output.last_hidden_state = self.sparse_attention(
            output.last_hidden_state, output.last_hidden_state, output.last_hidden_state, attention_mask
        )

        return output
