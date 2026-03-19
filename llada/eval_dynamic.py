# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import accelerate
import torch
import torch.nn as nn
from pathlib import Path
import random
import numpy as np
import math
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from contextlib import contextmanager
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM
import json
import time
import torch._dynamo


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

# ==============================================================================
# 注意力捕获钩子 (Attention Hooks)
# ==============================================================================
original_sdpa = F.scaled_dot_product_attention
original_F_softmax = F.softmax
original_torch_softmax = torch.softmax
captured_attentions = []

def hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(~attn_mask, float('-inf'))
        else:
            attn_weight += attn_mask
    elif is_causal:
        L, S = query.size(-2), key.size(-2)
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        attn_weight.masked_fill_(~causal_mask, float('-inf'))
            
    attn_probs = torch.softmax(attn_weight, dim=-1)
    
    # 【显存修复核心】：截获后立马放进列表
    captured_attentions.append(attn_probs.detach().float())
    # 只要超过 3 层，立马把最老的矩阵从显存里删掉！(立省 90% 显存)
    if len(captured_attentions) > 3:
        del captured_attentions[0]
        
    return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)

def hooked_F_softmax(*args, **kwargs):
    out = original_F_softmax(*args, **kwargs)
    if out.ndim == 4: 
        captured_attentions.append(out.detach().float())
        if len(captured_attentions) > 3: del captured_attentions[0]
    return out

def hooked_torch_softmax(*args, **kwargs):
    out = original_torch_softmax(*args, **kwargs)
    if out.ndim == 4: 
        captured_attentions.append(out.detach().float())
        if len(captured_attentions) > 3: del captured_attentions[0]
    return out

@contextmanager
def capture_attention():
    global captured_attentions
    captured_attentions = []
    F.scaled_dot_product_attention = hooked_sdpa
    F.softmax = hooked_F_softmax
    torch.softmax = hooked_torch_softmax
    try:
        yield captured_attentions
    finally:
        F.scaled_dot_product_attention = original_sdpa
        F.softmax = original_F_softmax
        torch.softmax = original_torch_softmax
        # 退出上下文时，彻底清空显存垃圾
        captured_attentions = []

# ==============================================================================
# 基础工具函数
# ==============================================================================
def add_gumbel_noise(logits, temperature):
    if temperature == 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    device = block_mask_index.device
    dtype = torch.long
    total = block_mask_index.sum(dim=1)
    base  = torch.div(total, steps, rounding_mode='floor')
    rem   = total - base * steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)
    cols = torch.arange(steps, device=device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    return num_transfer_tokens + add_mask.to(dtype)

def get_transfer_index(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    x0 = predicted_tokens  # (B, L)

    # Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)  # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else: raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0: continue
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]
        threshs[0] = -1
        sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]: break
        if top_i == 0 or top_i == len(threshs) - 1: top_i += 1
        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


import torch
import math

# ==============================================================================
# 终极版：带未来视野的峰值跌落检测 (Look-Ahead Recovery)
# ==============================================================================
def find_dependency_boundary(
    avg_attn: torch.Tensor,
    current_s: int,
    prev_s: int,
    min_block_length: int,
    probe_limit: int,
    drop_ratio: float = 0.3  # 半山腰线：低于最高点的 50% 视为跌落
):
    """能够完美识别并保留内部从句低谷，只切断真正的语义死亡点"""
    if probe_limit <= min_block_length + 1:
        return probe_limit

    # 1. 提取注意力依赖度
    prev_block_attn = avg_attn[current_s : current_s + probe_limit, prev_s : current_s]
    dependency = prev_block_attn.sum(dim=-1)
    
    # 2. 找到全场最高峰（通常是前两三个词带来的距离红利）
    peak = dependency.max().item()
    threshold = peak * drop_ratio
    
    best_k = probe_limit 

    # 3. 从前往后扫，遇到跌破阈值的词，先别急着切
    for k in range(min_block_length, probe_limit):
        if dependency[k] < threshold:
            
            future_max = dependency[k:].max().item()
            
            if future_max >= threshold:
                continue
            else:
                best_k = k+1
                break

    return best_k

# ==============================================================================
# 修改后的 generate_with_dynamic_dual_cache (支持 Attention Entropy Remasking)
# ==============================================================================
@torch.no_grad()
def generate_with_dynamic_dual_cache(
    model, prompt, steps=128, gen_length=128, init_block_length=32, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, 
    delimiter_ids=[198], delimiter_threshold=float('inf'),
    drop_epsilon=1.0, sink_epsilon=1.5, min_block_length=4,
    entropy_threshold=0.3  # 【新增核心参数】：熵增超过 0.2 的旧词将被强行 Remask
): 
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert threshold is not None, "threshold must be set"

    generated_length = 0
    nfe_history = []  
    block_history = []
    
    prompt_length = prompt.shape[1]
    # 【修改】：初始化设为 prompt_length，因为我们绝不能去 remask 用户的 prompt
    prev_s = prompt_length 
    
    while generated_length < gen_length: 
        nfe = 0
        current_s = prompt_length + generated_length
        active_start = current_s # 默认当前 Block 的解码起点

        old_disable_state = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            with capture_attention() as captured:
                output = model(x, use_cache=True)
        finally:
            torch._dynamo.config.disable = old_disable_state

        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        probe_limit = min(init_block_length, gen_length - generated_length)
        
        if len(captured) > 0:
            total_layers = len(captured)
            selected_captured = captured[-3:] if total_layers >= 3 else captured
            
            stacked_attn = torch.stack([c for c in selected_captured])
            if stacked_attn.dim() == 5:
                avg_attn = stacked_attn.mean(dim=(0, 2))[0].float()
            else:
                avg_attn = stacked_attn.mean(dim=0)[0].float()
                
            block_length = find_dependency_boundary(
                avg_attn=avg_attn,
                current_s=current_s,
                prev_s=prev_s, # 这里的 prev_s 保证是上一个生成的 Block
                min_block_length=min_block_length,
                probe_limit=probe_limit,
            )
            
            # ==============================================================================
            # 🚀 【核心新增逻辑】：动态注意力熵增检测与 Remask 机制
            # ==============================================================================
            if prev_s < current_s: # 确保存在前一个生成的 Block
                # 1. P_old：前一个 Block 仅看旧上下文时的注意力分布
                P_old = avg_attn[prev_s:current_s, :current_s]
                # 重新归一化，使其成为合法的概率分布
                P_old_norm = P_old / (P_old.sum(dim=-1, keepdim=True) + 1e-9)
                # 计算香农熵 H = -sum(p * log(p))
                H_old = -(P_old_norm * torch.log(P_old_norm + 1e-9)).sum(dim=-1)

                # 2. P_new：前一个 Block 看到“旧上下文 + 新 Block”时的注意力分布
                P_new = avg_attn[prev_s:current_s, :current_s + block_length]
                P_new_norm = P_new / (P_new.sum(dim=-1, keepdim=True) + 1e-9)
                H_new = -(P_new_norm * torch.log(P_new_norm + 1e-9)).sum(dim=-1)

                # 3. 寻找“事后迷茫”的 Token
                delta_H = H_new - H_old
                
                # 找到熵增超过阈值的相对索引
                remask_relative_idx = torch.nonzero(delta_H > entropy_threshold).squeeze(-1)
                print(len(remask_relative_idx))
                
                if len(remask_relative_idx) > 0:
                    remask_global_idx = prev_s + remask_relative_idx
                    # 【绝杀】：将这些动摇的位置重新变回 MASK！
                    x[:, remask_global_idx] = mask_id
                    
                    # 扩展当前循环的“活跃区域”，把前一个 Block 包容进来一起解码
                    active_start = prev_s 
        else:
            block_length = probe_limit 

        block_history.append(block_length)
        
        block_end = current_s + block_length
        generated_length += block_length
        
        # 限制第一步 Transfer 只能在活跃区域内进行 (防止解开未来的 MASK 或是历史的非 MASK)
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        mask_index[:, :active_start] = 0 
        
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        # 准备 Dual Cache 的局部重算遮罩
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, active_start:block_end] = 1
        
        # 2nd and later denoising steps in block
        while True:
            # 跳出条件：活跃区域内已经没有 MASK 了
            if (x[:, active_start:block_end] == mask_id).sum() == 0:
                break
                
            # 注意：内部循环的所有张量切片，都必须从 active_start 开始！
            mask_index = (x[:, active_start:block_end] == mask_id)
            
            block_output = model(
                x[:, active_start:block_end], 
                past_key_values=full_cache, 
                use_cache=True, 
                replace_position=replace_position
            )
            
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            
            x0, transfer_index = get_transfer_index(
                block_logits, block_predicted_tokens, remasking, mask_index, 
                x[:, active_start:block_end], None, threshold
            )
            
            x[:, active_start:block_end][transfer_index] = x0[transfer_index]
            
        nfe_history.append(nfe)
        
        # 核心流转：无论是否发生了回滚，当前的尽头变成了下一次的起点
        prev_s = active_start 

    return x, nfe, block_history

# ==============================================================================
# 工具函数 & Harness
# ==============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self, model_path='', mask_id=126336, max_length=4096, batch_size=32, mc_num=128,
        is_check_greedy=True, steps=1024, gen_length=1024, block_length=1024,
        remasking='low_confidence', device="cuda", use_cache=False, threshold=None,
        factor=None, save_dir=None, show_speed=False, dual_cache=False,
        use_dynamic_block=False, max_block_length=64, min_block_length=4, smooth_window=3, **kwargs
    ):
        super().__init__()
        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        config = AutoConfig.from_pretrained(model_path)
        # 关闭 Flash Attention 保证 Hook 可用
        config.flash_attention = False 
        
        self.model = LLaDAModelLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.is_instruct = True if 'instruct' in model_path.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        self.use_dynamic_block = use_dynamic_block
        self.max_block_length = int(max_block_length) 
        self.min_block_length = int(min_block_length) 
        self.smooth_window = int(smooth_window)

    @property
    def rank(self): return self._rank

    @property
    def world_size(self): return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b): is_mask[i] = is_mask[i][torch.randperm(target_len)]
        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        logits = self.model(batch).logits
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :].repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss_acc.append((loss.sum() / self.batch_size).item())
        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy: return False
        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix
        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        return torch.all(target == seq[0, len(prefix):])

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        return context_enc, whole_enc[len(context_enc):]

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {"prefix": prefix, "target": target}
        ds = Dataset.from_list([{"prefix": r.args[0], "target": r.args[1]} for r in requests]).map(_tokenize).with_format("torch")
        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix, target = elem["prefix"], elem["target"]
                ll = self.get_loglikelihood(prefix, target)
                out.append((ll, 1.0 if self.suffix_greedy_prediction(prefix, target) else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests): raise NotImplementedError

    def generate_until(self, requests):
        output, num_tokens, num_nfe, processed_count, total_count = [], 0, 0, 0, 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'rank_{getattr(self, "rank", 0)}.jsonl')
            if os.path.exists(save_path):
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)

        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count: continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size: batched_requests.append([])
        if len(batched_requests[-1]) == 0: batched_requests.pop()

        start_time, global_total_blocks, global_total_block_length = time.time(), 0, 0

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids, max_len, pad_len = [], 0, []
            for req in batch:
                question = req.args[0]
                user_input = self.tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False) if self.is_instruct else question
                input_ids = self.tokenizer(user_input)['input_ids']
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))

            batched_input_ids = torch.cat([
                torch.cat([torch.full((1, max_len - len(ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device),
                           torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1)
                for ids in batched_input_ids
            ], dim=0).to(self.device)

            stop_tokens = req.args[1].get('until', [])
            input_ids = batched_input_ids
            block_boundaries = []

            if self.use_cache:
                if self.dual_cache:
                    if getattr(self, "use_dynamic_block", False):
                        generated_answer, nfe, block_boundaries = generate_with_dynamic_dual_cache(
                            model=self.model, prompt=input_ids, steps=self.steps, gen_length=self.gen_length,
                            # max_block_length=self.max_block_length, min_block_length=self.min_block_length,
                            temperature=0, remasking=self.remasking, threshold=self.threshold,
                            mask_id=self.mask_id,
                        )
                    else:
                        generated_answer, nfe = generate_with_dual_cache(
                            self.model, input_ids, steps=self.steps, gen_length=self.gen_length,
                            block_length=self.block_length, temperature=0, remasking=self.remasking, 
                            mask_id=self.mask_id, threshold=self.threshold, factor=self.factor
                        )
                else:
                    generated_answer, nfe = generate_with_prefix_cache(
                        self.model, input_ids, steps=self.steps, gen_length=self.gen_length,
                        block_length=self.block_length, temperature=0, remasking=self.remasking, 
                        mask_id=self.mask_id, threshold=self.threshold, factor=self.factor
                    )
            else:
                generated_answer, nfe = generate(
                    self.model, input_ids, steps=self.steps, gen_length=self.gen_length,
                    block_length=self.block_length, temperature=0, remasking=self.remasking, 
                    mask_id=self.mask_id, threshold=self.threshold, factor=self.factor
                )

            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                batched_generated_answer = [self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) for i in range(len(generated_answer_ids))]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    gen_str = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                    for stop_seq in stop_tokens:
                        if stop_seq in gen_str: gen_str = gen_str.split(stop_seq)[0]
                    gen_ids = torch.tensor(self.tokenizer(gen_str)["input_ids"])
                    if self.show_speed:
                        num_tokens += (gen_ids != 126081).sum()
                        num_nfe += nfe
                    batched_generated_answer.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))

            output.extend(batched_generated_answer)
            if self.save_dir is not None:
                with open(save_path, 'a', encoding='utf-8') as f:
                    for ans in batched_generated_answer: f.write(json.dumps(ans, ensure_ascii=False) + '\n')

            # ---------------------------------------------------------
            # ✅ 新增：为每一道题精确打印详细的 Block 历史和平均大小
            # ---------------------------------------------------------
            if block_boundaries:
                global_total_blocks += len(block_boundaries)
                global_total_block_length += sum(block_boundaries)

            for i in range(len(batched_generated_answer)):
                total_count += 1
                print('=' * 60)
                
                # 提取这道题的每一个 Block 的确切大小
                if block_boundaries:
                    block_sizes = [s for s in block_boundaries]
                    avg_size_for_this_question = sum(block_sizes) / len(block_sizes)
                    
                    print(f"[Question {total_count}] Attention Entropy Dynamic Blocks:")
                    print(f" └─ History array: {block_sizes}")
                    print(f" └─ Number of Blocks: {len(block_sizes)}")
                    print(f" └─ ★ Average Block Size: {avg_size_for_this_question:.2f} ★")
                else:
                    print(f"[Question {total_count}] Static Block Size: {self.block_length}")

                preview = batched_generated_answer[i][:150].replace('\n', '\\n') + ("..." if len(batched_generated_answer[i]) > 150 else "")
                print(f" └─ Preview: {preview}")
                print('=' * 60)

        end_time = time.time()
        if total_count > 0:
            print("=" * 40)
            print("✅ Generation Complete. Ready for evaluation.")
            if global_total_blocks > 0:
                print(f"📏 Global Final Avg Block Size: {global_total_block_length / global_total_blocks:.2f}")
            print("=" * 40)

        if self.show_speed:
            print(f"Total tokens generated : {num_tokens}")
            print(f"Total time             : {end_time - start_time:.2f}s")
            if end_time - start_time > 0: print(f"Tokens/sec             : {num_tokens / (end_time - start_time):.1f}")
            print(f"Total NFE              : {num_nfe}")

        return output

if __name__ == "__main__":
    cli_evaluate()