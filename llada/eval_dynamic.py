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

DEFAULT_SEED = int(os.environ.get("LLADA_SEED", "42"))
ANCHOR_LAYER_INDICES = (2,3,6,7,8)
DEFAULT_MAX_DYNAMIC_BLOCK = 32
DEFAULT_MIN_DYNAMIC_BLOCK = 16
DEFAULT_DYNAMIC_WARMUP_BLOCKS = 2
DEFAULT_DYNAMIC_WINDOW_FACTOR = 2
DEFAULT_TAIL_MERGE_THRESHOLD = 12
DEFAULT_LATE_STAGE_STATIC_TOKENS = 48
BALANCE_EPS = 1e-6


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
# 注意力捕获钩子 (Attention Hooks) - 🌟 适配 Rollout 算法
# ==============================================================================
original_sdpa = F.scaled_dot_product_attention
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
    
    # 【Rollout 显存修复核心】：截获后立刻在 Head 维度求平均，缩减 32 倍显存！
    # 这样我们就可以安全地保留网络所有层的信息，而不需要删除。
    if attn_probs.dim() == 4:
        captured_attentions.append(attn_probs.detach().float().mean(dim=1))
    else:
        captured_attentions.append(attn_probs.detach().float())
        
    return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)

@contextmanager
def capture_attention():
    global captured_attentions
    captured_attentions = []
    F.scaled_dot_product_attention = hooked_sdpa
    try:
        yield captured_attentions
    finally:
        F.scaled_dot_product_attention = original_sdpa
        # 退出上下文时，彻底清空显存垃圾
        captured_attentions = []

# ==============================================================================
# 基础工具函数
# ==============================================================================
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

import matplotlib.pyplot as plt
import os
import math

def visualize_attention_across_layers(captured_attentions, prev_s, current_s, probe_limit, nfe, step_save_dir="attn_visuals"):
    """
    可视化每一层的 MASK 关注度求和曲线。
    这能帮你直观地挑出“信号最尖锐、底噪最低”的那几层。
    """
    os.makedirs(step_save_dir, exist_ok=True)
    num_layers = len(captured_attentions)
    
    # 动态计算网格大小 (例如 32 层就画一个 4x8 的网格)
    cols = 4
    rows = math.ceil(num_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), sharey=True)
    axes = axes.flatten()

    for layer_idx, attn_matrix in enumerate(captured_attentions):
        # 处理不同 Hook 返回的维度 (通常取 batch [0])
        if attn_matrix.dim() == 3:
            attn = attn_matrix[0] 
        else:
            attn = attn_matrix

        # 核心：提取前文对 MASK 的注意力，并沿 Query 维度求和
        # 形状：[prev_s : current_s, current_s : current_s + probe_limit]
        real_to_mask_attn = attn[prev_s : current_s, current_s : current_s + probe_limit]
        attention_sum = real_to_mask_attn.sum(dim=0).cpu().float().numpy()

        # 绘制折线图
        ax = axes[layer_idx]
        ax.plot(range(len(attention_sum)), attention_sum, marker='o', markersize=4, linewidth=1.5)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 标出平均底噪线
        mu_sum = attention_sum.mean()
        ax.axhline(y=mu_sum, color='r', linestyle=':', alpha=0.8, label="Mean Base Noise")

    # 隐藏多余的子图
    for i in range(num_layers, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    save_path = os.path.join(step_save_dir, f"nfe_{nfe}_attn_sum.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 每一层注意力波动图已保存至: {save_path}")

import torch
import math

def _sanitize_attention_tensor(values: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with finite values before scoring."""
    return torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_feature(values: torch.Tensor) -> torch.Tensor:
    values = _sanitize_attention_tensor(values)
    mean = values.mean()
    std = values.std(unbiased=False)
    if std.item() < BALANCE_EPS:
        return torch.zeros_like(values)
    return torch.relu((values - mean) / (std + BALANCE_EPS))


def compute_structural_consensus_scores(
    selected_attn_layers,
    prefix_start: int,
    current_s: int,
    window_size: int,
):
    """Raw prefix-to-window attention with soft cross-layer consensus."""
    if window_size <= 0 or prefix_start >= current_s or not selected_attn_layers:
        return None

    layer_scores = []
    for layer_attn in selected_attn_layers:
        rtm = layer_attn[0, prefix_start:current_s, current_s: current_s + window_size]
        if rtm.numel() == 0 or rtm.shape[0] == 0:
            continue
        layer_scores.append(_sanitize_attention_tensor(rtm.sum(dim=0)))

    if not layer_scores:
        return None

    score_matrix = torch.stack(layer_scores, dim=0)
    mean_scores = score_matrix.mean(dim=0)
    normalized_layers = torch.stack(
        [_normalize_feature(layer_score) for layer_score in layer_scores],
        dim=0,
    )
    soft_consensus = normalized_layers.mean(dim=0)
    normalized_sum = _normalize_feature(mean_scores)
    boundary_scores = normalized_sum * (1.0 + soft_consensus)
    boundary_scores = _sanitize_attention_tensor(boundary_scores)
    baseline = boundary_scores.mean()
    return {
        "score": boundary_scores,
        "baseline": baseline,
        "sum_score": mean_scores,
        "soft_consensus": soft_consensus,
    }


def compute_dependency_scores(
    avg_attn: torch.Tensor,
    prefix_start: int,
    current_s: int,
    window_size: int,
):
    """Use a simple prefix-to-mask attention sum as the dependency score."""
    if window_size <= 0 or prefix_start >= current_s:
        return None

    real_to_mask_attn = avg_attn[prefix_start:current_s, current_s: current_s + window_size]
    if real_to_mask_attn.numel() == 0 or real_to_mask_attn.shape[0] == 0:
        return None

    real_to_mask_attn = _sanitize_attention_tensor(real_to_mask_attn)
    attention_sum = _sanitize_attention_tensor(real_to_mask_attn.sum(dim=0))
    baseline = attention_sum.mean()
    return {
        "sum": attention_sum,
        "baseline": baseline,
    }


def find_dependency_boundary(
    avg_attn: torch.Tensor,
    selected_attn_layers,
    current_s: int,
    prefix_start: int,
    min_block_length: int,
    window_size: int,       # 💡 新增：大探测视野 (比如 128)
    fallback_length: int,   # 💡 新增：找不到时的安全退回长度 (比如 32)
):
    if window_size <= min_block_length or prefix_start >= current_s:
        return min(fallback_length, window_size)

    score_pack = compute_structural_consensus_scores(
        selected_attn_layers=selected_attn_layers,
        prefix_start=prefix_start,
        current_s=current_s,
        window_size=window_size,
    )
    if score_pack is None:
        return min(fallback_length, window_size)

    attention_scores = score_pack["score"]
    baseline = score_pack["baseline"].item()
    total_mass = attention_scores.sum().item()
    if total_mass <= 0:
        return min(fallback_length, window_size)

    cumulative = torch.cumsum(attention_scores, dim=0)
    global_density = total_mass / max(window_size, 1)
    closure_values = []
    objective_values = []

    for k in range(window_size):
        coverage = cumulative[k].item() / max(total_mass, BALANCE_EPS)
        future_length = window_size - (k + 1)
        future_mass = max(total_mass - cumulative[k].item(), 0.0)
        future_density = future_mass / max(future_length, 1) if future_length > 0 else 0.0
        leakage = future_density / max(global_density, BALANCE_EPS)
        closure = coverage - leakage
        objective = closure * attention_scores[k].item()
        closure_values.append(closure)
        objective_values.append(objective)

    chosen_idx = None
    best_local_objective = float("-inf")
    # Do not let the very last token in the probe window dominate by construction:
    # at the window end, coverage is always 1 and leakage is always 0.
    search_end = max(min_block_length, window_size - 1)
    for k in range(min_block_length - 1, search_end):
        if closure_values[k] <= 0 or objective_values[k] <= 0:
            continue
        left_val = objective_values[k - 1] if k - 1 >= min_block_length - 1 else float("-inf")
        right_val = objective_values[k + 1] if k + 1 < search_end else float("-inf")
        is_local_max = objective_values[k] >= left_val and objective_values[k] >= right_val
        if is_local_max and objective_values[k] > best_local_objective:
            best_local_objective = objective_values[k]
            chosen_idx = k

    if chosen_idx is None:
        valid_fallback_candidates = [
            idx
            for idx in range(min_block_length - 1, search_end)
            if closure_values[idx] > 0 and objective_values[idx] > 0
        ]
        if valid_fallback_candidates:
            chosen_idx = max(
                valid_fallback_candidates,
                key=lambda idx: objective_values[idx],
            )
        else:
            return min(fallback_length, window_size)

    if chosen_idx is not None:
        current_score = attention_scores[chosen_idx].item()
        coverage = cumulative[chosen_idx].item() / max(total_mass, BALANCE_EPS)
        future_length = window_size - (chosen_idx + 1)
        future_mass = max(total_mass - cumulative[chosen_idx].item(), 0.0)
        future_density = future_mass / max(future_length, 1) if future_length > 0 else 0.0
        leakage = future_density / max(global_density, BALANCE_EPS)
        closure_score = closure_values[chosen_idx]
        print(
            f"    🔭 [依赖闭包切分] 视野 {window_size} | 词 {chosen_idx} 处截断，"
            f"score={current_score:.4f}, baseline={baseline:.4f}, "
            f"sum={score_pack['sum_score'][chosen_idx].item():.4f}, "
            f"consensus={score_pack['soft_consensus'][chosen_idx].item():.4f}, "
            f"coverage={coverage:.4f}, leakage={leakage:.4f}, closure={closure_score:.4f}"
        )
        return chosen_idx + 1

    return min(fallback_length, window_size)

# ==============================================================================

@torch.no_grad()
def generate_with_dynamic_dual_cache(
    model, prompt, steps=128, gen_length=128, init_block_length=DEFAULT_MAX_DYNAMIC_BLOCK, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, min_block_length=DEFAULT_MIN_DYNAMIC_BLOCK,
    enable_dynamic_block=True,
    enable_attn_remask=False,
): 
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert threshold is not None, "threshold must be set"

    generated_length = 0
    nfe_history = []  
    block_history = []
    attention_history = []
    warmup_blocks = DEFAULT_DYNAMIC_WARMUP_BLOCKS
    tail_merge_threshold = max(DEFAULT_TAIL_MERGE_THRESHOLD, min_block_length)
    
    prompt_length = prompt.shape[1]
    prefix_start = 0
    
    while generated_length < gen_length: 
        nfe = 0
        current_s = prompt_length + generated_length
        active_start = current_s 

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
        
        remaining_length = gen_length - generated_length
        # 探测视野控制在初始块长附近，避免窗口过大后退化成“大块近似固定切分”
        window_size = min(
            128,
            remaining_length,
        )
        fallback_length = min(init_block_length, remaining_length)
        # Be conservative near the end of generation so answer formatting is less likely to break.
        late_stage_mode = remaining_length <= max(DEFAULT_LATE_STAGE_STATIC_TOKENS, init_block_length * 2)
        dynamic_enabled_this_step = (
            enable_dynamic_block
            and len(block_history) >= warmup_blocks
            and not late_stage_mode
        )
        
        if len(captured) > 0:
            B, L_seq, S_seq = captured[0].shape
            device = captured[0].device
            I = torch.eye(L_seq, device=device).unsqueeze(0).expand(B, -1, -1)
            # if nfe == 1 and current_s > 0:
            #     visualize_attention_across_layers(
            #         captured_attentions=captured,
            #         prev_s=prev_s,
            #         current_s=current_s,
            #         probe_limit=probe_limit,
            #         nfe=nfe
            #     )

            rollout = None
            selected_layer_indices = ANCHOR_LAYER_INDICES

            truncated_captured = [
                captured[i] for i in selected_layer_indices
                if 0 <= i < len(captured)
            ]

            if len(truncated_captured) == 0:
                truncated_captured = captured[-5:] if len(captured) > 0 else []

            
            for W in truncated_captured:
                A = 0.5 * W + 0.5 * I
                if rollout is None:
                    rollout = A
                else:
                    rollout = torch.bmm(A, rollout)
            if rollout is None:
                block_length = fallback_length
                avg_attn = None
            else:
                avg_attn = rollout[0]

            # 🚀 3. 动态边界探测 (传入大视窗和退回长度)
            if avg_attn is not None and dynamic_enabled_this_step:
                block_length = find_dependency_boundary(
                    avg_attn=avg_attn,
                    selected_attn_layers=truncated_captured,
                    current_s=current_s,
                    prefix_start=prefix_start,
                    min_block_length=min_block_length,
                    window_size=window_size,         # 传入大视野
                    fallback_length=fallback_length, # 传入安全底线 32
                )
            else:
                block_length = fallback_length

            # Merge tiny tail blocks so the final answer region is not fragmented into very short chunks.
            tail_length = remaining_length - block_length
            if 0 < tail_length <= tail_merge_threshold:
                block_length = remaining_length

            if avg_attn is not None:
                score_pack = compute_dependency_scores(
                    avg_attn=avg_attn,
                    prefix_start=prefix_start,
                    current_s=current_s,
                    window_size=block_length,
                )
                if score_pack is not None:
                    focus_scores = score_pack["sum"]
                    attention_history.extend(focus_scores.cpu().float().numpy().tolist())
                else:
                    attention_history.extend([0.0] * block_length)
            else:
                attention_history.extend([0.0] * block_length)
            
        else:
            block_length = fallback_length 

        block_history.append(block_length)
        block_end = current_s + block_length
        generated_length += block_length
        
        # 🚀 4. 内循环降噪
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        mask_index[:, :active_start] = 0 
        
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, active_start:block_end] = 1
        
        while True:
            current_mask_indices = (x[:, active_start:block_end] == mask_id)
            num_remaining_masks = current_mask_indices.sum().item()
            
            if num_remaining_masks == 0:
                break
                
            with capture_attention() as block_captured:
                block_output = model(
                    x[:, active_start:block_end], 
                    past_key_values=full_cache, 
                    use_cache=True, 
                    replace_position=replace_position
                )
            
            block_logits = block_output.logits
            block_predicted_tokens = torch.argmax(add_gumbel_noise(block_logits, temperature), dim=-1)
            nfe += 1

            x0, transfer_index = get_transfer_index(
                block_logits, block_predicted_tokens, remasking, current_mask_indices, 
                x[:, active_start:block_end], None, threshold
            )
            
            # 内部迭代的 Attention 校验保留了 Query 视角的微观判断
            if enable_attn_remask and num_remaining_masks > init_block_length/2 and len(block_captured) > 0:
                num_layers_to_fuse = 3 
                truncated_captured = block_captured[-num_layers_to_fuse:] if len(block_captured) >= num_layers_to_fuse else block_captured
                smoothed_attn = torch.stack(truncated_captured, dim=0).mean(dim=0)
                
                B, Q_len, KV_len = smoothed_attn.shape
                
                if active_start > prefix_start:
                    local_external_focus = smoothed_attn[:, :, prefix_start:active_start].max(dim=-1).values
                    
                    internal_start_idx = KV_len - Q_len
                    internal_attn_map = smoothed_attn[:, :, internal_start_idx:] 
                    eye = torch.eye(Q_len, device=internal_attn_map.device).bool().unsqueeze(0).expand(B, -1, -1)
                    
                    internal_attn_map = internal_attn_map.masked_fill(eye, 0)
                    internal_dependency = internal_attn_map.max(dim=-1).values
                    
                    epsilon_remask = 0.02 
                    is_unstable = internal_dependency > (local_external_focus * 10 + epsilon_remask)
                    
                    bad_attention_mask = is_unstable
                    original_transfer_index = transfer_index.clone()
                    transfer_index = transfer_index & (~bad_attention_mask)

                    if transfer_index.sum() == 0 and original_transfer_index.sum() > 0:
                        candidates = torch.nonzero(original_transfer_index[0]).squeeze(-1)
                        if len(candidates) > 0:
                            fallback_idx = candidates[0]
                            transfer_index[0, fallback_idx] = True
                
            x[:, active_start:block_end][transfer_index] = x0[transfer_index]
            
        nfe_history.append(nfe)
        prefix_start = 0

    return x, nfe, block_history, attention_history

# ==============================================================================
# 工具函数 & Harness
# ==============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self, model_path='', mask_id=126336, max_length=4096, batch_size=32, mc_num=128,
        is_check_greedy=True, steps=1024, gen_length=1024, block_length=1024,
        remasking='low_confidence', device="cuda", use_cache=False, threshold=None,
        factor=None, save_dir=None, show_speed=False, dual_cache=False,
        use_dynamic_block=False, max_block_length=DEFAULT_MAX_DYNAMIC_BLOCK,
        min_block_length=DEFAULT_MIN_DYNAMIC_BLOCK, smooth_window=3,
        seed=DEFAULT_SEED, **kwargs
    ):
        super().__init__()
        self.seed = int(seed)
        set_seed(self.seed)
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
                        generated_answer, nfe, block_boundaries, attn_history = generate_with_dynamic_dual_cache(
                            model=self.model, prompt=input_ids, steps=self.steps, gen_length=self.gen_length,
                            temperature=0, remasking=self.remasking, threshold=self.threshold,
                            mask_id=self.mask_id,
                            init_block_length=self.max_block_length,
                            min_block_length=self.min_block_length,
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
    set_seed(DEFAULT_SEED)
    cli_evaluate()
