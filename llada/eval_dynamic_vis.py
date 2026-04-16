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

import os
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
from contextlib import contextmanager
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM
import json
import time
import torch._dynamo

# 🌟【极其重要】：强制 matplotlib 使用无头后端，防止 Linux/HPC 环境下 NCCL 崩溃！
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 📊 绘图与数据导出工具
# ==============================================================================
def _sanitize_attention_history(history):
    """Convert attention values to finite floats so plotting/export never crashes."""
    sanitized = []
    for value in history or []:
        numeric = float(value)
        if not math.isfinite(numeric):
            numeric = 0.0
        sanitized.append(numeric)
    return sanitized


def _sanitize_attention_tensor(values: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with finite values before scoring."""
    return torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _compute_combined_scores(
    real_to_mask_attn: torch.Tensor,
    topk_ratio: float = 0.25,
    mean_weight: float = 0.2,
    topk_weight: float = 0.5,
    max_weight: float = 0.3,
) -> torch.Tensor:
    """Aggregate prefix-to-mask attention into a stronger per-position combined score."""
    if real_to_mask_attn.numel() == 0 or real_to_mask_attn.shape[0] == 0:
        return torch.zeros(real_to_mask_attn.shape[-1], device=real_to_mask_attn.device, dtype=torch.float32)

    real_to_mask_attn = _sanitize_attention_tensor(real_to_mask_attn)
    mean_scores = real_to_mask_attn.mean(dim=0)
    max_scores = real_to_mask_attn.max(dim=0).values
    topk_count = max(1, int(math.ceil(real_to_mask_attn.shape[0] * topk_ratio)))
    topk_scores = torch.topk(real_to_mask_attn, k=topk_count, dim=0).values.mean(dim=0)
    combined_scores = mean_weight * mean_scores + topk_weight * topk_scores + max_weight * max_scores
    return _sanitize_attention_tensor(combined_scores)


def _compute_score_bundle(real_to_mask_attn: torch.Tensor):
    if real_to_mask_attn.numel() > 0 and real_to_mask_attn.shape[0] > 0:
        real_to_mask_attn = _sanitize_attention_tensor(real_to_mask_attn)
        mean_scores = real_to_mask_attn.mean(dim=0)
        sum_scores = _sanitize_attention_tensor(real_to_mask_attn.sum(dim=0))
    else:
        mean_scores = torch.zeros(real_to_mask_attn.shape[-1], device=real_to_mask_attn.device, dtype=torch.float32)
        sum_scores = torch.zeros(real_to_mask_attn.shape[-1], device=real_to_mask_attn.device, dtype=torch.float32)
    return {
        "mean": mean_scores,
        "combined": sum_scores,
    }


def _find_code_start_index(generated_tokens):
    """Heuristic: find where the answer starts entering actual code rather than explanation text."""
    if not generated_tokens:
        return None

    prefixes = ["```python", "```", "\ndef ", "\nfrom ", "\nimport ", "\nclass ", "\nif __name__"]
    text = ""
    for idx, token in enumerate(generated_tokens):
        text += token
        if any(marker in text for marker in prefixes):
            return idx
    return None


def visualize_global_attention(score_history_bundle, block_boundaries, sample_id, generated_tokens=None, save_dir="global_attention_viz"):
    """画出全局序列生成的 mean/sum/code-only sum 趋势。"""
    history_mean = _sanitize_attention_history(score_history_bundle["mean"])
    history_combined = _sanitize_attention_history(score_history_bundle["combined"])
    history_code = _sanitize_attention_history(score_history_bundle["code_combined"])
    os.makedirs(save_dir, exist_ok=True)
    seq_len = len(history_combined)
    fig_width = max(18, seq_len * 0.15)
    plt.figure(figsize=(fig_width, 6))
    
    plt.plot(history_mean, color='#7f7f7f', linewidth=1.2, marker='.', markersize=3, label='Mean Score')
    plt.plot(history_combined, color='#d62728', linewidth=1.5, marker='.', markersize=4, label='Sum Score')
    plt.plot(history_code, color='#2ca02c', linewidth=1.5, marker='.', markersize=4, label='Code-only Sum')
    
    current_x = 0
    for i, size in enumerate(block_boundaries):
        current_x += size
        if i < len(block_boundaries) - 1:
            plt.axvline(x=current_x - 0.5, color='#2ca02c', linestyle='--', alpha=0.8, linewidth=1.5)
            
    plt.axvline(x=-100, color='#2ca02c', linestyle='--', alpha=0.8, linewidth=1.5, label='Dynamic Block Cut')
    plt.title(f"Global Attention Score Comparison - Sample #{sample_id}", fontsize=16, pad=15)
    plt.ylabel("Score", fontsize=12)
    
    if generated_tokens and len(generated_tokens) == seq_len:
        clean_tokens = [t.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') for t in generated_tokens]
        plt.xticks(ticks=range(len(clean_tokens)), labels=clean_tokens, rotation=90, fontsize=10, fontfamily='monospace')
        plt.tick_params(axis='x', pad=5)
    else:
        plt.xlabel("Absolute Generated Token Position", fontsize=12)
        
    plt.xlim(-0.5, seq_len - 0.5)
    max_val = max(history_mean + history_combined + history_code) if seq_len > 0 else 0.0
    plt.ylim(0, max(max_val * 1.1, 1.0))
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.savefig(os.path.join(save_dir, f"sample_{sample_id:04d}_global_attn.png"), dpi=150, bbox_inches='tight')
    plt.close()

def save_global_attention_json(score_history_bundle, block_boundaries, sample_id, generated_tokens=None, save_dir="global_attention_data"):
    """导出全局 mean/sum/code-only sum JSON 数据，并与实际 token 对齐。"""
    history_mean = _sanitize_attention_history(score_history_bundle["mean"])
    history_combined = _sanitize_attention_history(score_history_bundle["combined"])
    history_code = _sanitize_attention_history(score_history_bundle["code_combined"])
    os.makedirs(save_dir, exist_ok=True)
    cut_positions = set()
    current_x = 0
    for size in block_boundaries[:-1]:
        current_x += size
        cut_positions.add(current_x)
        
    data = {
        "sample_id": sample_id,
        "total_tokens": len(history_combined),
        "block_sizes": block_boundaries,
        "absolute_cut_positions": sorted(list(cut_positions)),
        "trajectory": []
    }
    
    for i in range(len(history_combined)):
        step_data = {
            "step_index": i,
            "mean_score": round(history_mean[i], 6),
            "combined_score": round(history_combined[i], 6),
            "sum_score": round(history_combined[i], 6),
            "code_combined_score": round(history_code[i], 6),
            "code_sum_score": round(history_code[i], 6),
            "focus_score": round(history_combined[i], 6),
            "is_cut_point": i in cut_positions
        }
        if generated_tokens and i < len(generated_tokens):
            step_data["token"] = generated_tokens[i]
        data["trajectory"].append(step_data)
        
    file_path = os.path.join(save_dir, f"sample_{sample_id:04d}_attn.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_layerwise_combined_score_json(layer_score_bundle, block_boundaries, sample_id, generated_tokens=None, save_dir="layers_attn"):
    """Export per-layer per-position mean/sum/code-only sum scores aligned with generated tokens."""
    os.makedirs(save_dir, exist_ok=True)
    if not layer_score_bundle["combined"]:
        return

    cut_positions = set()
    current_x = 0
    for size in block_boundaries[:-1]:
        current_x += size
        cut_positions.add(current_x)

    data = {
        "sample_id": sample_id,
        "total_tokens": len(layer_score_bundle["combined"][0]) if layer_score_bundle["combined"] else 0,
        "block_sizes": block_boundaries,
        "absolute_cut_positions": sorted(list(cut_positions)),
        "layers": []
    }

    for layer_idx in range(len(layer_score_bundle["combined"])):
        mean_history = _sanitize_attention_history(layer_score_bundle["mean"][layer_idx])
        combined_history = _sanitize_attention_history(layer_score_bundle["combined"][layer_idx])
        code_history = _sanitize_attention_history(layer_score_bundle["code_combined"][layer_idx])
        layer_data = {
            "layer_index": layer_idx,
            "trajectory": []
        }
        for step_idx in range(len(combined_history)):
            step_data = {
                "step_index": step_idx,
                "mean_score": round(mean_history[step_idx], 6),
                "combined_score": round(combined_history[step_idx], 6),
                "sum_score": round(combined_history[step_idx], 6),
                "code_combined_score": round(code_history[step_idx], 6),
                "code_sum_score": round(code_history[step_idx], 6),
                "is_cut_point": step_idx in cut_positions
            }
            if generated_tokens and step_idx < len(generated_tokens):
                step_data["token"] = generated_tokens[step_idx]
            layer_data["trajectory"].append(step_data)
        data["layers"].append(layer_data)

    file_path = os.path.join(save_dir, f"sample_{sample_id:04d}_layer_combined_scores.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_individual_layer_plots(layer_score_bundle, block_boundaries, generated_tokens=None, save_dir="layers_attn"):
    """为每一层生成 mean/sum/code-only sum 对照图，并与 token 对齐。"""
    os.makedirs(save_dir, exist_ok=True)
    if not layer_score_bundle["combined"] or len(layer_score_bundle["combined"]) == 0:
        return
        
    clean_tokens = [t.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') for t in generated_tokens] if generated_tokens else None
    seq_len = len(layer_score_bundle["combined"][0])
    fig_width = max(18, seq_len * 0.15)
    
    for layer_idx in enumerate(tqdm(layer_score_bundle["combined"], desc="    -> Plotting Layers", leave=False)):
        layer_idx = layer_idx[0]
        mean_history = _sanitize_attention_history(layer_score_bundle["mean"][layer_idx])
        combined_history = _sanitize_attention_history(layer_score_bundle["combined"][layer_idx])
        code_history = _sanitize_attention_history(layer_score_bundle["code_combined"][layer_idx])
        plt.figure(figsize=(fig_width, 5))
        plt.plot(mean_history, color='#7f7f7f', linewidth=1.0, marker='.', markersize=3, label='Mean')
        plt.plot(combined_history, color='#d62728', linewidth=1.4, marker='.', markersize=4, label='Sum')
        plt.plot(code_history, color='#2ca02c', linewidth=1.4, marker='.', markersize=4, label='Code-only Sum')
        
        current_x = 0
        for i, size in enumerate(block_boundaries):
            current_x += size
            if i < len(block_boundaries) - 1:
                plt.axvline(x=current_x - 0.5, color='#2ca02c', linestyle='--', alpha=0.8, linewidth=1.5)
                
        plt.title(f"Layer {layer_idx:02d} Score Comparison (All Prefix -> Next Block)", fontsize=14, pad=10)
        plt.ylabel("Score", fontsize=12)
        
        if clean_tokens and len(clean_tokens) == seq_len:
            plt.xticks(ticks=range(len(clean_tokens)), labels=clean_tokens, rotation=90, fontsize=10, fontfamily='monospace')
        
        plt.xlim(-0.5, seq_len - 0.5)
        
        # 保护：防止某些层全是 0 导致绘图崩溃
        max_val = max(mean_history + combined_history + code_history) if combined_history else 0.0
        plt.ylim(0, max(max_val * 1.1, 1.0))
            
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        
        # 保存到该 Case 专属的文件夹下
        plt.savefig(os.path.join(save_dir, f"layer_{layer_idx:02d}_attn.png"), dpi=150, bbox_inches='tight')
        plt.close()

def add_gumbel_noise(logits, temperature):
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
        F.softmax = original_F_softmax
        torch.softmax = original_torch_softmax
        captured_attentions = []

# ==============================================================================
# 核心生成逻辑
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
    mask_index: torch.Tensor,   
    x: torch.Tensor,            
    num_transfer_tokens,        
    threshold: float = None,
):
    x0 = predicted_tokens  

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | force_mask
        transfer_index = transfer_index & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    values, idx = torch.sort(confidence, dim=1, descending=True)
    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   
    select_sorted = cols < k_expanded                                            

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)  
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  

    return x0, transfer_index

def find_dependency_boundary(
    avg_attn: torch.Tensor,
    current_s: int,
    prefix_start: int,
    min_block_length: int,
    window_size: int,
    fallback_length: int,
    spike_multiplier: float = 4.0,
    relative_ratio: float = 4.0,
):
    if window_size <= min_block_length or prefix_start >= current_s:
        return min(fallback_length, window_size)

    real_to_mask_attn = avg_attn[prefix_start:current_s, current_s: current_s + window_size]
    if real_to_mask_attn.numel() == 0 or real_to_mask_attn.shape[0] == 0:
        return min(fallback_length, window_size)

    score_bundle = _compute_score_bundle(real_to_mask_attn)
    attention_scores = score_bundle["combined"]
    baseline = attention_scores.mean().item()
    variation = attention_scores.std(unbiased=False).item()
    absolute_threshold = baseline * spike_multiplier
    contrast_threshold = baseline + variation

    for k in range(min_block_length, window_size):
        current_score = attention_scores[k].item()
        if current_score <= 0:
            continue

        left_context = attention_scores[:k]
        local_baseline = left_context.mean().item() if left_context.numel() > 0 else baseline
        local_baseline = max(local_baseline, 1e-8)

        if current_score > max(absolute_threshold, contrast_threshold) and current_score / local_baseline >= relative_ratio:
            best_k = k + 1
            print(
                f"    🔭 [混合注意力切分] 视野 {window_size} | 词 {k} 处截断，"
                f"score={current_score:.4f}, baseline={baseline:.4f}, std={variation:.4f}, "
                f"local_ratio={current_score / local_baseline:.2f}"
            )
            return best_k

    return min(fallback_length, window_size)

@torch.no_grad()
def generate_with_dynamic_dual_cache(
    model, prompt, steps=128, gen_length=128, init_block_length=32, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, min_block_length=16,
    enable_dynamic_block=False, enable_attn_remask=False, tokenizer=None
): 
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert threshold is not None, "threshold must be set"

    generated_length = 0
    nfe_history = []  
    block_history = []
    score_history = {"mean": [], "combined": [], "code_combined": []}
    layer_score_history = None
    confirmed_generated_tokens = []
    warmup_blocks = 2
    tail_merge_threshold = max(4, min_block_length)
    
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
        window_size = min(int(0.25 * gen_length), remaining_length)
        fallback_length = min(init_block_length, remaining_length)
        dynamic_enabled_this_step = enable_dynamic_block and len(block_history) >= warmup_blocks
        
        if len(captured) > 0:
            # 初始化多层追踪列表
            if layer_score_history is None:
                layer_score_history = {
                    "mean": [[] for _ in range(len(captured))],
                    "combined": [[] for _ in range(len(captured))],
                    "code_combined": [[] for _ in range(len(captured))],
                }

            B, L_seq, S_seq = captured[0].shape
            device = captured[0].device
            I = torch.eye(L_seq, device=device).unsqueeze(0).expand(B, -1, -1)

            rollout = None
            selected_layer_indices = [8, 11, 28, 29, 30]
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
                avg_attn = None
                block_length = fallback_length
            else:
                avg_attn = rollout[0]

            if avg_attn is not None and dynamic_enabled_this_step:
                block_length = find_dependency_boundary(
                    avg_attn=avg_attn,
                    current_s=current_s,
                    prefix_start=prefix_start,
                    min_block_length=min_block_length,
                    window_size=window_size,
                    fallback_length=fallback_length,
                )
            else:
                block_length = fallback_length

            tail_length = remaining_length - block_length
            if 0 < tail_length <= tail_merge_threshold:
                block_length = remaining_length

            code_start_idx = _find_code_start_index(confirmed_generated_tokens)
            code_prefix_abs = current_s if code_start_idx is None else prompt_length + code_start_idx

            # 1. 记录全局 (Rollout) 的 mean / combined / code-only combined
            if avg_attn is not None:
                real_to_mask_attn = avg_attn[prefix_start:current_s, current_s : current_s + probe_limit]
                score_bundle = _compute_score_bundle(real_to_mask_attn)
                if code_prefix_abs < current_s:
                    code_score_bundle = _compute_score_bundle(
                        avg_attn[code_prefix_abs:current_s, current_s : current_s + probe_limit]
                    )
                    code_combined_scores = code_score_bundle["combined"]
                else:
                    code_combined_scores = torch.zeros(probe_limit, device=device, dtype=torch.float32)
            else:
                score_bundle = {
                    "mean": torch.zeros(probe_limit, device=device, dtype=torch.float32),
                    "combined": torch.zeros(probe_limit, device=device, dtype=torch.float32),
                }
                code_combined_scores = torch.zeros(probe_limit, device=device, dtype=torch.float32)

            score_history["mean"].extend(score_bundle["mean"][:block_length].cpu().float().numpy().tolist())
            score_history["combined"].extend(score_bundle["combined"][:block_length].cpu().float().numpy().tolist())
            score_history["code_combined"].extend(code_combined_scores[:block_length].cpu().float().numpy().tolist())
            
            # 2. 记录每一层单独的 mean / combined / code-only combined
            for layer_idx, W in enumerate(captured):
                layer_attn = W[0] # 取出 Batch 0
                layer_rtm = layer_attn[prefix_start:current_s, current_s : current_s + probe_limit]
                layer_score_bundle = _compute_score_bundle(layer_rtm)
                if code_prefix_abs < current_s:
                    layer_code_bundle = _compute_score_bundle(
                        layer_attn[code_prefix_abs:current_s, current_s : current_s + probe_limit]
                    )
                    layer_code_scores = layer_code_bundle["combined"]
                else:
                    layer_code_scores = torch.zeros(probe_limit, device=device, dtype=torch.float32)

                layer_score_history["mean"][layer_idx].extend(layer_score_bundle["mean"][:block_length].cpu().float().numpy().tolist())
                layer_score_history["combined"][layer_idx].extend(layer_score_bundle["combined"][:block_length].cpu().float().numpy().tolist())
                layer_score_history["code_combined"][layer_idx].extend(layer_code_scores[:block_length].cpu().float().numpy().tolist())
                
        else:
            block_length = fallback_length 

        block_history.append(block_length)
        block_end = current_s + block_length
        generated_length += block_length
        
        # 内循环降噪
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
            
            if num_remaining_masks == 0: break
                
            with capture_attention() as block_captured:
                block_output = model(x[:, active_start:block_end], past_key_values=full_cache, use_cache=True, replace_position=replace_position)
            
            block_logits = block_output.logits
            block_predicted_tokens = torch.argmax(add_gumbel_noise(block_logits, temperature), dim=-1)
            nfe += 1

            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, current_mask_indices, x[:, active_start:block_end], None, threshold)
            
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
                            transfer_index[0, candidates[0]] = True
                
            x[:, active_start:block_end][transfer_index] = x0[transfer_index]
            
        nfe_history.append(nfe)
        if tokenizer is not None:
            new_token_ids = x[0, active_start:block_end]
            for tid in new_token_ids:
                confirmed_generated_tokens.append(
                    tokenizer.decode([tid.item()], clean_up_tokenization_spaces=False)
                )
        prefix_start = 0

    return x, nfe, block_history, score_history, layer_score_history

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
        use_dynamic_block=False, max_block_length=64, min_block_length=4, smooth_window=3,
        **kwargs
    ):
        super().__init__()
        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = False 
        
        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
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
        
        # 默认基础存放目录
        self.base_save_dir = save_dir if save_dir else "output_analysis"
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        self.use_dynamic_block = use_dynamic_block
        self.max_block_length = int(max_block_length) 
        self.min_block_length = int(min_block_length) 

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
        return self.model(batch).logits[:, :batch.shape[1]]

    def loglikelihood(self, requests):
        raise NotImplementedError
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output, num_tokens, num_nfe, processed_count, total_count = [], 0, 0, 0, 0
        os.makedirs(self.base_save_dir, exist_ok=True)
        save_path = os.path.join(self.base_save_dir, f'rank_{getattr(self, "rank", 0)}.jsonl')

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

            if self.use_cache and self.dual_cache:
                # 🌟 调用接收最新的 5 个返回值
                generated_answer, nfe, block_boundaries, attn_history, attn_history_per_layer = generate_with_dynamic_dual_cache(
                    model=self.model, prompt=input_ids, steps=self.steps, gen_length=self.gen_length,
                    temperature=0, remasking=self.remasking, threshold=self.threshold, mask_id=self.mask_id,
                    enable_dynamic_block=self.use_dynamic_block, tokenizer=self.tokenizer
                )
            else:
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)

            batched_generated_answer = []
            for i in range(len(generated_answer)):
                gen_str = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                for stop_seq in stop_tokens:
                    if stop_seq in gen_str: gen_str = gen_str.split(stop_seq)[0]
                gen_ids = torch.tensor(self.tokenizer(gen_str)["input_ids"])
                batched_generated_answer.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))

            output.extend(batched_generated_answer)
            with open(save_path, 'a', encoding='utf-8') as f:
                for ans in batched_generated_answer: f.write(json.dumps(ans, ensure_ascii=False) + '\n')

            if block_boundaries:
                global_total_blocks += len(block_boundaries)
                global_total_block_length += sum(block_boundaries)

            for i in range(len(batched_generated_answer)):
                total_count += 1
                
                # ==================================================
                # 🌟 文件夹分配与数据可视化核心区
                # ==================================================
                gen_tokens_list = []
                if 'attn_history' in locals() and attn_history:
                    actual_len = len(attn_history["combined"])
                    gen_ids_for_plot = generated_answer[i][input_ids.shape[1] : input_ids.shape[1] + actual_len]
                    
                    for tid in gen_ids_for_plot:
                        token_str = self.tokenizer.decode([tid.item()], clean_up_tokenization_spaces=False)
                        gen_tokens_list.append(token_str)
                
                    # 🌟 为该题目创建独立文件夹
                    case_dir = os.path.join(self.base_save_dir, f"sample_{total_count:04d}")
                    os.makedirs(case_dir, exist_ok=True)
                    
                    # 1. 导出 JSON 到 case 文件夹
                    save_global_attention_json(
                        score_history_bundle=attn_history, 
                        block_boundaries=block_boundaries, 
                        sample_id=total_count,
                        generated_tokens=gen_tokens_list,
                        save_dir=case_dir
                    )

                    # 1.1 导出每层每个位置的 combined score，并与 token 对齐
                    save_layerwise_combined_score_json(
                        layer_score_bundle=attn_history_per_layer,
                        block_boundaries=block_boundaries,
                        sample_id=total_count,
                        generated_tokens=gen_tokens_list,
                        save_dir=case_dir
                    )
                    
                    # 2. 导出全局（Rollout）长图 到 case 文件夹
                    visualize_global_attention(
                        score_history_bundle=attn_history, 
                        block_boundaries=block_boundaries, 
                        sample_id=total_count,
                        generated_tokens=gen_tokens_list,
                        save_dir=case_dir
                    )

                    # 3. 🌟 遍历并保存所有层的注意力长图 到 case 文件夹
                    save_individual_layer_plots(
                        layer_score_bundle=attn_history_per_layer,
                        block_boundaries=block_boundaries,
                        generated_tokens=gen_tokens_list,
                        save_dir=case_dir
                    )

                print('=' * 60)
                if block_boundaries:
                    block_sizes = [s for s in block_boundaries]
                    print(f"[Question {total_count}] Dynamic Blocks: {block_sizes}")
                else:
                    print(f"[Question {total_count}] Static Block Size: {self.block_length}")

                preview = batched_generated_answer[i][:150].replace('\n', '\\n') + ("..." if len(batched_generated_answer[i]) > 150 else "")
                print(f" └─ Preview: {preview}")
                print('=' * 60)

        if total_count > 0:
            print("=" * 40)
            print("✅ Generation Complete. Ready for evaluation.")
        return output

if __name__ == "__main__":
    cli_evaluate()
