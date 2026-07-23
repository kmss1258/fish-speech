from dataclasses import dataclass

import torch
from torch import Tensor

from fish_speech.models.text2semantic.llama import BaseTransformer, DualARTransformer


@dataclass(frozen=True, slots=True)
class GRPORolloutConfig:
    group_size: int
    max_new_tokens: int
    eos_token_id: int
    seed: int


@dataclass(frozen=True, slots=True)
class GRPORolloutRecord:
    full_sequence: Tensor
    prompt_length: int
    eos_token_id: int
    generated_frames: Tensor
    slow_tokens: Tensor
    behavior_slow_logprobs: Tensor
    behavior_fast_logprobs: Tensor
    slow_mask: Tensor
    fast_mask: Tensor
    eos: Tensor
    truncated: Tensor
    lengths: Tensor
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GRPORolloutScores:
    slow_logprobs: Tensor
    fast_logprobs: Tensor


@dataclass(frozen=True, slots=True)
class _FastSampleState:
    hidden: Tensor
    q0: Tensor
    generators: list[torch.Generator]


def rollout_dual_ar(
    model: DualARTransformer,
    prompt: Tensor,
    config: GRPORolloutConfig,
) -> GRPORolloutRecord:
    device = prompt.device
    batch, channels, prompt_length = prompt.shape
    group_size = config.group_size
    steps = config.max_new_tokens
    flat_batch = batch * group_size
    full = prompt.repeat_interleave(group_size, dim=0).new_zeros(
        flat_batch, channels, prompt_length + steps
    )
    full[:, :, :prompt_length] = prompt.repeat_interleave(group_size, dim=0)
    frames = prompt.new_zeros(flat_batch, steps, model.config.num_codebooks)
    slow_tokens = prompt.new_full((flat_batch, steps), config.eos_token_id)
    slow_logps = torch.zeros(flat_batch, steps, device=device)
    fast_logps = torch.zeros(
        flat_batch, steps, model.config.num_codebooks - 1, device=device
    )
    active = torch.ones(flat_batch, dtype=torch.bool, device=device)
    eos = torch.zeros(flat_batch, dtype=torch.bool, device=device)
    lengths = torch.zeros(flat_batch, dtype=torch.long, device=device)
    seeds = tuple(config.seed + i for i in range(flat_batch))
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]

    with torch.no_grad():
        for step in range(steps):
            prefix = full[:, :, : prompt_length + step]
            out = BaseTransformer.forward(model, prefix)
            hidden = model.fast_project_in(out.hidden_states[:, -1])
            slow_log_probs = _slow_log_probs(model, out.logits[:, -1], config)
            drawn = _draw_each(slow_log_probs, generators)
            raw_q0 = (drawn - model.config.semantic_begin_id).clamp(
                min=0, max=model.config.codebook_size - 1
            )
            fast_codes, fast_step_logps = _sample_fast(
                model, _FastSampleState(hidden, raw_q0, generators)
            )
            chosen_slow = slow_log_probs.gather(-1, drawn.unsqueeze(-1)).squeeze(-1)
            is_eos = drawn == config.eos_token_id
            trainable = active & ~is_eos
            full[:, 0, prompt_length + step] = drawn
            full[:, 1:, prompt_length + step] = torch.cat(
                [raw_q0.unsqueeze(-1), fast_codes], dim=-1
            )
            frames[:, step] = full[:, 1:, prompt_length + step]
            slow_tokens[:, step] = drawn
            slow_logps[:, step] = chosen_slow
            fast_logps[:, step] = fast_step_logps
            lengths += trainable.long()
            eos |= active & is_eos
            active &= ~is_eos

    truncated = active & (steps > 0)
    slow_mask = _active_mask(slow_tokens, config.eos_token_id)
    fast_mask = slow_mask.unsqueeze(-1).expand(-1, -1, model.config.num_codebooks - 1)
    shape3 = (batch, group_size, steps)
    shape4 = (batch, group_size, steps, model.config.num_codebooks - 1)
    return GRPORolloutRecord(
        full_sequence=full.reshape(batch, group_size, channels, prompt_length + steps),
        prompt_length=prompt_length,
        eos_token_id=config.eos_token_id,
        generated_frames=frames.reshape(
            batch, group_size, steps, model.config.num_codebooks
        ),
        slow_tokens=slow_tokens.reshape(shape3),
        behavior_slow_logprobs=slow_logps.reshape(shape3),
        behavior_fast_logprobs=fast_logps.reshape(shape4),
        slow_mask=slow_mask.reshape(shape3),
        fast_mask=fast_mask.reshape(shape4),
        eos=eos.reshape(batch, group_size),
        truncated=truncated.reshape(batch, group_size),
        lengths=lengths.reshape(batch, group_size),
        seeds=seeds,
    )


def score_dual_ar_rollout(
    model: DualARTransformer, record: GRPORolloutRecord
) -> GRPORolloutScores:
    batch, group_size, channels, total = record.full_sequence.shape
    flat = record.full_sequence.reshape(batch * group_size, channels, total)
    inputs = flat[:, :, :-1]
    labels = flat[:, :, 1:]
    outputs = model(
        inputs,
        labels=labels,
        key_padding_mask=torch.zeros_like(labels[:, 0], dtype=torch.bool),
    )
    slow = _slow_log_probs(
        model, outputs.token_logits, GRPORolloutConfig(1, 1, record.eos_token_id, 0)
    )
    slow = slow.gather(-1, labels[:, 0].unsqueeze(-1)).squeeze(-1)
    fast = _dense_fast_scores(model, outputs.codebook_logits, labels)
    start = record.prompt_length - 1
    stop = start + record.generated_frames.shape[2]
    return GRPORolloutScores(
        slow_logprobs=slow[:, start:stop].reshape(batch, group_size, stop - start),
        fast_logprobs=fast[:, start:stop].reshape(
            batch, group_size, stop - start, model.config.num_codebooks - 1
        ),
    )


def _slow_log_probs(
    model: DualARTransformer, logits: Tensor, config: GRPORolloutConfig
) -> Tensor:
    allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    allowed[model.config.semantic_begin_id : model.config.semantic_end_id + 1] = True
    if config.eos_token_id >= 0:
        allowed[config.eos_token_id] = True
    return logits.masked_fill(~allowed, -torch.inf).log_softmax(dim=-1)


def _draw_each(log_probs: Tensor, generators: list[torch.Generator]) -> Tensor:
    draws = [
        torch.multinomial(log_probs[i].exp(), 1, generator=generators[i])
        for i in range(log_probs.shape[0])
    ]
    return torch.cat(draws, dim=0)


def _sample_fast(
    model: DualARTransformer, state: _FastSampleState
) -> tuple[Tensor, Tensor]:
    codes = []
    logps = []
    previous = state.q0.unsqueeze(-1)
    for pos in range(1, model.config.num_codebooks):
        logits = _fast_logits(model, state.hidden, previous)
        log_probs = logits.log_softmax(dim=-1)
        drawn = _draw_each(log_probs, state.generators)
        codes.append(drawn)
        logps.append(log_probs.gather(-1, drawn.unsqueeze(-1)).squeeze(-1))
        previous = torch.cat([previous, drawn.unsqueeze(-1)], dim=-1)
    return torch.stack(codes, dim=-1), torch.stack(logps, dim=-1)


def _fast_logits(model: DualARTransformer, hidden: Tensor, previous: Tensor) -> Tensor:
    pos = previous.shape[1]
    x = torch.cat([hidden[:, None], model.fast_embeddings(previous)], dim=1)
    mask = model.causal_mask[None, None, : pos + 1, : pos + 1]
    freqs = model.fast_freqs_cis[: pos + 1]
    for layer in model.fast_layers:
        x = layer(x, freqs, mask)
    return model.fast_output(model.fast_norm(x[:, -1:])).squeeze(1)


def _active_mask(tokens: Tensor, eos_id: int) -> Tensor:
    seen_eos = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    masks = []
    for step in range(tokens.shape[1]):
        is_eos = tokens[:, step] == eos_id
        masks.append(~seen_eos & ~is_eos)
        seen_eos |= is_eos
    return torch.stack(masks, dim=1)


def _dense_fast_scores(
    model: DualARTransformer, codebook_logits: Tensor, labels: Tensor
) -> Tensor:
    token_labels = labels[:, 0]
    semantic = (token_labels >= model.config.semantic_begin_id) & (
        token_labels <= model.config.semantic_end_id
    )
    targets = labels[:, 1:].permute(0, 2, 1)[semantic]
    selected = (
        codebook_logits.log_softmax(dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    dense = torch.zeros(
        labels.shape[0],
        labels.shape[2],
        model.config.num_codebooks - 1,
        device=labels.device,
    )
    dense.masked_scatter_(
        semantic.unsqueeze(-1).expand_as(dense), selected[:, 1:].reshape(-1)
    )
    return dense
