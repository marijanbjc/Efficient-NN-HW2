"""Помощники для LoRA fine-tuning: сравнение стандартного loss HF и CCE (аналог рис. 4 статьи).

- загрузка модели с LoRA и выбранным способом подсчёта loss;
- оптимизатор и расписание learning rate;
- цикл обучения с логом loss, пиковой памяти и токенов в секунду в CSV;
- поиск максимального батча до OOM.

Важно: `cce_patch` подменяет forward класса модели в модуле transformers ГЛОБАЛЬНО, то есть
действует на весь процесс Python. Поэтому baseline и CCE нужно запускать в разных процессах
(в ноутбуке — перезапуская ядро между прогонами).
"""

from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup

from preprocessing import IGNORE_INDEX

LossImpl = Literal["hf", "cce"]
OptimizerName = Literal["adamw", "adamw_8bit", "adafactor"]


@dataclass
class FinetuneConfig:
    """Все настройки одного прогона. Сохраняется рядом с логом, чтобы прогоны можно было сравнить."""

    model_name: str = "google/gemma-2-2b"
    loss_impl: LossImpl = "hf"                # "hf" — стандартный loss HF, "cce" — патч CCE
    cce_impl: str = "cce"                     # вариант CCE: "cce", "cce_kahan_full_c", ...
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # Обучение
    optimizer: OptimizerName = "adamw"
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 20
    num_steps: int = 300                      # число шагов оптимизатора
    batch_size: int = 8                       # примеров на один forward
    grad_accum: int = 1                       # шагов накопления на один шаг оптимизатора
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    seed: int = 0
    # Логирование
    log_every: int = 1
    log_path: str = "results/finetune_hf.csv"


# --------------------------------------------------------------------------- подготовка


def set_seed(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_for_finetune(
    cfg: FinetuneConfig,
    device: torch.device | str = "cuda",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Загружает модель в bf16, при необходимости патчит её под CCE и навешивает LoRA."""
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Для Gemma 2 HF рекомендует eager-внимание при обучении (из-за softcap в attention).
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )

    if cfg.loss_impl == "cce":
        from cut_cross_entropy.transformers import cce_patch

        # Патч заменяет «lm_head + cross-entropy» на CCE; logits в выходе модели станут None.
        model = cce_patch(model, impl=cfg.cce_impl, reduction="mean")

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # нужно, чтобы градиент дошёл до LoRA при чекпоинтинге
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    return model, tokenizer


def build_optimizer(model: torch.nn.Module, cfg: FinetuneConfig) -> torch.optim.Optimizer:
    """Создаёт оптимизатор только по обучаемым параметрам (адаптерам LoRA)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(params, lr=cfg.lr, weight_decay=cfg.weight_decay, scale_parameter=False, relative_step=False)
    raise ValueError(f"Неизвестный оптимизатор: {cfg.optimizer}")


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """(обучаемые параметры, все параметры)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def infinite_batches(dataloader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    """Бесконечно перебирает DataLoader по эпохам (обучение задаётся числом шагов, а не эпох)."""
    while True:
        yield from dataloader


@torch.no_grad()
def evaluate_loss(
    model: PreTrainedModel,
    batch: dict[str, torch.Tensor],
    device: torch.device | str = "cuda",
) -> float:
    """Loss модели на одном фиксированном батче без обновления весов.

    До обучения LoRA-адаптеры нулевые (матрица B инициализируется нулями), поэтому модель
    совпадает с исходной. Значит, loss на одном и том же батче в прогоне "hf" и в прогоне "cce"
    должен совпасть: так проверяется, что патч CCE считает тот же loss, что и HF.
    """
    was_training = model.training
    model.eval()
    out = model(**{k: v.to(device) for k, v in batch.items()})
    loss = float(out.loss)
    model.train(was_training)
    return loss


# --------------------------------------------------------------------------- обучение


def train(
    model: PreTrainedModel,
    dataloader: DataLoader,
    cfg: FinetuneConfig,
    device: torch.device | str = "cuda",
) -> pd.DataFrame:
    """Цикл обучения. Каждые cfg.log_every шагов пишет строку в CSV сразу на диск,
    чтобы при падении ядра сохранилась уже пройденная часть кривой.

    Колонки лога: step, loss, lr, peak_mem_mb, step_time_s, tokens_per_s.
    loss — средний по микробатчам шага; peak_mem_mb — пик за шаг (включая веса модели).
    """
    set_seed(cfg.seed)
    model.train()
    optimizer = build_optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.num_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]

    log_path = Path(cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["step", "loss", "lr", "peak_mem_mb", "step_time_s", "tokens_per_s"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(columns)

    batches = infinite_batches(dataloader)
    rows: list[dict[str, float]] = []

    for step in range(1, cfg.num_steps + 1):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss_sum = 0.0
        n_loss_tokens = 0
        for _ in range(cfg.grad_accum):
            batch = {k: v.to(device) for k, v in next(batches).items()}
            out = model(**batch)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            loss_sum += float(loss)
            # Токены с loss после сдвига: метка позиции 0 не используется.
            n_loss_tokens += int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
            del out, loss  # сразу отпускаем logits baseline, иначе они доживут до следующего forward

        torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        step_time = time.perf_counter() - t0

        if step % cfg.log_every == 0 or step == cfg.num_steps:
            row = {
                "step": step,
                "loss": loss_sum,
                "lr": scheduler.get_last_lr()[0],
                "peak_mem_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "step_time_s": step_time,
                "tokens_per_s": n_loss_tokens / step_time,
            }
            rows.append(row)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in columns])
            if not math.isfinite(row["loss"]):
                raise RuntimeError(f"loss стал {row['loss']} на шаге {step}")

    # Рядом с логом сохраняем конфиг прогона.
    pd.Series(asdict(cfg)).to_json(log_path.with_suffix(".config.json"), force_ascii=False, indent=1)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- максимальный батч


def fits_in_memory(
    model: PreTrainedModel,
    make_batch: Callable[[int], dict[str, torch.Tensor]],
    batch_size: int,
    device: torch.device | str = "cuda",
) -> bool:
    """Пробует один шаг forward + backward с данным размером батча. True, если не было OOM."""
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    try:
        batch = {k: v.to(device) for k, v in make_batch(batch_size).items()}
        out = model(**batch)
        out.loss.backward()
        ok = True
    except torch.cuda.OutOfMemoryError:
        ok = False
    finally:
        out = batch = None  # noqa: F841 — отпускаем ссылки до empty_cache
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return ok


def find_max_batch_size(
    model: PreTrainedModel,
    make_batch: Callable[[int], dict[str, torch.Tensor]],
    start: int = 1,
    limit: int = 1024,
    device: torch.device | str = "cuda",
) -> int:
    """Ищет максимальный батч без OOM: удвоение до первого OOM, затем двоичный поиск.

    make_batch(bs) должна возвращать батч фиксированной длины последовательности,
    чтобы сравнение baseline и CCE было честным. Возвращает 0, если не влезает даже start.
    """
    if not fits_in_memory(model, make_batch, start, device):
        return 0
    lo = start
    hi = start * 2
    while hi <= limit and fits_in_memory(model, make_batch, hi, device):
        lo, hi = hi, hi * 2
    hi = min(hi, limit + 1)
    # Инвариант: lo влезает, hi — нет (или превышает limit).
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits_in_memory(model, make_batch, mid, device):
            lo = mid
        else:
            hi = mid
    return lo


def make_random_batch_factory(
    vocab_size: int,
    seq_len: int,
) -> Callable[[int], dict[str, torch.Tensor]]:
    """Фабрика батчей из случайных токенов фиксированной длины для find_max_batch_size.

    Для поиска предела памяти содержимое токенов не важно, важны только размеры.
    """

    def make_batch(batch_size: int) -> dict[str, torch.Tensor]:
        ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}

    return make_batch
