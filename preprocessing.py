"""Подготовка данных.

- чтение CSV с примерами и сборка текста;
- токенизация токенизатором Gemma, маска -100 для токенов без loss;
- DataLoader с паддингом для обучения;
- получение входов последнего слоя для бенчмарка: E (скрытые состояния перед lm_head),
  C (lm_head.weight), targets и softcap.

Соглашение о метках: `labels` здесь НЕ сдвинуты, как принято в HF (модель сама сдвигает их
внутри forward). Сдвиг делаем вручную только в `extract_last_layer_inputs`, где позиция t
предсказывает токен t+1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

IGNORE_INDEX: int = -100

# Шаблон промпта в стиле Alpaca (так же размечены данные, на которых авторы делали fine-tuning).
PROMPT_WITH_INPUT: str = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT: str = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


# --------------------------------------------------------------------------- данные


@dataclass
class Example:
    """Один пример: промпт (без loss) и ответ (с loss). Для обычного текста prompt пустой."""

    prompt: str
    response: str


@dataclass
class TokenizedExample:
    """Токенизированный пример. labels совпадают с input_ids, кроме позиций без loss (-100)."""

    input_ids: list[int]
    labels: list[int]


def load_examples(
        csv_path: str | Path,
        instruction_col: str = "instruction",
        input_col: str = "input",
        output_col: str = "output",
        text_col: str = "text",
        max_examples: int | None = None,
) -> list[Example]:
    """Читает CSV и собирает список примеров.

    Поддерживаются два формата:
    - инструкционный: колонки instruction, (необязательно) input, output → промпт по шаблону Alpaca;
    - обычный текст: одна колонка text → весь текст идёт в loss.
    """
    df = pd.read_csv(csv_path)
    if max_examples is not None:
        df = df.head(max_examples)

    examples: list[Example] = []
    if instruction_col in df.columns and output_col in df.columns:
        has_input = input_col in df.columns
        for row in df.itertuples(index=False):
            instruction = str(getattr(row, instruction_col))
            extra = getattr(row, input_col) if has_input else None
            # Пустой input в CSV читается как NaN → используем шаблон без input.
            if extra is None or pd.isna(extra) or str(extra).strip() == "":
                prompt = PROMPT_NO_INPUT.format(instruction=instruction)
            else:
                prompt = PROMPT_WITH_INPUT.format(instruction=instruction, input=str(extra))
            examples.append(Example(prompt=prompt, response=str(getattr(row, output_col))))
    elif text_col in df.columns:
        examples = [Example(prompt="", response=str(t)) for t in df[text_col]]
    else:
        raise ValueError(
            f"Не нашёл нужных колонок в {csv_path}. Есть: {list(df.columns)}. "
            f"Нужны ({instruction_col}, {output_col}[, {input_col}]) или {text_col}."
        )
    return examples


def tokenize_example(
        tokenizer: PreTrainedTokenizerBase,
        example: Example,
        max_length: int,
        mask_prompt: bool = True,
) -> TokenizedExample:
    """Токенизирует пример: [BOS] + промпт + ответ + [EOS], обрезая до max_length.

    Если mask_prompt=True, токены промпта получают метку -100 и не участвуют в loss.
    BOS тоже всегда маскируется: его никто не предсказывает.
    """
    prompt_ids: list[int] = tokenizer(example.prompt, add_special_tokens=False)["input_ids"] if example.prompt else []
    response_ids: list[int] = tokenizer(example.response, add_special_tokens=False)["input_ids"]

    bos: list[int] = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos: list[int] = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = bos + prompt_ids + response_ids + eos
    prompt_labels = [IGNORE_INDEX] * len(prompt_ids) if mask_prompt else list(prompt_ids)
    labels = [IGNORE_INDEX] * len(bos) + prompt_labels + response_ids + eos

    return TokenizedExample(input_ids=input_ids[:max_length], labels=labels[:max_length])


def tokenize_examples(
        tokenizer: PreTrainedTokenizerBase,
        examples: list[Example],
        max_length: int,
        mask_prompt: bool = True,
) -> list[TokenizedExample]:
    """Токенизирует все примеры и выбрасывает те, где после обрезки не осталось ни одной метки."""
    out: list[TokenizedExample] = []
    for ex in examples:
        tok = tokenize_example(tokenizer, ex, max_length, mask_prompt)
        # Нужна хотя бы одна метка не на первой позиции: после сдвига метка позиции 0 пропадает.
        if any(lbl != IGNORE_INDEX for lbl in tok.labels[1:]):
            out.append(tok)
    return out


def count_loss_tokens(tokenized: list[TokenizedExample]) -> int:
    """Сколько токенов реально участвует в loss (после сдвига на 1)."""
    return sum(sum(lbl != IGNORE_INDEX for lbl in t.labels[1:]) for t in tokenized)


# --------------------------------------------------------------------------- обучение


class TokenizedDataset(Dataset):
    """Обёртка списка TokenizedExample в torch Dataset."""

    def __init__(self, items: list[TokenizedExample]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TokenizedExample:
        return self.items[idx]


def collate_with_padding(batch: list[TokenizedExample], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Склеивает примеры в батч, дополняя справа до самой длинной последовательности.

    Паддинг: input_ids → pad_token_id, labels → -100, attention_mask → 0.
    """
    max_len = max(len(x.input_ids) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x.input_ids)
        input_ids[i, :n] = torch.tensor(x.input_ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(x.labels, dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def make_train_dataloader(
        tokenized: list[TokenizedExample],
        batch_size: int,
        pad_token_id: int,
        shuffle: bool = True,
        seed: int = 0,
) -> DataLoader:
    """DataLoader для обучения. Сид фиксирует порядок, чтобы у baseline и CCE были одни и те же батчи."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TokenizedDataset(tokenized),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda b: collate_with_padding(b, pad_token_id),
        drop_last=True,
    )


# --------------------------------------------------------------------------- вход последнего слоя


@dataclass
class LastLayerInputs:
    """Всё, что нужно функции loss последнего слоя.

    E:       (N, D) скрытые состояния перед lm_head, только позиции с loss;
    C:       (V, D) веса lm_head (у Gemma связаны с входными эмбеддингами);
    targets: (N,)   правильный следующий токен для каждой строки E;
    softcap: значение final_logit_softcapping (у Gemma 2 = 30) или None.
    """

    E: torch.Tensor
    C: torch.Tensor
    targets: torch.Tensor
    softcap: float | None

    @property
    def n_tokens(self) -> int:
        return self.E.shape[0]

    def to(self, device: torch.device | str) -> LastLayerInputs:
        return LastLayerInputs(self.E.to(device), self.C.to(device), self.targets.to(device), self.softcap)

    def save(self, path: str | Path) -> None:
        torch.save({"E": self.E.cpu(), "C": self.C.cpu(), "targets": self.targets.cpu(), "softcap": self.softcap}, path)

    @staticmethod
    def load(path: str | Path, device: torch.device | str = "cpu") -> LastLayerInputs:
        d = torch.load(path, map_location=device)
        return LastLayerInputs(d["E"], d["C"], d["targets"], d["softcap"])


def load_model_and_tokenizer(
        model_name: str,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Загружает модель в режиме инференса (eval, без градиентов) и её токенизатор."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Для Gemma 2 HF рекомендует eager-внимание (из-за softcap в attention).
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager")
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def extract_last_layer_inputs(
        model: PreTrainedModel,
        tokenized: list[TokenizedExample],
        n_tokens: int,
        device: torch.device | str = "cuda",
        shuffle_seed: int | None = 0,
) -> LastLayerInputs:
    """Прогоняет примеры через бэкбон и собирает ровно n_tokens строк (E, target).

    Каждый пример прогоняется отдельно (батч из одной последовательности), поэтому паддинг
    не нужен. Внутри примера делаем сдвиг: скрытое состояние позиции t ↔ токен t+1.
    Позиции с меткой -100 выбрасываются (как в приложении B статьи).
    """
    order = list(range(len(tokenized)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)

    # get_decoder() — бэкбон без lm_head (model.model); его last_hidden_state уже прошёл финальную нормализацию,
    # и у Gemma 2 к нему напрямую применяется lm_head (затем softcap).
    backbone = model.get_decoder()

    chunks_e: list[torch.Tensor] = []
    chunks_t: list[torch.Tensor] = []
    collected = 0
    for idx in order:
        ex = tokenized[idx]
        input_ids = torch.tensor([ex.input_ids], device=device)
        hidden = backbone(input_ids=input_ids).last_hidden_state[0]  # (T, D)

        labels = torch.tensor(ex.labels, device=device)
        e = hidden[:-1]  # позиции 0..T-2
        t = labels[1:]  # их правильные следующие токены
        keep = t != IGNORE_INDEX
        chunks_e.append(e[keep])
        chunks_t.append(t[keep])

        collected += int(keep.sum())
        if collected >= n_tokens:
            break

    if collected < n_tokens:
        raise ValueError(f"Данных хватило только на {collected} токенов с loss, а нужно {n_tokens}.")

    E = torch.cat(chunks_e)[:n_tokens].contiguous()
    targets = torch.cat(chunks_t)[:n_tokens].contiguous()
    # clone, чтобы C не держала ссылку на модель и её можно было удалить.
    C = model.get_output_embeddings().weight.detach().clone()
    softcap: float | None = getattr(model.config, "final_logit_softcapping", None)
    return LastLayerInputs(E=E, C=C, targets=targets, softcap=softcap)
