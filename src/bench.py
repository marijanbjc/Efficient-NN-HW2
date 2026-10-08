"""Замер пиковой памяти и времени для функций loss последнего слоя (аналог таблицы 1 статьи).

Любой метод — это функция loss_fn(E, C, targets, softcap) -> скаляр (среднее по токенам).

Режимы и что считается памятью (пик сверх того, что уже было выделено перед замеряемой фазой):
- "loss":      только forward (с включёнными градиентами, как при обучении);
- "grad":      только backward; forward делается заранее и в замер не входит;
- "loss+grad": forward + backward вместе.

В режимах с backward в пик входят сами ∇E и ∇C: это нижняя граница памяти для любого
метода ("Lower bound" в таблице 1).
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd
import torch

from .preprocessing import LastLayerInputs

LossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float | None], torch.Tensor
]
Mode = Literal["loss", "grad", "loss+grad"]

MB: float = 1024.0**2


@dataclass
class BenchResult:
    """Результат замера одного метода в одном режиме."""

    method: str
    mode: str
    peak_mem_mb: float  # пик памяти сверх выделенной до замеряемой фазы
    time_ms: float  # среднее время по повторам
    time_ms_std: float  # стандартное отклонение времени
    status: str = "ok"  # "ok" или "OOM"


# --------------------------------------------------------------------------- вспомогательное


def free_memory() -> None:
    """Освобождает всё, что можно: мусор Python и кэш аллокатора CUDA."""
    gc.collect()
    torch.cuda.empty_cache()


def is_oom(err: BaseException) -> bool:
    """True, если исключение означает нехватку памяти GPU.

    На MIG-слайсе PyTorch при OOM иногда падает не с OutOfMemoryError, а с
    RuntimeError "NVML_SUCCESS == r INTERNAL ASSERT FAILED": аллокатор пытается спросить
    у NVML объём свободной памяти, а NVML на MIG этот запрос не поддерживает.
    """
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err)
    return isinstance(err, RuntimeError) and (
        "NVML_SUCCESS" in msg or "out of memory" in msg
    )


def make_leaf_inputs(
    inputs: LastLayerInputs, requires_grad: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Возвращает E и C как листовые тензоры (без истории), готовые копить .grad."""
    E = inputs.E.detach().requires_grad_(requires_grad)
    C = inputs.C.detach().requires_grad_(requires_grad)
    return E, C


def _run_once(
    loss_fn: LossFn,
    E: torch.Tensor,
    C: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None,
    mode: Mode,
) -> tuple[float, float]:
    """Один замер. Возвращает (пик памяти в МБ, время в мс) для замеряемой фазы."""
    E.grad = None
    C.grad = None
    # Только сборка мусора, без torch.cuda.empty_cache(): иначе каждый повтор заново выпрашивает
    # у драйвера гигабайты через cudaMalloc, и это время попадает в замер (сильнее всего у Baseline).
    # Пиковую память это не искажает: max_memory_allocated считает выделенные тензоры, а не кэш.
    gc.collect()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    if mode == "grad":
        # Forward вне замера: в памяти остаётся то, что метод сохранил для backward.
        loss = loss_fn(E, C, targets, softcap)

        # Ждём на ЦПУ и ничего не делаем, пока ГПУ не досчитает лосс
        # т.к. в данном режиме замеряется только подсчёт градиентов
        torch.cuda.synchronize()

        # Отдаёт сколько на данный момент занято памяти на ГПУ
        base = torch.cuda.memory_allocated()

        # Сбрасываем счётчик пиковой памяти, который произошёл до текущего момента
        torch.cuda.reset_peak_memory_stats()
        start.record()
        loss.backward()
        end.record()
    else:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start.record()
        loss = loss_fn(E, C, targets, softcap)
        if mode == "loss+grad":
            loss.backward()
        end.record()

    torch.cuda.synchronize()

    peak_mb = (torch.cuda.max_memory_allocated() - base) / MB
    elapsed_ms = start.elapsed_time(end)

    del loss
    E.grad = None
    C.grad = None
    return peak_mb, elapsed_ms


# --------------------------------------------------------------------------- замеры


def benchmark_method(
    name: str,
    loss_fn: LossFn,
    inputs: LastLayerInputs,
    mode: Mode,
    n_warmup: int = 2,
    n_repeats: int = 5,
) -> BenchResult:
    """Замер одного метода: прогрев (компиляция torch.compile, автотюнинг Triton), затем повторы.

    Память берётся максимальной по повторам, время — среднее.
    """
    E, C = make_leaf_inputs(inputs)
    try:
        for _ in range(n_warmup):
            _run_once(loss_fn, E, C, inputs.targets, inputs.softcap, mode)

        mems: list[float] = []
        times: list[float] = []
        for _ in range(n_repeats):
            m, t = _run_once(loss_fn, E, C, inputs.targets, inputs.softcap, mode)
            mems.append(m)
            times.append(t)
    except Exception as err:
        if not is_oom(err):
            raise
        del err  # трейсбек держит ссылки на тензоры упавшего прогона
        free_memory()
        return BenchResult(name, mode, math.nan, math.nan, math.nan, status="OOM")
    finally:
        del E, C
        free_memory()

    mean_t = sum(times) / len(times)
    std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
    return BenchResult(name, mode, max(mems), mean_t, std_t)


def benchmark_all(
    methods: dict[str, LossFn],
    inputs: LastLayerInputs,
    modes: tuple[Mode, ...] = ("loss", "grad", "loss+grad"),
    n_warmup: int = 2,
    n_repeats: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Прогоняет все методы во всех режимах и возвращает длинную таблицу (строка = метод × режим)."""
    rows: list[dict] = []
    for name, fn in methods.items():
        for mode in modes:
            res = benchmark_method(name, fn, inputs, mode, n_warmup, n_repeats)
            rows.append(asdict(res))
            if verbose:
                print(
                    f"{name:<28} {mode:<10} {res.peak_mem_mb:>10.1f} MB {res.time_ms:>9.1f} ms  {res.status}"
                )
    return pd.DataFrame(rows)


def to_paper_table(df: pd.DataFrame) -> pd.DataFrame:
    """Переводит длинную таблицу в широкую, как таблица 1: колонки (режим, Memory/Time)."""
    wide = df.pivot(index="method", columns="mode", values=["peak_mem_mb", "time_ms"])
    wide = wide.swaplevel(axis=1).sort_index(axis=1)
    order = [
        m
        for m in ("loss", "grad", "loss+grad")
        if m in wide.columns.get_level_values(0)
    ]
    return wide.reindex(columns=order, level=0).reindex(df["method"].unique())


def lower_bound_mb(inputs: LastLayerInputs) -> float:
    """Нижняя граница памяти для режимов с backward: размер ∇E + ∇C (в типе самих E и C)."""
    return (
        inputs.E.numel() * inputs.E.element_size()
        + inputs.C.numel() * inputs.C.element_size()
    ) / MB


# --------------------------------------------------------------------------- точность


def compare_gradients(
    ref_fn: LossFn,
    test_fn: LossFn,
    inputs: LastLayerInputs,
) -> dict[str, float | str]:
    """Сравнивает loss и градиенты метода test_fn с эталоном ref_fn на одних и тех же входах.

    Возвращает абсолютную разницу loss, относительную ошибку ‖g_test − g_ref‖ / ‖g_ref‖
    и косинусное сходство для ∇E и ∇C. Подходит и для проверки своей реализации,
    и для оценки того, насколько фильтрация градиента искажает результат.
    """

    def run(fn: LossFn) -> tuple[float, torch.Tensor, torch.Tensor]:
        E, C = make_leaf_inputs(inputs)
        loss = fn(E, C, inputs.targets, inputs.softcap)
        loss.backward()
        # Градиенты уносим на CPU, чтобы эталонные ∇E, ∇C не занимали память GPU во время второго прогона.
        out = (float(loss), E.grad.cpu(), C.grad.cpu())
        del E, C, loss
        free_memory()
        return out

    try:
        loss_ref, ge_ref, gc_ref = run(ref_fn)
        loss_tst, ge_tst, gc_tst = run(test_fn)
    except Exception as err:
        if not is_oom(err):
            raise
        del err
        free_memory()
        return {"status": "OOM"}

    def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.float(), b.float()
        return float((a - b).norm() / b.norm().clamp_min(1e-30))

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            torch.nn.functional.cosine_similarity(
                a.float().flatten(), b.float().flatten(), dim=0
            )
        )

    return {
        "status": "ok",
        "loss_ref": loss_ref,
        "loss_test": loss_tst,
        "loss_abs_diff": abs(loss_tst - loss_ref),
        "grad_E_rel_err": rel_err(ge_tst, ge_ref),
        "grad_E_cos": cosine(ge_tst, ge_ref),
        "grad_C_rel_err": rel_err(gc_tst, gc_ref),
        "grad_C_cos": cosine(gc_tst, gc_ref),
    }
