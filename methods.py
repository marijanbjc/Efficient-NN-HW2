"""Реализации loss последнего слоя с единым интерфейсом: (E, C, targets, softcap) -> loss.

- baseline на PyTorch;
- baseline под torch.compile;
- варианты CCE из официального пакета (cut-cross-entropy);
- my_cce.
"""
from functools import partial

import torch
import torch.nn.functional as F

from cut_cross_entropy import linear_cross_entropy

"""
Обычная реализация на pytorch на каждой операции делает копию матрицы.
В ней постоянно что-то пишется и читается из HBM.

E @ C.T          → ядро cuBLAS,  пишет логиты bf16    (N×V)
.to(float32)     → ядро,          пишет копию fp32     (N×V)
/ softcap        → ядро,          пишет ещё копию      (N×V)
tanh             → ядро,          пишет ещё копию      (N×V)
* softcap        → ядро,          пишет ещё копию      (N×V)
cross_entropy    → log_softmax + nll: ещё копия (N×V) и редукция

Каждая поэлементная операция почти ничего не считает: деление или tanh дёшевы. 
Но она целиком прогоняет матрицу N×V через HBM, туда и обратно. 
Такие операции ограничены пропускной способностью памяти (memory-bound), а не вычислениями. 
Кроме того, autograd сохраняет промежуточные тензоры для backward, и память копится.
"""


def baseline_loss_fn(E: torch.Tensor, C: torch.Tensor, targets: torch.Tensor, softcap: float | None) -> torch.Tensor:
    """
    Бейзлайн для расчёта лосса без каких-либо оптимизаций:
    - Создаются копии в операциях softcap
    - Матрица логитов считается полностью разом без тайлинга
    - Делает конвертацию в fp32

    :param E: (N, D)  bf16, requieres_grad - матрица выходных скрытых состояний
    для каждой позиции текстовой последовательности
    :param C: (V, D)  bf16, requieres_grad - матрица классифицирующей головы
    (проецирующей из скрытого состояния в словарь)
    :param targets: (N,)    long, индексы правильных токенов
    :param softcap: float | None

    :return: loss (1) (тензор с requieres_grad) — среднее по N токенам
    """

    logits = (E @ C.T).to(torch.float32)
    if softcap is not None and softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)

    loss = F.cross_entropy(logits, targets)

    return loss


"""
Скомпилированная торчом версия обычного торчового лосса.
При первом вызове она перехватывает код, строит граф операций, оптимизирует его и генерирует новые ядра
Перехватывает исполнение Python-байткода функции и записывает тензорные операции в граф (FX graph). 
Ставит guards: условия, при которых граф остаётся верным (формы, dtype, устройство). 
Если на следующем вызове guard не выполнился, например изменился N, функция перекомпилируется.
Inductor Сливает цепочки поэлементных операций и редукций в одно ядро (kernel fusion) 
и генерирует его на Triton для GPU или на C++ для CPU.


За счёт чего оптимизирует:
- Слияние ядер
to(float) → /softcap → tanh → * softcap → logsumexp выполняется за один проход по памяти вместо пяти. Меньше трафика HBM — быстрее
- Нет промежуточных тензоров
Копии N×V между слитыми операциями не создаются, они живут в регистрах
- Пересчёт вместо хранения
В backward дешёвые операции пересчитываются, меньше сохранённых тензоров в HBM

Inductor, как правило, не сливает matmul с последующей редукцией по словарю. 
Логиты N×V всё равно один раз целиком записываются в HBM после E @ C.T. 
Дальше компилятор лишь эффективнее с ними обходится.
"""

compiled_loss_fn = torch.compile(baseline_loss_fn)

"""
Имплементация подсчёта лосса с помощью реализованных авторами статьи ядер на Triton.
Внутри происходят следующие оптимизации:
- Делается тайлинг для параллельных вычислений отдельных блоков E, C
- Матрица логитов никогда не пишется в HBM - только lse и логиты правильных токенов словаря
- Операции делаются в fp32, но в HBM пишутся в bf16

for C_v in C:
    for E_n in E:
        A = C_v.T @ E_n
        G_nv = exp(A_nv - LSE_n) - onehot(x)  # onehot правильных токенов в пределах блока
        is_empty = filter_eps is not None and all(|G_nv| < eps)  # Проверяем включена ли фильтрация и если да, то пропускать ли блок
        if not (is_empty and filter_e_grad):
            # Сюда попадаем тогда, когда фильтрация выключена/блок не пустой и изначально не требуется фильтрация для градиентов по E
            ∇E_n += G_nv @ C_v
        if not (is_empty and filter_c_grad):
            # Сюда попадаем тогда, когда фильтрация выключена/блок не пустой и изначально не требуется фильтрация для градиентов по C
            ∇C_v += G_nv.T @ E_n
"""


def cce_base_loss_fn(
        E: torch.Tensor,
        C: torch.Tensor,
        targets: torch.Tensor,
        softcap: float | None,
        impl: str = "cce",
        filter_eps: float | str | None = "auto",
        accum_e_fp32: bool = False,
        accum_c_fp32: bool = False,
) -> torch.Tensor:
    """Обёртка над linear_cross_entropy с интерфейсом бенчмарка (E, C, targets, softcap) -> loss.

    impl:          "cce" — параметры ниже действуют как переданы;
                   "cce_kahan_full_c" / "cce_kahan_full_e" — пресеты пакета,
                   они сами выставляют filter_eps, accum_* и filter_*_grad (наши значения игнорируются).
    filter_eps:    порог «пустого» блока G в backward; None — фильтрация выключена.
    accum_*_fp32:  копить ∇E / ∇C в fp32-буфере вместо bf16 (точнее, но вдвое больше памяти).
    """

    return linear_cross_entropy(
        E, C, targets,
        softcap=softcap, reduction="mean", shift=0, ignore_index=-100,
        impl=impl, filter_eps=filter_eps,
        accum_e_fp32=accum_e_fp32, accum_c_fp32=accum_c_fp32,
    )


# Строки таблицы 1 статьи. Имена ключей METHODS попадают в итоговую таблицу бенчмарка.

# CCE: фильтрация градиента включена, накопление ∇E / ∇C в bf16.
cce_loss_fn = partial(cce_base_loss_fn, impl="cce")

# CCE без фильтрации градиента: все блоки G считаются полностью.
cce_no_filter_loss_fn = partial(cce_base_loss_fn, impl="cce", filter_eps=None)

# CCE с точным накоплением градиентов (accum_*_fp32=True): на Triton ≥ 3.2 — fp32-буфер,
# на Triton < 3.2 — суммирование Кэхэна (bf16 + bf16-буфер поправок). Аналог CCE-Kahan из статьи.
cce_fp32_accum_loss_fn = partial(cce_base_loss_fn, impl="cce", accum_e_fp32=True, accum_c_fp32=True)

# Пресет пакета; в статье — CCE-Kahan-FullC: точное накопление + фильтр отключён для ∇C.
cce_kahan_full_c_loss_fn = partial(cce_base_loss_fn, impl="cce_kahan_full_c")

# Пресет пакета; в статье — CCE-Kahan-FullE: точное накопление + фильтр отключён для ∇E.
cce_kahan_full_e_loss_fn = partial(cce_base_loss_fn, impl="cce_kahan_full_e")

METHODS = {
    "Baseline": baseline_loss_fn,
    "torch.compile": compiled_loss_fn,
    "CCE": cce_loss_fn,
    "CCE (no grad filter)": cce_no_filter_loss_fn,
    "CCE (fp32 accum)": cce_fp32_accum_loss_fn,
    "CCE-Kahan-FullC": cce_kahan_full_c_loss_fn,
    "CCE-Kahan-FullE": cce_kahan_full_e_loss_fn,
}
