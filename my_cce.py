"""Собственная реализация Cut Cross-Entropy на PyTorch (torch.autograd.Function)."""
import torch
import triton
import triton.language as tl


@triton.jit
def forward(
        E_ptr,  # Указатель на матрицу E (скрытые состояния токенов) -  N x D [8192x2304]
        C_ptr,  # Указатель на матрицу C (классификационная голова) - V x D [250_000x2304]
        output_ptr,  # Указатель на участок памяти для записи результата
        stride,  # Сколько элементов в 1-й строке матрицы E и в 1-м столбце матрицы C
        BLOCK_SIZE_D: tl.constexpr,  # Размер блока по скрытому состоянию
        BLOCK_SIZE_E: tl.constexpr,  # Размер блока по токенам последовательности
        BLOCK_SIZE_C: tl.constexpr,  # Размер блока по по словам словаря
        Lock,  # Просто указатель на вектор блокировок (0/1) по одному числу на каждый блок [N/BLOCK_SIZE_E]
):

    # Идентификаторы текущей программы в 2D сетке
    # x - отвечает за то, какой блок слов словаря мы обрабатываем
    # y - отвечает за то, какой блок токенов последовательности мы обрабатываем
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Мы сейчас в данной координате матриц E и C
    # т.к. матрица в памяти хранится в плоском виде со страйдами
    start_C = pid_x * BLOCK_SIZE_C * stride
    start_E = pid_y * BLOCK_SIZE_E * stride

    accumulator = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_E), dtype=tl.float32)

    for block_d in range(0, stride, BLOCK_SIZE_D):
        # tl.arange(0, BLOCK_SIZE_C) * stride = [0, 1, 2] * 128 = [0, 128, 256] - три токена
        # tl.arange(0, BLOCK_SIZE_D)[:, None] = [[0], [1], [2], ..., [127], [128] - кусок по d, для которого делаем вычисления
        # offset = [[0, 128, 256], [1, 129, 257], ..., [127, 255, 511]]

        # Берём BLOCK_SIZE_D измерений начиная с block_d для BLOCK_SIZE_C слов словаря и BLOCK_SIZE_E токенов последовательности
        # Сдвиг stride нужен для того, чтобы перескочить по измерению D в каждой из двух матриц
        # И взять одни и те же координаты для пачки токенов и пачки слов
        offset_c = tl.arange(0, BLOCK_SIZE_C) * stride + (block_d + tl.arange(0, BLOCK_SIZE_D))[:, None]  # [BD, BV]
        offset_e = tl.arange(0, BLOCK_SIZE_E) * stride + (block_d + tl.arange(0, BLOCK_SIZE_D))[:, None]  # [BD, BN]

        c_block = tl.load(C_ptr + start_C + offset_c)
        e_block = tl.load(E_ptr + start_E + offset_e)

        accumulator += tl.dot(c_block.T, e_block)  # [BV, BD] x [BD, BN] = [BV, BN]

    new_val = tl.log(tl.exp(accumulator).sum(axis=0))  # [BN]

    # Считываем из памяти начиная с pid_y-индекса блока BLOCK_SIZE_E элементов
    out_offset = pid_y * BLOCK_SIZE_E + tl.arange(0, BLOCK_SIZE_E)

    # Далее нам нужно прочитать прошлое значение LSE
    # Выполнить log-sum-exp со старым и новым значением
    # и записать результат в выходную ячейку памяти

    # Тут мы обращаемся к pid_y элементу, т.к. конфликтуют друг с другом только те программы,
    # которые в 1 момент времени работают с одними и теми же токенами последовательности
    # и обращаются к памяти за LSE и пишут для них LSE. Разные блоки токенов друг другу не мешают.

    # atomic_cas
    # Возвращает СТАРОЕ значение *ptr (до операции).
    #   вернула 0 -> замок был свободен, мы его заняли (записали 1) → выходим из цикла;
    #   вернула 1 -> замок держит другая программа → крутимся дальше.
    # sem="acquire": чтения/записи ПОСЛЕ захвата не могут выполниться раньше него,
    # поэтому чтение LSE ниже увидит всё, что записал предыдущий владелец замка.
    while tl.atomic_cas(Lock + pid_y, 0, 1, sem="acquire") == 1:
        # Т.е. пока atomic_cas не выдаст 0, будет крутится цикл, а 0 он выдаст тогда,
        # когда в Lock'е будет 0, т.е. никакие другие программы не занимают сейчас эту память
        pass

    # --- Критическая секция ---
    # 1. Загружаем текущее значение (log-sum-exp) с семантикой acquire.
    old_val = tl.load(output_ptr + out_offset)

    # 2. Вычисляем новое log-sum-exp: log(exp(old) + exp(new_val)).
    #    Используем стабильную формулу с вычитанием максимума для избежания переполнения.

    # LSE(A U B) = max(LSE(A), LSE(B)) + log(1 + exp(-|LSE(A) - LSE(B)|))
    m = tl.maximum(old_val, new_val)  # [BN]
    new = m + tl.log(1 + tl.exp(-tl.abs(new_val - old_val)))  # [BN]

    # 3. Сохраняем результат с семантикой release.
    tl.store(output_ptr + out_offset, new)

    # Барьер нужен для того, чтобы мы дождались записи ВСЕХ элементов BLOCK_SIZE_E всеми потоками программы
    tl.debug_barrier()

    # --- Освобождение блокировки ---
    # sem="release": чтения/записи ДО освобождения не могут выполниться позже него,
    # поэтому следующая программа, захватившая замок, увидит наш записанный LSE.
    tl.atomic_xchg(Lock + pid_y, 0, sem="release")


def backward(
        ...
):
    pass

