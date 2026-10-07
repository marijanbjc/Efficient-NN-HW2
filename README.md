# Домашнее задание 2 - воспроизведение Cut Cross-Entropy

Разбор и воспроизведение статьи **Cut Your Losses in Large-Vocabulary Language Models** (E. Wijmans et al., Apple, ICLR 2025, [arXiv:2411.09009](https://arxiv.org/abs/2411.09009)).

При обучении языковой модели с большим словарём больше всего памяти занимает последний слой: матрица логитов «токены × словарь». Для Gemma 2 (2B) на 8192 токенах одна её копия весит около 8 ГБ. Cut Cross-Entropy (CCE) считает loss и его градиент, не записывая эту матрицу в память GPU: логиты считаются кусками в быстрой памяти на чипе и сразу сворачиваются. В работе проверено, насколько это экономит память и время, не портит ли градиент и обучение.

| Что сдаётся | Где |
|---|---|
| Презентация с объяснением статьи | [`Cut_losses_presentation.pdf`](Cut_losses_presentation.pdf) |
| Отчёт о воспроизведении | [`03_report.ipynb`](03_report.ipynb) |
| Код воспроизведения | [`src/`](src), [`01_benchmark.ipynb`](01_benchmark.ipynb), [`02_finetune.ipynb`](02_finetune.ipynb) |

---

## 1. Главные результаты

Подробности, графики и объяснение расхождений - в [`03_report.ipynb`](03_report.ipynb).

**Последний слой Gemma 2 (2B): loss + градиент, 8192 токена, |V| = 256 000** (аналог таблицы 1 статьи)

| Метод | Память, МБ | Время, мс | Статья: память / время |
|---|---:|---:|---:|
| Baseline (PyTorch) | 28 000 | 175 | 28 000 / 208 |
| `torch.compile` | 5 161 | 85 | 16 000 / 143 |
| **CCE** | **1 163** | 187 | 1 164 / 145 |
| CCE без фильтрации градиента | 1 161 | 431 | 1 162 / 357 |
| CCE с точным накоплением (fp32) | 3 450 | 210 | 2 326 / 160 (Kahan) |

Нижняя граница памяти (размер самих ∇E + ∇C) - 1161 МБ.

| Утверждение статьи | Результат |
|---|---|
| Память CCE ≈ размер самих градиентов | ✅ в 24 раза меньше Baseline, как в статье |
| Фильтрация градиента ускоряет backward и не портит градиент | ✅ ускорение ×2,3 (в статье ×2,5), ошибка на уровне шума bf16 |
| Softmax разрежен: < 0,02% элементов выше порога bf16 | ✅ 0,014%, ниже порога с 50-го по вероятности токена |
| Обучение с CCE не отличается от обычного | ✅ кривые LoRA fine-tuning совпадают |
| Освободившаяся память позволяет увеличить батч | ✅ ×1,7 (LoRA без чекпоинтинга активаций) |
| CCE не медленнее альтернатив | ❌ на H100 MIG и PyTorch 2.7 CCE медленнее Baseline на 7% и `torch.compile` в 2,2 раза |

---

## 2. Оборудование и версии

| | |
|---|---|
| Видеокарта | NVIDIA H100 80GB, MIG-слайс 3g.40gb (60 SM из 132, 40 ГБ) |
| PyTorch | 2.7.1+cu118 |
| Triton | 3.3.1 |
| transformers | 4.48.2 (версия, под которую написан патч CCE) |
| Пакет CCE | [`cut-cross-entropy`](https://github.com/apple/ml-cross-entropy) 25.9.3 |
| Модель | [Gemma 2 2B Instruct](https://huggingface.co/google/gemma-2-2b-it) |
| Данные | 500 первых примеров [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) |

У авторов: A100 80 GB, PyTorch 2.4.1. Все отличия условий перечислены в начале отчёта.

---

## 3. Структура репозитория

```
HW2/
├── src/
│   ├── preprocessing.py    # CSV → токены → батчи; извлечение E, C, targets из модели
│   ├── methods.py          # реализации loss последнего слоя: Baseline, torch.compile, варианты CCE
│   ├── bench.py            # замер пиковой памяти и времени, сравнение градиентов
│   ├── finetune.py         # LoRA fine-tuning с переключателем HF / CCE, поиск максимального батча
│   └── my_cce.py           # своё ядро CCE на Triton (черновик forward, в замерах не участвует)
├── 01_benchmark.ipynb      # таблица 1, проверка градиентов, разреженность softmax
├── 02_finetune.ipynb       # обучение HF против CCE
├── 03_report.ipynb         # отчёт: читает только results/
├── data/data.csv           # 500 примеров Alpaca: instruction, input, output
├── results/                # сырые замеры (CSV, JSON) и графики
├── materials/              # статья, разборы статьи, исходник презентации
└── Cut_losses_presentation.pdf
```

Ноутбуки запускаются из корня репозитория и импортируют код как пакет: `from src.bench import ...`.

Все реализации loss в `src/methods.py` имеют один интерфейс `loss_fn(E, C, targets, softcap) -> loss`:

| Имя в таблицах | Что это |
|---|---|
| `Baseline` | Обычный PyTorch в порядке HF: логиты и softcap в bf16, затем fp32 и `F.cross_entropy` |
| `torch.compile` | Тот же Baseline под `torch.compile` |
| `CCE` | `linear_cross_entropy(impl="cce")` из пакета авторов |
| `CCE (no grad filter)` | CCE с `filter_eps=None` |
| `CCE (fp32 accum)` | CCE с накоплением ∇E, ∇C в fp32 - аналог CCE-Kahan из статьи |
| `CCE-Kahan-FullC` / `-FullE` | Пресеты пакета: точное накопление + фильтр отключён для ∇C / ∇E |

---

## 4. Как воспроизвести

Нужна NVIDIA GPU (ядра CCE написаны на Triton; на macOS пакет молча переключается на `torch.compile`) и около 40 ГБ видеопамяти для Baseline на 8192 токенах.

**Установка**

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Модель.** Gemma 2 выдаётся по запросу: на [странице модели](https://huggingface.co/google/gemma-2-2b-it) нужно принять лицензию, затем выполнить `huggingface-cli login`. В ноутбуках `MODEL_NAME` указывает на локальную копию модели на сервере - замените на `"google/gemma-2-2b-it"` или свой путь.

**Данные.** Файл `data/data.csv` уже в репозитории. Чтобы собрать его заново:

```python
from datasets import load_dataset
load_dataset("tatsu-lab/alpaca", split="train[:500]").to_pandas()[["instruction", "input", "output"]].to_csv("data/alpaca.csv", index=False)
```

В ноутбуках путь к данным задан в `DATA_CSV` - он должен указывать на этот файл.

**Порядок запуска**

| Шаг | Что запустить |
|---|---|
| 1 | `01_benchmark.ipynb` целиком. Первый запуск прогоняет Gemma и сохраняет E, C на диск, повторные - читают их из файла |
| 2 | `02_finetune.ipynb` с `LOSS_IMPL = "hf"` (300 шагов по ~0,25 с) |
| 3 | **Перезапустить ядро**, `LOSS_IMPL = "cce"`, запустить ещё раз. `cce_patch` меняет модель во всём процессе Python, поэтому прогоны - в разных ядрах |
| 4 | `03_report.ipynb` - собирает таблицы и графики из `results/`, GPU не нужен |

**Если не хватает памяти.** На MIG-слайсе нехватка памяти приходит не как `OutOfMemoryError`, а как `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED`. Бенчмарк и поиск батча распознают её и помечают замер как `OOM`. После такой ошибки память держит упавшее исключение - перезапустите ядро. Можно уменьшить `N_TOKENS` (бенчмарк) или `batch_size` (обучение).

---

## 5. Ограничения

- Один MIG-слайс вместо целой GPU и другое поколение железа: абсолютное время несравнимо со статьёй, сравниваются отношения.
- Fine-tuning: LoRA вместо полного обучения, одна модель, один сид, 500 примеров (≈ 5 эпох - модель запоминает данные, поэтому loss ниже, чем в статье).
- Не измерялись Liger Kernels, Torch Tune и CCE без сортировки словаря (в API пакета нет переключателя).
- Собственная реализация CCE (`src/my_cce.py`) не доведена: есть черновик forward-ядра на Triton.
