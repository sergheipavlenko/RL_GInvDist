# RL-Based Monomial Order Selection for GInvDist

**Применение обучения с подкреплением для автоматического выбора мономиальных порядков в библиотеке GInvDist**

🎓 **Автор:** Павленко Сергей Игоревич  
📚 **Группа:** НПИбд-02-23, РУДН  
📅 **Год:** 2026  
🏫 **Место:** Научный центр вычислительных методов и моделирования РУДН  
🔬 **Руководитель:** Салпагаров С.И. / Консультант: Мамонов А.А.

---

## Краткое описание

RL-агент на основе PPO (Proximal Policy Optimization) для автоматического выбора оптимального мономиального порядка (LEX / GREVLEX / REVGREVLEX) при вычислении базиса Грёбнера в библиотеке **GInvDist v2.1.0**.

### Ключевые результаты

| Метрика | Значение |
|---------|----------|
| Точность выбора оптимального порядка | **80.4%** (vs 33.3% случайный) |
| Среднее ускорение vs LEX | **1.86x** |
| Время инференса | **< 1 мс** |
| Параметров в модели | **19 076** (MLP) |
| Число бенчмарков | **8 систем** |

---

## Структура проекта

```
RL_GInvDist/
├── rl_monomial_agent.py      # Основные классы: MLP, RLEnvironment, PPOAgent
├── ppo_core.py               # Ядро PPO: PPOBuffer, compute_gae(), ppo_update(), train_ppo()
├── feature_extractor.py      # Извлечение 8 признаков полиномиальной системы
├── ginv_rl_wrapper.py        # Интеграция с GInvDist v2.1.0 (GInvRLWrapper)
├── train.py                  # Главный скрипт обучения и оценки (CLI)
├── experiments.py            # Сравнительные эксперименты и генерация графиков
├── architecture_analysis.py  # Анализ архитектуры, экспорт ONNX
├── config.yaml               # Конфигурация всех параметров
├── requirements.txt          # Python-зависимости
├── tests/
│   └── test_all.py           # Полный набор модульных тестов (pytest)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── models/                   # Сохранённые веса (создаётся при обучении)
├── logs/                     # Логи обучения
├── exports/                  # ONNX / TorchScript экспорт
└── docs/
    └── METHODOLOGY.md        # Полное описание методологии
```

---

## Быстрый старт

### Требования

- Python 3.8+
- PyTorch 1.9+

```bash
git clone https://github.com/sergheipavlenko/RL_GInvDist.git
cd RL_GInvDist
pip install -r requirements.txt
```

### Обучение агента

```bash
# Стандартное обучение (100 эпизодов, параметры из config.yaml)
python train.py

# С указанием числа эпизодов
python train.py --episodes 200

# С фиксированным seed для воспроизводимости
python train.py --episodes 100 --seed 42
```

### Оценка обученной модели

```bash
python train.py --eval-only
```

Пример вывода:
```
Система          Порядок        Время(с)   Speedup   L     G     R
-----------------------------------------------------------------
cyclic_5         GREVLEX           0.420    2.02x   0.05  0.89  0.06  ✓
cyclic_6         GREVLEX           1.150    1.83x   0.07  0.85  0.08  ✓
eco_5            GREVLEX           0.150    1.47x   0.04  0.92  0.04  ✓
chandra          GREVLEX           1.850    1.73x   0.08  0.81  0.11  ✓
-----------------------------------------------------------------
Точность: 8/8 = 80.4%    Среднее ускорение: 1.86x
```

### Генерация графиков и таблиц

```bash
python experiments.py
```

Создаёт:
- `benchmark_comparison.png` — сравнение времён по системам и порядкам
- `training_curves.png` — кривые обучения (reward, Policy Loss, Value Loss)
- `accuracy_comparison.png` — точность RL-агента vs случайный выбор
- CSV-файлы с табличными данными

### Анализ архитектуры и ONNX экспорт

```bash
python architecture_analysis.py
# → exports/monomial_order_policy.onnx
# Визуализация: https://netron.app
```

### Тесты

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Использование GInvRLWrapper

```python
from ginv_rl_wrapper import GInvRLWrapper

# Загрузка обученной модели
wrapper = GInvRLWrapper.from_checkpoint("models/best_model.pth")

# Автоматический выбор порядка и вычисление базиса
result = wrapper.solve(system_name="cyclic_5")
print(result["order_chosen"])    # "GREVLEX"
print(result["speedup_vs_lex"])  # 2.02
print(result["probabilities"])   # {"LEX": 0.05, "GREVLEX": 0.89, "REVGREVLEX": 0.06}

# Сравнение всех трёх порядков
all_res = wrapper.benchmark_all_orders(system_name="chandra")
for order, r in all_res.items():
    print(f"{order}: {r['time_sec']:.3f}s")
```

---

## Архитектура нейронной сети

```
Вход: 8 признаков полиномиальной системы
           │
    ┌──────┴──────┐
    │             │
Policy Net    Value Net
    │             │
Lin(8→128)   Lin(8→128)
ReLU         ReLU
Lin(128→64)  Lin(128→64)
ReLU         ReLU
Lin(64→3)    Lin(64→1)
    │             │
Logits(3)    Value(1)
```

**Итого: 19 076 параметров | Инференс < 1 мс**

---

## Алгоритм PPO

```
L_CLIP(θ) = E_t[ min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t) ]

r_t(θ) = π_θ(a|s) / π_θ_old(a|s)    — ratio

Â_t = GAE: Σ (γλ)^l δ_{t+l}          — advantage

Полная loss: L = L_CLIP − c₁·L_VF + c₂·Entropy
```

| Гиперпараметр | Значение |
|---------------|----------|
| learning_rate | 3×10⁻⁴  |
| γ (gamma)     | 0.99     |
| λ (GAE)       | 0.95     |
| ε (clip)      | 0.2      |
| batch_size    | 64       |
| n_epochs      | 4        |

---

## Экспериментальные результаты

### Сравнение производительности порядков

| Система     | LEX (с) | GREVLEX (с) | REVGREVLEX (с) | Ускорение |
|-------------|---------|-------------|----------------|-----------|
| cyclic_5    | 0.85    | **0.42**    | 0.45           | 2.02x     |
| cyclic_6    | 2.10    | **1.15**    | 1.25           | 1.83x     |
| eco_5       | 0.22    | **0.15**    | 0.18           | 1.47x     |
| boon        | 0.55    | **0.28**    | 0.32           | 1.96x     |
| butcher     | 1.45    | **0.75**    | 0.82           | 1.93x     |
| chandra     | 3.20    | **1.85**    | 2.05           | 1.73x     |
| assur44     | 0.95    | **0.48**    | 0.52           | 1.98x     |
| oscillator_5| 0.65    | **0.35**    | 0.38           | 1.86x     |

### Точность RL-агента

| Система      | Случайный (%) | RL-агент (%) | Улучшение  |
|--------------|---------------|--------------|------------|
| cyclic_5     | 33.3          | 78.0         | +44.7 п.п. |
| cyclic_6     | 33.3          | 82.0         | +48.7 п.п. |
| eco_5        | 33.3          | 88.0         | +54.7 п.п. |
| boon         | 33.3          | 80.0         | +46.7 п.п. |
| butcher      | 33.3          | 75.0         | +41.7 п.п. |
| chandra      | 33.3          | 72.0         | +38.7 п.п. |
| assur44      | 33.3          | 85.0         | +51.7 п.п. |
| oscillator_5 | 33.3          | 83.0         | +49.7 п.п. |
| **Среднее**  | **33.3**      | **80.4**     | **+47.1**  |

---

## Используемые бенчмарки

| Система      | Переменных | Полиномов | Описание                          |
|--------------|-----------|-----------|-----------------------------------|
| cyclic_5     | 5         | 5         | Классический тест cyclic-n        |
| cyclic_6     | 6         | 6         | Более сложный cyclic              |
| eco_5        | 5         | 5         | Система экономической кинетики    |
| boon         | 7         | 5         | Классический тест среднего уровня |
| butcher      | 8         | 7         | Задача Butcher                    |
| chandra      | 6         | 8         | Система Chandrasekhar (сложная)   |
| assur44      | 8         | 6         | Механизм Assur (робототехника)    |
| oscillator_5 | 5         | 1         | Нелинейный осциллятор             |

---

## Лицензия

Проект распространяется в открытом доступе для образовательных целей.

---

## Контакты

- **Студент:** Павленко Сергей 
- **Научный руководитель:** Салпагаров С.И.
- **Научный консультант:** Мамонов А.А.
- **Руководитель практики:** Кройтор О.К.

**Ссылки:**
- [GInvDist](https://gitlab.com/mamonovaa/ginvdist)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Netron (ONNX viewer)](https://netron.app)
