# SOP Validator

Агент-валидатор стандартных операционных процедур (СОП) для ОЦ Сбербанка на базе **GigaChat-2-Max**.

Принимает на вход СОП и Excel с вопросами + эталонными ответами, прогоняет вопросы через **3 архитектуры промптинга параллельно** (Zero-shot, Chain-of-Thought, Self-Consistency), оценивает каждый ответ через LLM-as-judge (бинарная шкала) и возвращает Excel с детализацией и агрегатами accuracy по типам вопросов.

---

## Возможности

- **3 архитектуры промптинга** запускаются независимо для одного и того же вопроса:
  - `zero_shot` — прямой промпт, temperature=0
  - `cot` — Chain-of-Thought через XML-теги `<размышление>`/`<ответ>`, temperature=0.3
  - `self_consistency` — N=5 прогонов CoT с temp=0.7 + majority vote (binary/multiple) или LLM-агрегация (open)
- **LLM-as-judge** с двухпроходным парсингом JSON (main → fallback) и устойчивостью к «мусору» в ответе
- **Параллельная обработка** вопросов через `ThreadPoolExecutor` + tqdm прогресс-бар
- **Промежуточные чекпоинты** каждые N вопросов — при падении не теряем результаты
- **mTLS-аутентификация** GigaChat через клиентский сертификат (для внутреннего контура Сбера)
- **Retry с экспоненциальным backoff** на 5xx и сетевых ошибках
- **Расширяемость**: добавление новой архитектуры = один класс в `core/architectures/`

---

## Структура проекта

```
sop_validator/
├── config.yaml                 # настройки (пути к сертам, модель, N, temp, таймауты)
├── main.py                     # CLI entry point
├── requirements.txt
├── README.md
├── core/
│   ├── gigachat_client.py      # HTTP-клиент GigaChat с mTLS + tenacity retry
│   ├── models.py               # pydantic-модели + парсеры CoT/бинарных ответов
│   ├── config_loader.py        # загрузчик YAML с дефолтами и валидацией
│   ├── judge.py                # LLM-as-judge с двухпроходным парсингом
│   ├── excel_io.py             # чтение/запись Excel + чекпоинты
│   ├── aggregator.py           # Summary + Disagreements
│   ├── runner.py               # оркестрация (ThreadPoolExecutor)
│   └── architectures/
│       ├── base.py             # абстрактный класс + реестр
│       ├── zero_shot.py
│       ├── cot.py
│       └── self_consistency.py
├── prompts/
│   ├── zero_shot.txt
│   ├── cot.txt
│   ├── sc_aggregator.txt       # агрегатор open-ответов для SC
│   ├── judge.txt
│   └── judge_fallback.txt
├── tests/
│   ├── test_parsers.py         # парсеры CoT-тегов, нормализаторы
│   ├── test_voting.py          # majority vote, JSON-парсеры
│   ├── test_aggregator.py      # Summary, Disagreements
│   └── test_smoke.py           # интеграционный тест с моком клиента
└── examples/
    ├── example_sop.md          # пример СОП АК.03(М)
    ├── example_questions.xlsx  # пример входного Excel
    └── make_example_questions.py
```

---

## Установка

**Требования:** Python 3.11+

```bash
cd sop_validator
pip install -r requirements.txt
```

---

## Настройка

Открой `config.yaml` и заполни пути к mTLS-сертификатам:

```yaml
gigachat:
  url: "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1/chat/completions"
  model: "GigaChat-2-Max"
  cert_path: "/path/to/client.crt"   # ← заполнить
  key_path: "/path/to/client.key"    # ← заполнить
  verify_ssl: false                  # внутренний контур
```

Остальные параметры (N прогонов Self-Consistency, температуры, таймауты, max_workers) настраиваются там же. Любой параметр можно переопределить, не трогая код.

**Безопасность:** не коммитьте `config.yaml` с реальными путями к сертам в git. Добавьте его в `.gitignore` или используйте шаблон + env-переменные.

---

## Запуск

### Минимальный

```bash
python main.py \
    --sop ./examples/example_sop.md \
    --questions ./examples/example_questions.xlsx
```

Результат сохранится в `./examples/example_questions_results_<timestamp>.xlsx`.

### С явным путём вывода

```bash
python main.py \
    --sop ./data/sop_ak03m.txt \
    --questions ./data/questions.xlsx \
    --output ./results/ak03m_run1.xlsx \
    --config ./config.yaml
```

### Только две архитектуры (без Self-Consistency)

```bash
python main.py \
    --sop ./data/sop.md \
    --questions ./data/questions.xlsx \
    --architectures zero_shot,cot
```

Доступные имена: `zero_shot`, `cot`, `self_consistency` (или алиас `sc`).

---

## Формат входного Excel

Лист `Questions` (или первый лист, если называется иначе) с колонками:

| sop_id | question | question_type | reference_answer |
|--------|----------|---------------|------------------|
| AK03M_v1 | Что произойдёт, если выплата произведена 26 числа? | open | Если срок — до 20 числа, выплата 26-го… |
| AK03M_v1 | Требуется ли согласование с отделом рисков при сумме > 500 тыс.? | binary | Да |
| AK03M_v1 | Какой этап идёт после проверки документов? Варианты: (А)… (Б)… (В)… | multiple | Согласование с андеррайтером |

**Типы вопросов:**
- `open` — открытый, свободная формулировка
- `binary` — да/нет
- `multiple` — выбор из перечисленных в `question` вариантов

**Ключевой принцип составления:** вопросы не должны содержать прямых цитат из СОП. Проверяется смысловое понимание логики процесса, а не поиск по ключевым словам.

---

## Формат выходного Excel

### Лист `Questions`
Исходные колонки + для каждой архитектуры:

| Столбец | Описание |
|---------|----------|
| `answer_zeroshot` / `answer_cot` / `answer_sc` | Финальный ответ модели |
| `answer_cot_reasoning` | Блок рассуждения CoT (для аудита) |
| `answer_sc_all_votes` | JSON со всеми N прогонами Self-Consistency |
| `score_<arch>` | Вердикт судьи: 0/1 (пусто если судья не смог распарсить) |
| `judge_comment_<arch>` | 1–2 предложения обоснования от судьи |
| `latency_<arch>_sec` | Время в секундах |
| `tokens_<arch>` | Суммарно использованные токены |
| `error_<arch>` | Сообщение об ошибке, если прогон упал |

### Лист `Summary`
Одна строка на архитектуру: `accuracy`, `accuracy_open`, `accuracy_binary`, `accuracy_multiple`, `evaluated`, `correct`, `judge_failed`, `exec_errors`, `total_tokens`, `avg_latency_sec`.

### Лист `Disagreements` (опционально)
Создаётся, если между архитектурами есть расхождения в оценках — для быстрого ручного разбора спорных кейсов.

---

## Тесты

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

Покрытие:
- `test_parsers.py` — извлечение `<ответ>`/`<размышление>`, нормализация Да/Нет, подготовка к голосованию
- `test_voting.py` — majority vote по типам вопросов, JSON-парсеры (markdown-fences, мусор вокруг JSON, ru-строковые score)
- `test_aggregator.py` — Summary (accuracy, breakdown по типам, judge_failed), Disagreements
- `test_smoke.py` — **интеграционный тест полного пайплайна на моке клиента**: 2 вопроса × 3 архитектуры + judge + запись Excel

Текущее состояние: **39/39 зелёные**.

---

## Что учтено в реализации

**Особенности GigaChat:**
- Нет нативного CoT / Extended Thinking — реализовано через XML-теги в промпте
- Нет structured output / JSON schema — парсинг через регулярки + fallback-промпт
- mTLS без Bearer-токена, `verify=False` с предупреждением в логах (внутренний контур)
- Retry: 3 попытки, экспоненциальный backoff (1s → 2s → 4s, cap 20s)

**Устойчивость:**
- Каждый вопрос изолирован: падение одного не останавливает прогон
- Двухпроходный парсинг JSON у судьи с fallback-промптом
- Пустой ответ модели → автоматический score=0 без вызова судьи (экономим токены)
- Промежуточное сохранение `.checkpoint.xlsx` каждые N вопросов

**Воспроизводимость:**
- Zero-shot и judge при temp=0 детерминистичны
- CoT при temp=0.3 почти детерминистичен (для аудита)
- Self-Consistency при temp=0.7 диверсифицирован специально — в `answer_sc_all_votes` сохраняются все прогоны для дебага

---

## Расширение новой архитектурой

1. Создай `core/architectures/my_new_arch.py`:
    ```python
    from core.architectures.base import ArchitectureRegistry, BaseArchitecture
    from core.models import ArchitectureName, ArchitectureResult, InputQuestion

    @ArchitectureRegistry.register(ArchitectureName.MY_NEW)
    class MyNewArchitecture(BaseArchitecture):
        def answer(self, question: InputQuestion, sop_text: str) -> ArchitectureResult:
            ...
    ```
2. Добавь `MY_NEW` в `ArchitectureName` (`core/models.py`).
3. Добавь ключ в `config.yaml` под `architectures:`.
4. Добавь имя в `_ARCH_LABELS`, `ARCH_OUTPUT_COLUMNS` и mapping в `ArchitectureRegistry.build_enabled`.
5. Импортируй в `core/architectures/__init__.py` — иначе не зарегистрируется.

---

## Известные ограничения

- **Контекст GigaChat-2-Max**: очень длинные СОП (>30k символов) могут упереться в лимит. В `judge.py` уже есть truncation до 6000 символов для судьи. Для основного промпта — пока обрезка не реализована (при необходимости — добавить chunking).
- **LLM-агрегация open-ответов в SC**: если все 5 прогонов дали разные формулировки, агрегатор выбирает «наиболее консенсусный», но это не гарантирует семантическую близость к эталону. Для критичных кейсов смотрите `answer_sc_all_votes`.
- **Rate limits GigaChat**: дефолт `max_workers=3` рассчитан на умеренную нагрузку. Если упираетесь в rate limit — уменьшайте до 1–2.

---


