# blender_data_sampling

Генерация синтетического датасета для object detection из Blender-сцен с профилями соревнований по схеме `scenes/<competition>/<year>/`.

Текущий runtime больше не привязан к `teknofest`: выбор соревнования и года идёт через путь к конфигу, а вся scene-specific кастомизация описывается в YAML.

## Что изменилось

- единая точка входа: [main.py](main.py)
- два CLI-сценария: `run` и `debug`
- профили сцен лежат в [scenes](scenes)
- старый `pipenv`-flow удалён, проект теперь описан через [pyproject.toml](pyproject.toml)
- экспорт аннотаций сейчас поддерживает только `YOLO bbox`

## Структура профилей сцен

Каждый профиль должен лежать так:

```text
scenes/
  <competition>/
    <year>/
      config.yaml
      <scene>.blend
      resources/
```

Правила:

- `competition` и `year` выводятся из пути к `config.yaml`
- `.blend` должен лежать рядом с `config.yaml`
- все дополнительные файлы профиля описываются относительными путями от директории профиля

Сейчас в репозитории есть:

- [scenes/teknofest/2024/config.yaml](scenes/teknofest/2024/config.yaml) — мигрированный профиль из старого сценария
- [scenes/sauvc/2026/config.yaml](scenes/sauvc/2026/config.yaml) — шаблон под новую сцену SAUVC

## Требования

- Python `3.10`
- установленный `uv`
- Blender executable, путь к которому можно передать в CLI

Основные Python-зависимости:

- `pydantic`
- `PyYAML`
- `opencv-python`
- `joblib`
- `scikit-learn`
- `numpy`

## Установка

```bash
uv sync
```

Если `uv` попросит сначала создать lockfile, выполните:

```bash
uv lock
uv sync
```

## Запуск

Минимальный запуск:

```bash
uv run python main.py run \
  --blender /path/to/blender \
  --config scenes/teknofest/2024/config.yaml
```

Тестовый debug-запуск с bbox прямо на изображениях:

```bash
uv run python main.py debug \
  --blender /path/to/blender \
  --config scenes/teknofest/2024/config.yaml
```

Что делает `main.py`:

1. Валидирует путь `scenes/<competition>/<year>/config.yaml`
2. Проверяет, что `.blend` лежит рядом с конфигом
3. Создаёт export-директорию
4. Сериализует resolved config
5. Запускает Blender в headless-режиме через внутренний runner
6. После рендера собирает YOLO-датасет со split `train/val/test`

## CLI

Поддерживаются две команды:

```bash
python main.py run --blender <path-to-blender> --config <path-to-config>
python main.py debug --blender <path-to-blender> --config <path-to-config>
```

Аргументы:

- `--blender` — абсолютный путь до Blender executable
- `--config` — путь до `scenes/<competition>/<year>/config.yaml`

Режимы:

- `run` — обычный production export без нарисованных bbox
- `debug` — тестовый export на `20` изображений с bbox, нарисованными прямо в `images/`

## Конфиг

Ключевые секции YAML:

- `scene` — имена объектов, коллекций, материалов, нод и относительный путь до `.blend`
- `dataset` — список классов и отображение `object_name -> class_name`
- `render` — размер изображения и Cycles samples
- `sampling` — число кадров, seed, лимиты и пороги валидации bbox
- `randomization` — палитра цветов для recolorable объектов
- `visibility` — `geometric` или `mist_model`
- `export` — формат экспорта и размеры `val/test`

Ключевые ограничения:

- `dataset.classes` — явный упорядоченный список классов для YOLO `class_id`
- `export.format` пока допускает только `yolo_bbox`
- `val_size + test_size` должно быть меньше `1`

Ключевые поля в `sampling`, влияющие на устойчивость генерации:

- `min_valid_bboxes` — минимальное число объектов на каждом сохранённом кадре
- `max_scene_attempts` — watchdog на число пересэмплингов раскладки сцены для одного кадра
- `target_object_attempts` — число object-aware попыток, когда камера ставится рядом с выбранным объектом и наводится на него
- `random_fallback_attempts` и `max_camera_attempts` — резервный random search по сцене
- `target_horizontal_distance_range`, `target_vertical_offset_range` — как далеко от целевого объекта можно ставить камеру
- `target_yaw_jitter_deg_range`, `target_pitch_jitter_deg_range` — насколько сильно шумить вокруг target-oriented ракурса

Runtime сначала пытается собрать кадр через target-oriented camera placement, а потом уходит в random fallback. Кадр сохраняется только если после всех фильтров в нём осталось минимум `min_valid_bboxes` валидных bbox.

## Выходные данные

Обычный запуск пишет результат в:

```text
exports/<competition>_<year>_<timestamp>/
```

Debug-запуск пишет результат в:

```text
exports/debug/<competition>_<year>_<timestamp>/
```

Внутри:

```text
dataset.yaml
manifest.json
images/
  train/
  val/
  test/
labels/
  train/
  val/
  test/
```

В `debug` режиме bbox рисуются прямо в `images/train|val|test`.

Имена изображений:

- `img_000000.jpg`
- `img_000001.jpg`

Имена label-файлов:

- `img_000000.txt`
- `img_000001.txt`

`dataset.yaml` содержит `train`, `val`, `test` и список `names`.

`manifest.json` хранит metadata запуска, seed, классы и распределение по split.

## YOLO bbox

Экспортируется только один target label на bbox:

- `class_id` берётся из индекса класса в `dataset.classes`
- значение класса строится только по `type`, не по `color`

Формат строки в label-файле:

```text
<class_id> <cx> <cy> <w> <h>
```

Все координаты нормализованы относительно размера изображения.

## SAUVC

Профиль [scenes/sauvc/2026](scenes/sauvc/2026) сейчас создан как шаблон.

Чтобы сделать его рабочим, нужно:

- положить реальный `.blend` рядом с `config.yaml`
- заменить placeholder-имена объектов и коллекций на фактические
- при необходимости описать `scene.mist` и включить `visibility.mode: mist_model`

## Проверки

Pure-python тесты:

```bash
python3 -m unittest discover -s tests -v
```

Покрыто:

- разбор пути профиля `scenes/<competition>/<year>/`
- валидация resolved context
- YOLO export и split `train/val/test`

Blender runtime отдельно нужно проверять на машине, где установлен Blender.

## TODO

- `COCO bbox` export
- `YOLO segmentation` export
- отдельные smoke-тесты с реальным Blender executable в CI или локальном профиле
