import json
import os
import tkinter as tk
from tkinter import messagebox

import torch
from torch import nn
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor


# ============================================================
# НАСТРОЙКИ
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_MODEL_FILE = "mnist_cnn_model.pth"
USER_MODEL_FILE = "mnist_cnn_user.pth"
UNKNOWN_DETECTOR_FILE = "unknown_detector.pth"
UNKNOWN_CONFIG_FILE = "unknown_detector_config.json"

BRUSH_SIZE = 14
DEFAULT_SPLIT_GAP = 18
DEFAULT_ROW_GAP = 18
MIN_FOREGROUND_PIXELS = 20
MAX_DIGITS = 20

# Если выделенный фрагмент значительно шире обычной цифры,
# программа попробует найти внутри него наиболее тонкое место.
AUTO_SPLIT_WIDE_RATIO = 0.92
EXPECTED_DIGIT_WIDTH_RATIO = 0.62


# ============================================================
# АРХИТЕКТУРА CNN
# Полностью совпадает с mnist_cnn_train.py.
# ============================================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class UnknownDetector(nn.Module):
    """Отдельная сеть: цифра (1) или неизвестный символ (0)."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def load_model():

    if os.path.exists(USER_MODEL_FILE):
        model_file = USER_MODEL_FILE
    elif os.path.exists(BASE_MODEL_FILE):
        model_file = BASE_MODEL_FILE
    else:
        raise FileNotFoundError(
            f"Не найдены {USER_MODEL_FILE} и {BASE_MODEL_FILE}.\n"
            "Положите программу в папку C:\\AI рядом с файлом модели."
        )

    loaded_model = CNN().to(device)

    state = torch.load(
        model_file,
        map_location=device,
        weights_only=True
    )

    loaded_model.load_state_dict(state)
    loaded_model.eval()

    for parameter in loaded_model.parameters():
        parameter.requires_grad_(False)

    return loaded_model, model_file


def load_unknown_detector():

    if not os.path.exists(UNKNOWN_DETECTOR_FILE):
        raise FileNotFoundError(
            f"Не найден {UNKNOWN_DETECTOR_FILE}.\n"
            "Сначала запустите: python train_unknown_detector.py"
        )

    if not os.path.exists(UNKNOWN_CONFIG_FILE):
        raise FileNotFoundError(
            f"Не найден {UNKNOWN_CONFIG_FILE}.\n"
            "Сначала запустите: python train_unknown_detector.py"
        )

    with open(UNKNOWN_CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    threshold = float(config.get("digit_probability_threshold", 0.50))
    threshold = max(0.01, min(threshold, 0.99))

    loaded_detector = UnknownDetector().to(device)
    state = torch.load(
        UNKNOWN_DETECTOR_FILE,
        map_location=device,
        weights_only=True
    )
    loaded_detector.load_state_dict(state)
    loaded_detector.eval()

    for parameter in loaded_detector.parameters():
        parameter.requires_grad_(False)

    return loaded_detector, threshold, config


model, current_model_file = load_model()
unknown_detector, unknown_threshold, unknown_config = load_unknown_detector()

print("Устройство:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Модель:", current_model_file)
print("Детектор неизвестных символов:", UNKNOWN_DETECTOR_FILE)
print(f"Порог 'это цифра': {unknown_threshold:.4f}")
print("Режим: только распознавание, веса модели не изменяются")


# ============================================================
# ОКНО И ХОЛСТ
# ============================================================

root = tk.Tk()
root.title("Цифры и неизвестные символы — CNN")
root.state("zoomed")
root.resizable(True, True)
root.update_idletasks()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

CANVAS_WIDTH = min(max(screen_width - 180, 850), 1500)
CANVAS_HEIGHT = min(max(int(screen_height * 0.43), 300), 480)

drawing_image = Image.new(
    "L",
    (CANVAS_WIDTH, CANVAS_HEIGHT),
    color=0
)
drawing = ImageDraw.Draw(drawing_image)

last_x = None
last_y = None

# Позиции ручных разделителей в формате (x, y). Координата y нужна,
# чтобы разделитель действовал только на выбранную строку.
manual_separators = []


# ============================================================
# РИСОВАНИЕ
# ============================================================

def start_draw(event):

    global last_x, last_y

    canvas.delete("segment_box")
    canvas.delete("auto_separator")
    last_x = event.x
    last_y = event.y

    radius = BRUSH_SIZE // 2

    canvas.create_oval(
        event.x - radius,
        event.y - radius,
        event.x + radius,
        event.y + radius,
        fill="white",
        outline="white",
        tags="ink"
    )

    drawing.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius
        ),
        fill=255
    )


def draw_digit(event):

    global last_x, last_y

    if last_x is None or last_y is None:
        return

    canvas.create_line(
        last_x,
        last_y,
        event.x,
        event.y,
        fill="white",
        width=BRUSH_SIZE,
        capstyle=tk.ROUND,
        smooth=True,
        tags="ink"
    )

    drawing.line(
        (last_x, last_y, event.x, event.y),
        fill=255,
        width=BRUSH_SIZE
    )

    radius = BRUSH_SIZE // 2

    drawing.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius
        ),
        fill=255
    )

    last_x = event.x
    last_y = event.y


def stop_draw(event=None):

    global last_x, last_y

    last_x = None
    last_y = None

    canvas.tag_raise("manual_separator")


def redraw_manual_separators():

    canvas.delete("manual_separator")

    for separator_x, separator_y in manual_separators:
        canvas.create_line(
            separator_x,
            max(0, separator_y - 85),
            separator_x,
            min(CANVAS_HEIGHT, separator_y + 85),
            fill="#ff3b30",
            width=2,
            dash=(8, 5),
            tags="manual_separator"
        )


def add_manual_separator(event):

    # Повторный правый щелчок около существующей линии удаляет её.
    nearest = None

    if manual_separators:
        nearest = min(
            manual_separators,
            key=lambda value: (
                abs(value[0] - event.x)
                + abs(value[1] - event.y)
            )
        )

    if (
        nearest is not None
        and abs(nearest[0] - event.x) <= 8
        and abs(nearest[1] - event.y) <= 85
    ):
        manual_separators.remove(nearest)
        action = "Ручной разделитель удалён."
    else:
        manual_separators.append((event.x, event.y))
        manual_separators.sort(key=lambda value: (value[1], value[0]))
        action = "Красный ручной разделитель добавлен."

    canvas.delete("segment_box")
    canvas.delete("auto_separator")
    redraw_manual_separators()

    status_label.config(
        text=(
            f"{action} Нажмите «Распознать последовательность». "
            "Повторный правый щелчок рядом с линией удаляет её."
        )
    )


def clear_canvas(event=None):

    global drawing_image, drawing, manual_separators

    canvas.delete("all")

    drawing_image = Image.new(
        "L",
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        color=0
    )
    drawing = ImageDraw.Draw(drawing_image)
    manual_separators = []

    result_label.config(text="Нарисуйте несколько цифр")
    details_label.config(text="")
    status_label.config(
        text="Можно писать в любом месте и в несколько строк."
    )


# ============================================================
# СЕГМЕНТАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТИ
# ============================================================

def get_foreground_rows():

    pixels = drawing_image.load()
    rows = []

    for y in range(CANVAS_HEIGHT):
        if any(
            pixels[x, y] > 0
            for x in range(CANVAS_WIDTH)
        ):
            rows.append(y)

    return rows


def get_foreground_columns(y1, y2):

    pixels = drawing_image.load()
    columns = []

    for x in range(CANVAS_WIDTH):

        has_ink = False

        for y in range(y1, y2 + 1):
            if pixels[x, y] > 0:
                has_ink = True
                break

        if has_ink:
            columns.append(x)

    return columns


def make_axis_runs(values):

    if not values:
        return []

    runs = []
    start = values[0]
    previous = values[0]

    for value in values[1:]:

        if value == previous + 1:
            previous = value
            continue

        runs.append((start, previous))
        start = value
        previous = value

    runs.append((start, previous))

    return runs


def merge_nearby_runs(runs, split_gap):

    if not runs:
        return []

    groups = [runs[0]]

    for start, end in runs[1:]:

        previous_start, previous_end = groups[-1]
        empty_columns = start - previous_end - 1

        # Маленькие пустые промежутки могут находиться внутри одной
        # цифры. Большой промежуток считается границей цифр.
        if empty_columns < split_gap:
            groups[-1] = (previous_start, end)
        else:
            groups.append((start, end))

    return groups


def find_text_rows():

    try:
        row_gap = int(row_gap_var.get())
    except ValueError:
        row_gap = DEFAULT_ROW_GAP
        row_gap_var.set(str(DEFAULT_ROW_GAP))

    row_gap = max(3, min(row_gap, 100))

    active_rows = get_foreground_rows()
    raw_rows = make_axis_runs(active_rows)

    return merge_nearby_runs(raw_rows, row_gap)


def get_vertical_projection(x1, x2, y1, y2):

    pixels = drawing_image.load()
    projection = {}

    for x in range(x1, x2 + 1):
        projection[x] = sum(
            1
            for y in range(y1, y2 + 1)
            if pixels[x, y] > 0
        )

    return projection


def get_group_height(x1, x2, row_y1, row_y2):

    pixels = drawing_image.load()
    rows = []

    for y in range(row_y1, row_y2 + 1):
        if any(
            pixels[x, y] > 0
            for x in range(x1, x2 + 1)
        ):
            rows.append(y)

    if not rows:
        return 0

    return rows[-1] - rows[0] + 1


def split_at_manual_separators(group, row):

    x1, x2 = group
    row_y1, row_y2 = row

    cuts = [
        separator_x
        for separator_x, separator_y in manual_separators
        if (
            x1 + 2 < separator_x < x2 - 2
            and row_y1 - 15 <= separator_y <= row_y2 + 15
        )
    ]

    if not cuts:
        return [group]

    parts = []
    part_start = x1

    for cut in sorted(cuts):
        parts.append((part_start, cut - 1))
        part_start = cut

    parts.append((part_start, x2))

    return parts


def find_best_valley(projection, ideal_x, radius, min_x, max_x):

    search_start = max(min_x, ideal_x - radius)
    search_end = min(max_x, ideal_x + radius)

    if search_start > search_end:
        return ideal_x

    def valley_score(x):

        left = projection.get(x - 1, projection.get(x, 0))
        center = projection.get(x, 0)
        right = projection.get(x + 1, projection.get(x, 0))

        # Сглаженный профиль устойчивее к одиночным пустым пикселям.
        return left + center * 2 + right

    return min(
        range(search_start, search_end + 1),
        key=lambda x: (valley_score(x), abs(x - ideal_x))
    )


def auto_split_wide_group(group, row):

    x1, x2 = group
    row_y1, row_y2 = row
    width = x2 - x1 + 1
    height = get_group_height(x1, x2, row_y1, row_y2)

    if height <= 0 or width / height < AUTO_SPLIT_WIDE_RATIO:
        return [group], []

    estimated_digit_width = max(
        1.0,
        height * EXPECTED_DIGIT_WIDTH_RATIO
    )

    estimated_count = round(width / estimated_digit_width)
    estimated_count = max(2, min(estimated_count, 6))

    # Защита от ошибочного дробления одной просто широкой цифры.
    if width / estimated_count < height * 0.35:
        return [group], []

    projection = get_vertical_projection(
        x1,
        x2,
        row_y1,
        row_y2
    )
    expected_part_width = width / estimated_count
    radius = max(3, round(expected_part_width * 0.28))
    minimum_part_width = max(5, round(height * 0.28))

    cuts = []

    for part_number in range(1, estimated_count):

        ideal_x = round(x1 + expected_part_width * part_number)
        minimum_x = (
            x1 + minimum_part_width
            if not cuts
            else cuts[-1] + minimum_part_width
        )
        maximum_x = x2 - minimum_part_width * (estimated_count - part_number)

        cut = find_best_valley(
            projection,
            ideal_x,
            radius,
            minimum_x,
            maximum_x
        )

        cuts.append(cut)

    parts = []
    part_start = x1

    for cut in cuts:
        parts.append((part_start, cut - 1))
        part_start = cut

    parts.append((part_start, x2))

    return parts, cuts


def find_digit_boxes():

    try:
        split_gap = int(split_gap_var.get())
    except ValueError:
        split_gap = DEFAULT_SPLIT_GAP
        split_gap_var.set(str(DEFAULT_SPLIT_GAP))

    split_gap = max(3, min(split_gap, 80))

    text_rows = find_text_rows()
    automatic_cuts = []
    pixels = drawing_image.load()
    row_boxes = []

    # Строки отсортированы сверху вниз. Внутри каждой строки
    # цифры обрабатываются слева направо.
    for row_y1, row_y2 in text_rows:

        columns = get_foreground_columns(row_y1, row_y2)
        runs = make_axis_runs(columns)
        groups = merge_nearby_runs(runs, split_gap)
        separated_groups = []

        for group in groups:

            manual_parts = split_at_manual_separators(
                group,
                (row_y1, row_y2)
            )

            for manual_part in manual_parts:
                automatic_parts, part_cuts = auto_split_wide_group(
                    manual_part,
                    (row_y1, row_y2)
                )
                separated_groups.extend(automatic_parts)
                automatic_cuts.extend(
                    (cut, row_y1, row_y2)
                    for cut in part_cuts
                )

        current_row_boxes = []

        for x1, x2 in separated_groups:

            active_rows = []
            foreground_count = 0

            for y in range(row_y1, row_y2 + 1):
                row_has_ink = False

                for x in range(x1, x2 + 1):
                    if pixels[x, y] > 0:
                        row_has_ink = True
                        foreground_count += 1

                if row_has_ink:
                    active_rows.append(y)

            if not active_rows or foreground_count < MIN_FOREGROUND_PIXELS:
                continue

            y1 = active_rows[0]
            y2 = active_rows[-1]
            padding = 3

            current_row_boxes.append(
                (
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(CANVAS_WIDTH, x2 + padding + 1),
                    min(CANVAS_HEIGHT, y2 + padding + 1)
                )
            )

        if current_row_boxes:
            row_boxes.append(current_row_boxes)

    total_boxes = sum(len(boxes) for boxes in row_boxes)
    was_truncated = total_boxes > MAX_DIGITS
    limited_row_boxes = []
    remaining = MAX_DIGITS

    for boxes in row_boxes:

        if remaining <= 0:
            break

        selected = boxes[:remaining]

        if selected:
            limited_row_boxes.append(selected)
            remaining -= len(selected)

    return limited_row_boxes, was_truncated, automatic_cuts


# ============================================================
# ПРЕОБРАЗОВАНИЕ ОДНОЙ ВЫРЕЗАННОЙ ЦИФРЫ В MNIST 28x28
# ============================================================

def center_by_mass(mnist_image):

    pixels = mnist_image.load()
    total = 0
    sum_x = 0
    sum_y = 0

    for y in range(28):
        for x in range(28):
            value = pixels[x, y]
            total += value
            sum_x += x * value
            sum_y += y * value

    if total == 0:
        return mnist_image

    center_x = sum_x / total
    center_y = sum_y / total
    shift_x = round(13.5 - center_x)
    shift_y = round(13.5 - center_y)

    centered = Image.new("L", (28, 28), 0)
    centered.paste(mnist_image, (shift_x, shift_y))

    return centered


def prepare_digit(box):

    cropped = drawing_image.crop(box)
    bbox = cropped.getbbox()

    if bbox is None:
        return None

    cropped = cropped.crop(bbox)
    width, height = cropped.size

    scale = 20 / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    cropped = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    mnist_image = Image.new("L", (28, 28), 0)
    x = (28 - new_width) // 2
    y = (28 - new_height) // 2
    mnist_image.paste(cropped, (x, y))

    return center_by_mass(mnist_image)


def save_diagnostic_strip(images):

    if not images:
        return

    strip = Image.new(
        "L",
        (28 * len(images), 28),
        0
    )

    for index, digit_image in enumerate(images):
        strip.paste(digit_image, (index * 28, 0))

    strip.save("last_sequence_input.png")


def show_boxes(row_boxes, automatic_cuts, fragment_results=None):

    canvas.delete("segment_box")
    canvas.delete("auto_separator")

    for cut, row_y1, row_y2 in automatic_cuts:
        canvas.create_line(
            cut,
            max(0, row_y1 - 5),
            cut,
            min(CANVAS_HEIGHT, row_y2 + 5),
            fill="#ffb000",
            width=2,
            dash=(6, 5),
            tags="auto_separator"
        )

    result_index = 0

    for row_index, boxes in enumerate(row_boxes, start=1):

        for digit_index, (x1, y1, x2, y2) in enumerate(boxes, start=1):

            fragment_result = None

            if fragment_results is not None and result_index < len(fragment_results):
                fragment_result = fragment_results[result_index]

            is_unknown = bool(
                fragment_result is not None
                and fragment_result["is_unknown"]
            )
            color = "#ff2d2d" if is_unknown else "#00d8ff"
            box_width = 4 if is_unknown else 2
            marker = "?" if is_unknown else f"{row_index}.{digit_index}"

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=color,
                width=box_width,
                tags="segment_box"
            )

            canvas.create_text(
                x1 + 4,
                max(12, y1 - 12),
                text=marker,
                fill=color,
                anchor="w",
                font=("Arial", 13 if is_unknown else 11, "bold"),
                tags="segment_box"
            )

            result_index += 1

    canvas.tag_raise("manual_separator")


# ============================================================
# РАСПОЗНАВАНИЕ ВСЕЙ ПОСЛЕДОВАТЕЛЬНОСТИ
# ============================================================

def recognize_sequence(event=None):

    row_boxes, was_truncated, automatic_cuts = find_digit_boxes()

    if not row_boxes:
        result_label.config(text="Рисунок не найден")
        details_label.config(text="")
        status_label.config(text="Сначала нарисуйте одну или несколько цифр.")
        return

    digit_images = []
    valid_boxes = []
    valid_row_boxes = []
    row_lengths = []

    for boxes in row_boxes:

        valid_in_row = 0
        valid_boxes_in_row = []

        for box in boxes:
            prepared = prepare_digit(box)

            if prepared is not None:
                digit_images.append(prepared)
                valid_boxes.append(box)
                valid_boxes_in_row.append(box)
                valid_in_row += 1

        if valid_in_row:
            row_lengths.append(valid_in_row)
            valid_row_boxes.append(valid_boxes_in_row)

    if not digit_images:
        result_label.config(text="Не удалось выделить цифры")
        return

    batch = torch.stack(
        [ToTensor()(digit_image) for digit_image in digit_images]
    ).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        digit_probabilities = torch.sigmoid(unknown_detector(batch))

    predicted_digits = predictions.tolist()
    sequences = []
    detail_lines = []
    fragment_results = []
    prediction_index = 0

    for row_index, row_length in enumerate(row_lengths, start=1):

        row_digits = []
        row_details = []

        for digit_index in range(1, row_length + 1):
            digit = predicted_digits[prediction_index]
            confidence = (
                probabilities[prediction_index, digit].item()
                * 100
            )
            digit_probability = digit_probabilities[prediction_index].item()
            is_unknown = digit_probability < unknown_threshold

            fragment_results.append(
                {
                    "is_unknown": is_unknown,
                    "digit": digit,
                    "confidence": confidence,
                    "digit_probability": digit_probability * 100,
                }
            )

            if is_unknown:
                row_digits.append("?")
                row_details.append(
                    f"{row_index}.{digit_index}: неизвестный символ "
                    f"(похожесть на цифру {digit_probability * 100:.2f}%; "
                    f"предположение CNN {digit} отклонено)"
                )
            else:
                row_digits.append(str(digit))
                row_details.append(
                    f"{row_index}.{digit_index}: {digit} ({confidence:.2f}%; "
                    f"цифра {digit_probability * 100:.2f}%)"
                )
            prediction_index += 1

        sequences.append("".join(row_digits))
        detail_lines.append("   |   ".join(row_details))

    show_boxes(valid_row_boxes, automatic_cuts, fragment_results)

    save_diagnostic_strip(digit_images)

    if len(sequences) == 1:
        result_text = f"Результат: {sequences[0]}"
    else:
        result_text = "Результат по строкам:\n" + "\n".join(
            f"{index}: {sequence}"
            for index, sequence in enumerate(sequences, start=1)
        )

    result_label.config(text=result_text)
    details_label.config(text="\n".join(detail_lines))

    wide_fragments = 0

    for x1, y1, x2, y2 in valid_boxes:
        width = x2 - x1
        height = max(1, y2 - y1)

        if width > height * 1.25:
            wide_fragments += 1

    messages = [
        f"Найдено строк: {len(row_lengths)}, цифр: {len(digit_images)}.",
        "Голубая рамка — цифра; красная рамка и ? — неизвестный символ."
    ]

    unknown_count = sum(
        1 for fragment_result in fragment_results
        if fragment_result["is_unknown"]
    )

    if unknown_count:
        messages.append(
            f"Неизвестных символов: {unknown_count}; программа не стала их угадывать."
        )

    if automatic_cuts:
        messages.append(
            "Оранжевые линии — автоматическое разделение слипшихся цифр."
        )

    if manual_separators:
        messages.append(
            "Красные линии — разделители, поставленные пользователем."
        )

    if wide_fragments:
        messages.append(
            "Есть широкий фрагмент: возможно, некоторые цифры соприкасаются."
        )

    if was_truncated:
        messages.append(f"Обработаны только первые {MAX_DIGITS} цифр.")

    status_label.config(text=" ".join(messages))


# ============================================================
# ИНТЕРФЕЙС
# ============================================================

title_label = tk.Label(
    root,
    text="Распознавание цифр с отказом от угадывания",
    font=("Arial", 28, "bold")
)
title_label.pack(pady=(16, 4))

instruction_label = tk.Label(
    root,
    text=(
        "Можно писать в любом месте и в несколько строк. Порядок: сверху вниз, "
        "в каждой строке слева направо. Неизвестные символы выделяются красным. "
        "Слипшиеся символы разделяйте правой кнопкой мыши."
    ),
    font=("Arial", 14)
)
instruction_label.pack(pady=(0, 10))

canvas = tk.Canvas(
    root,
    width=CANVAS_WIDTH,
    height=CANVAS_HEIGHT,
    bg="black",
    highlightthickness=2,
    highlightbackground="#777777",
    cursor="crosshair"
)
canvas.pack()

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", stop_draw)
canvas.bind("<Button-3>", add_manual_separator)

controls_frame = tk.Frame(root)
controls_frame.pack(pady=12)

predict_button = tk.Button(
    controls_frame,
    text="Распознать последовательность",
    command=recognize_sequence,
    font=("Arial", 14, "bold"),
    width=29,
    height=2
)
predict_button.pack(side=tk.LEFT, padx=8)

clear_button = tk.Button(
    controls_frame,
    text="Очистить",
    command=clear_canvas,
    font=("Arial", 14),
    width=14,
    height=2
)
clear_button.pack(side=tk.LEFT, padx=8)

gap_label = tk.Label(
    controls_frame,
    text="Граница цифр, пикс.:",
    font=("Arial", 12)
)
gap_label.pack(side=tk.LEFT, padx=(22, 5))

split_gap_var = tk.StringVar(value=str(DEFAULT_SPLIT_GAP))

split_gap_spinbox = tk.Spinbox(
    controls_frame,
    from_=3,
    to=80,
    textvariable=split_gap_var,
    width=5,
    justify="center",
    font=("Arial", 13)
)
split_gap_spinbox.pack(side=tk.LEFT)

row_gap_label = tk.Label(
    controls_frame,
    text="Граница строк:",
    font=("Arial", 12)
)
row_gap_label.pack(side=tk.LEFT, padx=(18, 5))

row_gap_var = tk.StringVar(value=str(DEFAULT_ROW_GAP))

row_gap_spinbox = tk.Spinbox(
    controls_frame,
    from_=3,
    to=100,
    textvariable=row_gap_var,
    width=5,
    justify="center",
    font=("Arial", 13)
)
row_gap_spinbox.pack(side=tk.LEFT)

result_label = tk.Label(
    root,
    text="Нарисуйте несколько цифр",
    font=("Arial", 26, "bold"),
    fg="#164f86"
)
result_label.pack(pady=(4, 2))

details_label = tk.Label(
    root,
    text="",
    font=("Consolas", 12),
    wraplength=max(800, CANVAS_WIDTH - 40),
    justify=tk.CENTER
)
details_label.pack(pady=2)

status_label = tk.Label(
    root,
    text="Можно писать в любом месте и в несколько строк.",
    font=("Arial", 11),
    fg="#555555",
    wraplength=max(800, CANVAS_WIDTH - 40),
    justify=tk.CENTER
)
status_label.pack(pady=(3, 4))

model_info_label = tk.Label(
    root,
    text=(
        f"Модель: {current_model_file}   |   Устройство: {device}   |   "
        f"Порог цифры: {unknown_threshold:.1%}   |   Веса зафиксированы"
    ),
    font=("Arial", 10),
    fg="#777777"
)
model_info_label.pack(pady=(0, 8))

root.bind("<Return>", recognize_sequence)
root.bind("<Control-l>", clear_canvas)
root.bind("<Control-L>", clear_canvas)


try:
    root.mainloop()
except KeyboardInterrupt:
    pass
