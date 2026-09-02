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

BRUSH_SIZE = 7
DEFAULT_SPLIT_GAP = 1
MIN_FOREGROUND_PIXELS = 2
MAX_DIGITS = 30
MIN_CNN_CONFIDENCE_PERCENT = 50.0
MIN_DIGIT_PROBABILITY_PERCENT = 50.0
AUTO_SPLIT_MIN_ASPECT_RATIO = 0.82
AUTO_SPLIT_MAX_PARTS = 6
AUTO_SPLIT_MIN_AVERAGE_CONFIDENCE = 50.0
AUTO_SPLIT_MAX_CUT_DENSITY = 0.42


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
    """Отдельная CNN: цифра (1) или неизвестный символ (0)."""

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
            "Сначала запустите train_unknown_detector.py."
        )

    if not os.path.exists(UNKNOWN_CONFIG_FILE):
        raise FileNotFoundError(
            f"Не найден {UNKNOWN_CONFIG_FILE}.\n"
            "Сначала запустите train_unknown_detector.py."
        )

    with open(UNKNOWN_CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    threshold = float(config.get("digit_probability_threshold", 0.50))
    threshold = max(0.01, min(threshold, 0.99))

    detector = UnknownDetector().to(device)
    detector_state = torch.load(
        UNKNOWN_DETECTOR_FILE,
        map_location=device,
        weights_only=True
    )
    detector.load_state_dict(detector_state)
    detector.eval()

    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    return detector, threshold


model, current_model_file = load_model()
unknown_detector, unknown_threshold = load_unknown_detector()

print("Устройство:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Модель:", current_model_file)
print("Детектор неизвестных символов:", UNKNOWN_DETECTOR_FILE)
print(f"Порог 'это цифра': {unknown_threshold:.6f}")
print("Режим: только распознавание, веса модели не изменяются")


# ============================================================
# ОКНО И ХОЛСТ
# ============================================================

root = tk.Tk()
root.title("Распознавание цифр и неизвестных символов — CNN")
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


# ============================================================
# РИСОВАНИЕ
# ============================================================

def start_draw(event):

    global last_x, last_y

    canvas.delete("segment_box")
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


def clear_canvas(event=None):

    global drawing_image, drawing

    canvas.delete("all")

    drawing_image = Image.new(
        "L",
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        color=0
    )
    drawing = ImageDraw.Draw(drawing_image)

    result_label.config(text="Нарисуйте несколько цифр")
    show_colored_details([])
    status_label.config(
        text="Пишите слева направо и оставляйте промежутки между цифрами."
    )


# ============================================================
# СЕГМЕНТАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТИ
# ============================================================

def get_foreground_columns():

    pixels = drawing_image.load()
    columns = []

    for x in range(CANVAS_WIDTH):

        has_ink = False

        for y in range(CANVAS_HEIGHT):
            if pixels[x, y] > 0:
                has_ink = True
                break

        if has_ink:
            columns.append(x)

    return columns


def make_column_runs(columns):

    if not columns:
        return []

    runs = []
    start = columns[0]
    previous = columns[0]

    for column in columns[1:]:

        if column == previous + 1:
            previous = column
            continue

        runs.append((start, previous))
        start = column
        previous = column

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


def find_digit_boxes():

    try:
        split_gap = int(split_gap_var.get())
    except ValueError:
        split_gap = DEFAULT_SPLIT_GAP
        split_gap_var.set(str(DEFAULT_SPLIT_GAP))

    split_gap = max(3, min(split_gap, 80))

    columns = get_foreground_columns()
    runs = make_column_runs(columns)
    groups = merge_nearby_runs(runs, split_gap)

    pixels = drawing_image.load()
    boxes = []

    for x1, x2 in groups:

        active_rows = []
        foreground_count = 0

        for y in range(CANVAS_HEIGHT):
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
        boxes.append(
            (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(CANVAS_WIDTH, x2 + padding + 1),
                min(CANVAS_HEIGHT, y2 + padding + 1)
            )
        )

    return boxes[:MAX_DIGITS], len(boxes) > MAX_DIGITS


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


def show_boxes(boxes, fragment_results=None):

    canvas.delete("segment_box")

    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):

        fragment_result = None

        if fragment_results is not None and index - 1 < len(fragment_results):
            fragment_result = fragment_results[index - 1]

        is_unknown = bool(
            fragment_result is not None
            and fragment_result["is_unknown"]
        )
        color = "#ff2d2d" if is_unknown else "#00d8ff"
        marker = (
            "?"
            if is_unknown
            else str(fragment_result["digit"])
            if fragment_result is not None
            else str(index)
        )

        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline=color,
            width=4 if is_unknown else 2,
            tags="segment_box"
        )

        canvas.create_text(
            x1 + 4,
            max(12, y1 - 12),
            text=marker,
            fill=color,
            anchor="w",
            font=("Arial", 14 if is_unknown else 11, "bold"),
            tags="segment_box"
        )


def show_colored_details(detail_rows):
    """Показывает проценты ниже 70% красным цветом."""

    details_label.config(state=tk.NORMAL)
    details_label.delete("1.0", tk.END)

    for row_index, row_parts in enumerate(detail_rows):
        if row_index:
            details_label.insert(tk.END, "   |   ", "normal")

        for text, tag in row_parts:
            details_label.insert(tk.END, text, tag)

    details_label.tag_add("center", "1.0", tk.END)
    details_label.config(state=tk.DISABLED)


def classify_digit_images(images):

    if not images:
        return []

    batch = torch.stack(
        [ToTensor()(digit_image) for digit_image in images]
    ).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        digit_probabilities = torch.sigmoid(unknown_detector(batch))

    results = []

    for index, digit in enumerate(predictions.tolist()):
        confidence = probabilities[index, digit].item() * 100
        digit_probability = digit_probabilities[index].item()
        digit_probability_percent = digit_probability * 100
        effective_detector_threshold_percent = max(
            unknown_threshold * 100,
            MIN_DIGIT_PROBABILITY_PERCENT,
        )
        detector_rejected = (
            digit_probability_percent < effective_detector_threshold_percent
        )
        low_cnn_confidence = confidence < MIN_CNN_CONFIDENCE_PERCENT

        results.append(
            {
                "is_unknown": detector_rejected or low_cnn_confidence,
                "digit": digit,
                "confidence": confidence,
                "digit_probability": digit_probability_percent,
                "detector_rejected": detector_rejected,
                "detector_threshold_percent": effective_detector_threshold_percent,
                "low_cnn_confidence": low_cnn_confidence,
            }
        )

    return results


def get_tight_ink_bounds(box):

    x1, y1, x2, y2 = box
    pixels = drawing_image.load()
    active_x = []
    active_y = []

    for y in range(y1, y2):
        for x in range(x1, x2):
            if pixels[x, y] > 0:
                active_x.append(x)
                active_y.append(y)

    if not active_x:
        return None

    return min(active_x), min(active_y), max(active_x) + 1, max(active_y) + 1


def get_vertical_projection(box):

    x1, y1, x2, y2 = box
    pixels = drawing_image.load()

    return [
        sum(1 for y in range(y1, y2) if pixels[x, y] > 0)
        for x in range(x1, x2)
    ]


def smooth_projection_value(projection, local_x):

    values = []

    for offset in (-2, -1, 0, 1, 2):
        position = local_x + offset

        if 0 <= position < len(projection):
            weight = 3 if offset == 0 else 2 if abs(offset) == 1 else 1
            values.extend([projection[position]] * weight)

    return sum(values) / max(1, len(values))


def find_valley_cut(projection, ideal_local_x, radius, min_x, max_x):

    search_start = max(min_x, ideal_local_x - radius)
    search_end = min(max_x, ideal_local_x + radius)

    if search_start > search_end:
        return None

    return min(
        range(search_start, search_end + 1),
        key=lambda x: (
            smooth_projection_value(projection, x),
            abs(x - ideal_local_x),
        )
    )


def make_tight_sub_box(parent_box, start_x, end_x):

    _, parent_y1, _, parent_y2 = parent_box
    pixels = drawing_image.load()
    active_x = []
    active_y = []
    foreground_count = 0

    for y in range(parent_y1, parent_y2):
        for x in range(start_x, end_x):
            if pixels[x, y] > 0:
                foreground_count += 1
                active_x.append(x)
                active_y.append(y)

    if foreground_count < MIN_FOREGROUND_PIXELS or not active_x:
        return None

    padding = 2
    return (
        max(start_x, min(active_x) - padding),
        max(0, min(active_y) - padding),
        min(end_x, max(active_x) + padding + 1),
        min(CANVAS_HEIGHT, max(active_y) + padding + 1),
    )


def try_split_unknown_box(box):

    tight_box = get_tight_ink_bounds(box)

    if tight_box is None:
        return None

    ink_x1, ink_y1, ink_x2, ink_y2 = tight_box
    width = ink_x2 - ink_x1
    height = max(1, ink_y2 - ink_y1)

    if width / height < AUTO_SPLIT_MIN_ASPECT_RATIO:
        return None

    projection = get_vertical_projection(tight_box)
    maximum_projection = max(projection, default=0)

    if maximum_projection <= 0:
        return None

    minimum_part_width = max(6, round(height * 0.13))
    maximum_parts = min(
        AUTO_SPLIT_MAX_PARTS,
        width // minimum_part_width,
    )

    if maximum_parts < 2:
        return None

    best_candidate = None

    for part_count in range(2, maximum_parts + 1):
        expected_width = width / part_count
        radius = max(3, round(expected_width * 0.32))
        cuts = []
        valid_cuts = True

        for part_number in range(1, part_count):
            ideal = round(expected_width * part_number)
            minimum_cut = (
                minimum_part_width
                if not cuts
                else cuts[-1] + minimum_part_width
            )
            maximum_cut = width - minimum_part_width * (part_count - part_number)
            cut = find_valley_cut(
                projection,
                ideal,
                radius,
                minimum_cut,
                maximum_cut,
            )

            if cut is None:
                valid_cuts = False
                break

            cuts.append(cut)

        if not valid_cuts or not cuts:
            continue

        cut_densities = [
            smooth_projection_value(projection, cut) / maximum_projection
            for cut in cuts
        ]

        if max(cut_densities) > AUTO_SPLIT_MAX_CUT_DENSITY:
            continue

        boundaries = [0, *cuts, width]
        sub_boxes = []

        for index in range(part_count):
            sub_box = make_tight_sub_box(
                tight_box,
                ink_x1 + boundaries[index],
                ink_x1 + boundaries[index + 1],
            )

            if sub_box is None:
                sub_boxes = []
                break

            sub_boxes.append(sub_box)

        if len(sub_boxes) != part_count:
            continue

        sub_images = [prepare_digit(sub_box) for sub_box in sub_boxes]

        if any(image is None for image in sub_images):
            continue

        sub_results = classify_digit_images(sub_images)

        if any(result["is_unknown"] for result in sub_results):
            continue

        average_confidence = sum(
            result["confidence"] for result in sub_results
        ) / part_count

        if average_confidence < AUTO_SPLIT_MIN_AVERAGE_CONFIDENCE:
            continue

        average_digit_probability = sum(
            result["digit_probability"] for result in sub_results
        ) / part_count
        average_cut_density = sum(cut_densities) / len(cut_densities)
        score = (
            average_confidence
            + average_digit_probability * 0.20
            - average_cut_density * 20
            - (part_count - 2) * 1.5
        )

        candidate = {
            "boxes": sub_boxes,
            "images": sub_images,
            "results": sub_results,
            "score": score,
            "forced_middle_split": False,
        }

        if best_candidate is None or score > best_candidate["score"]:
            best_candidate = candidate

    return best_candidate


def split_unknown_box_strictly_in_middle(box):
    """Резервный вариант: один разрез ровно по центру красного фрагмента."""

    tight_box = get_tight_ink_bounds(box)

    if tight_box is None:
        return None

    ink_x1, _, ink_x2, _ = tight_box
    width = ink_x2 - ink_x1

    if width < 4:
        return None

    middle_x = ink_x1 + width // 2
    left_box = make_tight_sub_box(tight_box, ink_x1, middle_x)
    right_box = make_tight_sub_box(tight_box, middle_x, ink_x2)

    if left_box is None or right_box is None:
        return None

    sub_boxes = [left_box, right_box]
    sub_images = [prepare_digit(sub_box) for sub_box in sub_boxes]

    if any(image is None for image in sub_images):
        return None

    return {
        "boxes": sub_boxes,
        "images": sub_images,
        "results": classify_digit_images(sub_images),
        "score": 0.0,
        "forced_middle_split": True,
    }


# ============================================================
# РАСПОЗНАВАНИЕ ВСЕЙ ПОСЛЕДОВАТЕЛЬНОСТИ
# ============================================================

def recognize_sequence(event=None):

    boxes, was_truncated = find_digit_boxes()

    if not boxes:
        result_label.config(text="Рисунок не найден")
        details_label.config(text="")
        status_label.config(text="Сначала нарисуйте одну или несколько цифр.")
        return

    digit_images = []
    valid_boxes = []

    for box in boxes:
        prepared = prepare_digit(box)

        if prepared is not None:
            digit_images.append(prepared)
            valid_boxes.append(box)

    if not digit_images:
        result_label.config(text="Не удалось выделить цифры")
        return

    initial_results = classify_digit_images(digit_images)
    final_boxes = []
    final_images = []
    fragment_results = []
    safe_split_group_count = 0
    forced_split_group_count = 0

    for box, digit_image, initial_result in zip(
        valid_boxes,
        digit_images,
        initial_results,
    ):
        split_candidate = None

        if initial_result["is_unknown"]:
            split_candidate = try_split_unknown_box(box)

            if split_candidate is None:
                split_candidate = split_unknown_box_strictly_in_middle(box)

        if (
            split_candidate is not None
            and len(final_boxes) + len(split_candidate["boxes"]) <= MAX_DIGITS
        ):
            final_boxes.extend(split_candidate["boxes"])
            final_images.extend(split_candidate["images"])
            fragment_results.extend(split_candidate["results"])

            if split_candidate["forced_middle_split"]:
                forced_split_group_count += 1
            else:
                safe_split_group_count += 1
        else:
            final_boxes.append(box)
            final_images.append(digit_image)
            fragment_results.append(initial_result)

    valid_boxes = final_boxes
    digit_images = final_images
    sequence_parts = []
    detail_rows = []

    for index, result in enumerate(fragment_results):
        digit = result["digit"]
        confidence = result["confidence"]
        digit_probability = result["digit_probability"]
        detector_rejected = result["detector_rejected"]
        low_cnn_confidence = result["low_cnn_confidence"]
        is_unknown = result["is_unknown"]

        if is_unknown:
            reasons = []

            if detector_rejected:
                reasons.append(
                    f"детектор цифры {digit_probability:.2f}% "
                    f"< {result['detector_threshold_percent']:.0f}%"
                )

            if low_cnn_confidence:
                reasons.append(
                    f"уверенность CNN {confidence:.2f}% "
                    f"< {MIN_CNN_CONFIDENCE_PERCENT:.0f}%"
                )

            sequence_parts.append("?")
            detail_rows.append([
                (
                    f"{index + 1}: неизвестный "
                    f"({'; '.join(reasons)}; предположение {digit} отклонено)",
                    "warning",
                )
            ])
        else:
            sequence_parts.append(str(digit))
            detail_rows.append([
                (f"{index + 1}: {digit} (", "normal"),
                (
                    f"{confidence:.2f}%",
                    "warning"
                    if confidence < MIN_CNN_CONFIDENCE_PERCENT
                    else "normal",
                ),
                ("; цифра ", "normal"),
                (
                    f"{digit_probability:.2f}%",
                    "warning"
                    if digit_probability < MIN_DIGIT_PROBABILITY_PERCENT
                    else "normal",
                ),
                (")", "normal"),
            ])

    sequence = "".join(sequence_parts)
    show_boxes(valid_boxes, fragment_results)

    save_diagnostic_strip(digit_images)

    result_label.config(text=f"Результат: {sequence}")
    show_colored_details(detail_rows)

    wide_fragments = 0

    for x1, y1, x2, y2 in valid_boxes:
        width = x2 - x1
        height = max(1, y2 - y1)

        if width > height * 1.25:
            wide_fragments += 1

    messages = [
        f"Найдено фрагментов: {len(digit_images)}.",
        "Голубая рамка — цифра; красная рамка и ? — неизвестный символ."
    ]

    if safe_split_group_count:
        messages.append(
            f"Безопасно разделено слипшихся групп: {safe_split_group_count}."
        )

    if forced_split_group_count:
        messages.append(
            f"Принудительно разделено по центру: {forced_split_group_count}."
        )

    unknown_count = sum(
        1 for result in fragment_results
        if result["is_unknown"]
    )

    if unknown_count:
        messages.append(
            f"Неизвестных символов: {unknown_count}; они не были угаданы как цифры."
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
    text="Распознавание цифр без угадывания неизвестных символов",
    font=("Arial", 20, "bold")
)
title_label.pack(pady=(16, 4))

instruction_label = tk.Label(
    root,
    text=(
        "Пишите слева направо. Между цифрами оставляйте 25–30 пикселей; "
        "слипшиеся цифры программа разделяет автоматически или строго по центру. "
        "CNN или показатель «цифра» ниже 70% выделяется красным."
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

# gap_label = tk.Label(
#     controls_frame,
#     text="Граница цифр, пикс.:",
#     font=("Arial", 12)
# )
# gap_label.pack(side=tk.LEFT, padx=(22, 5))

split_gap_var = tk.StringVar(value=str(DEFAULT_SPLIT_GAP))

# split_gap_spinbox = tk.Spinbox(
#     controls_frame,
#     from_=3,
#     to=80,
#     textvariable=split_gap_var,
#     width=5,
#     justify="center",
#     font=("Arial", 13)
# )
# split_gap_spinbox.pack(side=tk.LEFT)

result_label = tk.Label(
    root,
    text="Нарисуйте несколько цифр",
    font=("Arial", 26, "bold"),
    fg="#164f86"
)
result_label.pack(pady=(4, 2))

details_label = tk.Text(
    root,
    font=("Consolas", 12),
    width=max(80, (CANVAS_WIDTH - 40) // 10),
    height=3,
    wrap=tk.WORD,
    borderwidth=0,
    highlightthickness=0,
    background=root.cget("background"),
    cursor="arrow"
)
details_label.tag_configure("normal", foreground="#111111")
details_label.tag_configure("warning", foreground="#ff2d2d")
details_label.tag_configure("center", justify=tk.CENTER)
details_label.config(state=tk.DISABLED)
details_label.pack(pady=2)

status_label = tk.Label(
    root,
    text="Пишите слева направо и оставляйте промежутки между цифрами.",
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
        f"Порог цифры: {unknown_threshold:.2%}   |   "
        f"Минимум CNN: {MIN_CNN_CONFIDENCE_PERCENT:.0f}%   |   "
        f"Минимум детектора: {MIN_DIGIT_PROBABILITY_PERCENT:.0f}%   |   "
        "Веса зафиксированы"
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

