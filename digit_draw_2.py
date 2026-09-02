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
MIN_FOREGROUND_PIXELS = 3
MAX_DIGITS = 30
MIN_CNN_CONFIDENCE_PERCENT = 50.0


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
    details_label.config(text="")
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

    batch = torch.stack(
        [ToTensor()(digit_image) for digit_image in digit_images]
    ).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        digit_probabilities = torch.sigmoid(unknown_detector(batch))

    predicted_digits = predictions.tolist()
    sequence_parts = []
    details = []
    fragment_results = []

    for index, digit in enumerate(predicted_digits):
        confidence = probabilities[index, digit].item() * 100
        digit_probability = digit_probabilities[index].item()
        detector_rejected = digit_probability < unknown_threshold
        low_cnn_confidence = confidence < MIN_CNN_CONFIDENCE_PERCENT
        is_unknown = detector_rejected or low_cnn_confidence

        fragment_results.append(
            {
                "is_unknown": is_unknown,
                "digit": digit,
                "confidence": confidence,
                "digit_probability": digit_probability * 100,
                "detector_rejected": detector_rejected,
                "low_cnn_confidence": low_cnn_confidence,
            }
        )

        if is_unknown:
            reasons = []

            if detector_rejected:
                reasons.append(
                    f"детектор цифры {digit_probability * 100:.2f}%"
                )

            if low_cnn_confidence:
                reasons.append(f"уверенность CNN {confidence:.2f}% < 50%")

            sequence_parts.append("?")
            details.append(
                f"{index + 1}: неизвестный "
                f"({'; '.join(reasons)}; предположение {digit} отклонено)"
            )
        else:
            sequence_parts.append(str(digit))
            details.append(
                f"{index + 1}: {digit} ({confidence:.2f}%; "
                f"цифра {digit_probability * 100:.2f}%)"
            )

    sequence = "".join(sequence_parts)
    show_boxes(valid_boxes, fragment_results)

    save_diagnostic_strip(digit_images)

    result_label.config(text=f"Результат: {sequence}")
    details_label.config(text="   |   ".join(details))

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
        "символы не должны касаться. Неизвестные символы выделяются красным."
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

