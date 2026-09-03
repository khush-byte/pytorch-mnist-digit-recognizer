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


model, current_model_file = load_model()

print("Устройство:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Модель:", current_model_file)
print("Режим: только распознавание, веса модели не изменяются")


# ============================================================
# ОКНО И ХОЛСТ
# ============================================================

root = tk.Tk()
root.title("Распознавание последовательности цифр — CNN")
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


def show_boxes(boxes, results=None):
    """
    Показывает рамку вокруг каждой выделенной цифры.

    Над рамкой отображается цифра и процент уверенности CNN.
    При уверенности ниже 50% рамка красная, а символ считается неизвестным.
    """

    canvas.delete("segment_box")

    for index, (x1, y1, x2, y2) in enumerate(boxes):

        result = None
        if results is not None and index < len(results):
            result = results[index]

        if result is None:
            color = "#00d8ff"
            marker = str(index + 1)
            confidence_text = ""
            is_unknown = False
        else:
            confidence = result["confidence"]
            is_unknown = confidence < MIN_CNN_CONFIDENCE_PERCENT

            if is_unknown:
                color = "#ff2d2d"
                marker = "?"
            else:
                color = "#00d8ff"
                marker = str(result["digit"])

            confidence_text = f"{confidence:.2f}%"

        label_text = marker

        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline=color,
            width=4 if is_unknown else 2,
            tags="segment_box"
        )

        # Значение распознавания и уверенность — сверху рамки.
        canvas.create_text(
            (x1 + x2) // 2,
            max(12, y1 - 8),
            text=label_text,
            fill=color,
            anchor="s",
            font=("Arial", 13, "bold"),
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
        status_label.config(
            text="Сначала нарисуйте одну или несколько цифр."
        )
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
        details_label.config(text="")
        return

    batch = torch.stack(
        [ToTensor()(digit_image) for digit_image in digit_images]
    ).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    predicted_digits = predictions.tolist()
    results = []

    for index, digit in enumerate(predicted_digits):
        confidence = probabilities[index, digit].item() * 100

        results.append(
            {
                "digit": digit,
                "confidence": confidence,
                "is_unknown": confidence < MIN_CNN_CONFIDENCE_PERCENT,
            }
        )

    # Показываем цифру и уверенность непосредственно сверху рамки.
    show_boxes(valid_boxes, results)

    sequence_parts = []
    details = []

    for index, result in enumerate(results):
        digit = result["digit"]
        confidence = result["confidence"]

        if result["is_unknown"]:
            sequence_parts.append("?")
            details.append(
                f"{index + 1}: неизвестный "
                f"(предположение {digit}, "
                f"уверенность {confidence:.2f}% < "
                f"{MIN_CNN_CONFIDENCE_PERCENT:.0f}%)"
            )
        else:
            sequence_parts.append(str(digit))
            details.append(
                f"{index + 1}: {digit} ({confidence:.2f}%)"
            )

    sequence = "".join(sequence_parts)

    save_diagnostic_strip(digit_images)

    result_label.config(text=f"Результат: {sequence}")
    details_label.config(text="   |   ".join(details))

    unknown_count = sum(
        1 for result in results
        if result["is_unknown"]
    )

    messages = [
        f"Найдено фрагментов: {len(digit_images)}.",
        f"Порог распознавания: {MIN_CNN_CONFIDENCE_PERCENT:.0f}%.",
        "Красная рамка и ? означают уверенность ниже 50%."
    ]

    if unknown_count:
        messages.append(f"Неизвестных символов: {unknown_count}.")

    if was_truncated:
        messages.append(
            f"Обработаны только первые {MAX_DIGITS} цифр."
        )

    status_label.config(text=" ".join(messages))


# ============================================================
# ИНТЕРФЕЙС
# ============================================================

title_label = tk.Label(
    root,
    text="Распознавание последовательности цифр",
    font=("Arial", 20, "bold")
)
title_label.pack(pady=(16, 4))

instruction_label = tk.Label(
    root,
    text=(
        "Пишите слева направо. Между цифрами оставляйте 25–30 пикселей; "
        "цифры не должны касаться. Над рамкой показываются цифра и уверенность. "
        'При уверенности ниже 50% рамка красная, а результат — "?".'
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


split_gap_var = tk.StringVar(value=str(DEFAULT_SPLIT_GAP))


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
        f"Минимальная уверенность: {MIN_CNN_CONFIDENCE_PERCENT:.0f}%   |   "
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

