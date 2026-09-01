import tkinter as tk

import torch
from torch import nn

from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor


# ============================================================
# 1. НАСТРОЙКА PYTORCH
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Устройство:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. АРХИТЕКТУРА НАШЕЙ НЕЙРОСЕТИ
#
# Она ОБЯЗАТЕЛЬНО должна совпадать с архитектурой,
# которую мы использовали при обучении MNIST.
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(

            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):

        x = self.flatten(x)

        return self.network(x)


# ============================================================
# 3. ЗАГРУЖАЕМ ОБУЧЕННУЮ МОДЕЛЬ
# ============================================================

model = NeuralNetwork().to(device)

model.load_state_dict(
    torch.load(
        "mnist_model.pth",
        map_location=device
    )
)

model.eval()

print("Модель успешно загружена.")


# ============================================================
# 4. ОСНОВНОЕ ОКНО
# ============================================================

root = tk.Tk()

root.title("Распознавание цифр — PyTorch")

# Открываем окно развёрнутым на весь экран
root.state("zoomed")

# Разрешаем изменение размера окна
root.resizable(True, True)


# Размер области рисования
CANVAS_SIZE = 280

# Толщина кисти
BRUSH_SIZE = 14


# ============================================================
# 5. CANVAS — ОБЛАСТЬ ДЛЯ РИСОВАНИЯ
# ============================================================

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    cursor="cross"
)

canvas.pack(
    padx=20,
    pady=(20, 10)
)


# ============================================================
# 6. СОЗДАЁМ ВНУТРЕННЕЕ ИЗОБРАЖЕНИЕ
#
# Canvas нужен человеку для отображения.
#
# PIL Image нужен нейросети.
# ============================================================

image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    color=0
)

draw = ImageDraw.Draw(image)


# ============================================================
# 7. ПЕРЕМЕННЫЕ ДЛЯ МЫШИ
# ============================================================

last_x = None
last_y = None


# ============================================================
# 8. НАЧАЛО РИСОВАНИЯ
# ============================================================

def start_draw(event):

    global last_x, last_y

    last_x = event.x
    last_y = event.y

    radius = BRUSH_SIZE // 2

    # Рисуем точку на Canvas
    canvas.create_oval(
        event.x - radius,
        event.y - radius,
        event.x + radius,
        event.y + radius,
        fill="white",
        outline="white"
    )

    # То же самое рисуем на PIL Image
    draw.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius
        ),
        fill=255
    )


# ============================================================
# 9. РИСОВАНИЕ ПРИ ДВИЖЕНИИ МЫШИ
# ============================================================

def draw_digit(event):

    global last_x, last_y

    if last_x is None or last_y is None:
        return

    # Рисуем линию на Canvas
    canvas.create_line(
        last_x,
        last_y,
        event.x,
        event.y,
        fill="white",
        width=BRUSH_SIZE,
        capstyle=tk.ROUND,
        smooth=True
    )

    # Рисуем такую же линию на PIL Image
    draw.line(
        (
            last_x,
            last_y,
            event.x,
            event.y
        ),
        fill=255,
        width=BRUSH_SIZE
    )

    # Делаем конец линии круглым
    radius = BRUSH_SIZE // 2

    draw.ellipse(
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


# ============================================================
# 10. КОНЕЦ РИСОВАНИЯ
# ============================================================

def stop_draw(event):

    global last_x, last_y

    last_x = None
    last_y = None


# ============================================================
# 11. ПОДГОТОВКА КАРТИНКИ К MNIST
# ============================================================

def prepare_image():

    bbox = image.getbbox()

    if bbox is None:
        return None

    # Обрезаем пустое пространство
    cropped = image.crop(bbox)

    width, height = cropped.size

    # MNIST: сама цифра занимает примерно 20x20
    max_size = 20

    scale = max_size / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    cropped = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Создаём изображение MNIST
    mnist_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # Первичное центрирование
    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    mnist_image.paste(
        cropped,
        (x, y)
    )

    # -------------------------------------------------
    # Центрирование по центру массы
    # -------------------------------------------------

    pixels = mnist_image.load()

    total = 0
    sum_x = 0
    sum_y = 0

    for py in range(28):
        for px in range(28):

            value = pixels[px, py]

            total += value
            sum_x += px * value
            sum_y += py * value

    if total > 0:

        center_x = sum_x / total
        center_y = sum_y / total

        shift_x = round(13.5 - center_x)
        shift_y = round(13.5 - center_y)

        centered = Image.new(
            "L",
            (28, 28),
            0
        )

        centered.paste(
            mnist_image,
            (shift_x, shift_y)
        )

        mnist_image = centered

    # Для диагностики
    mnist_image.save("last_input.png")

    return mnist_image

# ============================================================
# 12. РАСПОЗНАВАНИЕ
# ============================================================

def predict():

    mnist_image = prepare_image()

    if mnist_image is None:

        result_label.config(
            text="Сначала нарисуйте цифру"
        )

        confidence_label.config(
            text=""
        )

        probability_label.config(
            text=""
        )

        return


    # PIL Image -> PyTorch Tensor
    tensor = ToTensor()(mnist_image)

    # Было:
    #
    # [1, 28, 28]
    #
    # Делаем batch:
    #
    # [1, 1, 28, 28]

    tensor = tensor.unsqueeze(0)

    # Переносим на GPU
    tensor = tensor.to(device)


    # --------------------------------------------------------
    # ПРЕДСКАЗАНИЕ
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_digit = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_digit
        ].item()


    # --------------------------------------------------------
    # РЕЗУЛЬТАТ
    # --------------------------------------------------------

    result_label.config(
        text=f"AI считает, что это: {predicted_digit}"
    )

    confidence_label.config(
        text=f"Уверенность: {confidence * 100:.2f}%"
    )


    # --------------------------------------------------------
    # ПОКАЗЫВАЕМ ВЕРОЯТНОСТИ 0-9
    # --------------------------------------------------------

    text = ""

    for digit in range(10):

        probability = (
            probabilities[0, digit].item() * 100
        )

        text += (
            f"{digit}: {probability:6.2f}%"
        )

        if digit == 4:
            text += "\n"
        else:
            text += "     "


    probability_label.config(
        text=text
    )


# ============================================================
# 13. ОЧИСТКА
# ============================================================

def clear():

    global image, draw
    global last_x, last_y

    canvas.delete("all")

    image = Image.new(
        "L",
        (CANVAS_SIZE, CANVAS_SIZE),
        color=0
    )

    draw = ImageDraw.Draw(image)

    last_x = None
    last_y = None

    result_label.config(
        text="Нарисуйте цифру от 0 до 9"
    )

    confidence_label.config(
        text=""
    )

    probability_label.config(
        text=""
    )


# ============================================================
# 14. СОБЫТИЯ МЫШИ
# ============================================================

canvas.bind(
    "<Button-1>",
    start_draw
)

canvas.bind(
    "<B1-Motion>",
    draw_digit
)

canvas.bind(
    "<ButtonRelease-1>",
    stop_draw
)


# ============================================================
# 15. КНОПКИ
# ============================================================

button_frame = tk.Frame(root)

button_frame.pack(
    pady=10
)


predict_button = tk.Button(
    button_frame,
    text="Распознать",
    command=predict,
    font=("Arial", 12),
    width=14,
    height=2
)

predict_button.pack(
    side=tk.LEFT,
    padx=5
)


clear_button = tk.Button(
    button_frame,
    text="Очистить",
    command=clear,
    font=("Arial", 12),
    width=14,
    height=2
)

clear_button.pack(
    side=tk.LEFT,
    padx=5
)


# ============================================================
# 16. ВЫВОД РЕЗУЛЬТАТА
# ============================================================

result_label = tk.Label(
    root,
    text="Нарисуйте цифру от 0 до 9",
    font=("Arial", 18, "bold")
)

result_label.pack(
    pady=(10, 5)
)


confidence_label = tk.Label(
    root,
    text="",
    font=("Arial", 14)
)

confidence_label.pack()


probability_label = tk.Label(
    root,
    text="",
    font=("Consolas", 10),
    justify=tk.LEFT
)

probability_label.pack(
    padx=10,
    pady=(10, 20)
)


# ============================================================
# 17. ЗАПУСК ПРОГРАММЫ
# ============================================================

root.mainloop()