import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------------------------------------------
# Настройки
# ------------------------------------------------------------

SEED = 42
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.001
DATA_DIR = Path("data")
MODEL_FILE = Path("mnist_cnn_model.pth")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"

print(f"Устройство: {device}")
if use_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram_gb:.2f} GB")
    torch.backends.cudnn.benchmark = True
else:
    print("CUDA не найдена. Обучение будет выполняться на процессоре.")


# ------------------------------------------------------------
# Данные MNIST и аугментация
# ------------------------------------------------------------

train_transform = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees=10,
            translate=(0.08, 0.08),
            scale=(0.90, 1.10),
            shear=5,
        ),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform,
)

test_data = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=test_transform,
)

print(f"Train: {len(train_data)}")
print(f"Test: {len(test_data)}")

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Надёжно работает в Windows без дополнительных настроек.
    pin_memory=use_cuda,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=use_cuda,
)


# ------------------------------------------------------------
# CNN. Этот же класс затем используем в программе рисования.
# ------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),       # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),       # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print()
print(model)


def train_one_epoch() -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=use_cuda)
        labels = labels.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_index % 200 == 0:
            print(
                f"Batch {batch_index:4d}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return average_loss, accuracy


@torch.inference_mode()
def evaluate() -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=use_cuda)
        labels = labels.to(device, non_blocking=use_cuda)

        logits = model(images)
        loss = loss_fn(logits, labels)

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return average_loss, accuracy


best_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):
    print()
    print("=" * 60)
    print(f"Epoch {epoch}/{EPOCHS}")
    print("=" * 60)

    train_loss, train_accuracy = train_one_epoch()
    test_loss, test_accuracy = evaluate()

    print()
    print(f"Train loss:     {train_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test loss:      {test_loss:.4f}")
    print(f"Test accuracy:  {test_accuracy:.2f}%")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), MODEL_FILE)
        print(f"✓ Новая лучшая модель сохранена ({best_accuracy:.2f}%)")


print()
print("=" * 60)
print("Обучение завершено.")
print(f"Лучшая точность: {best_accuracy:.2f}%")
print(f"Модель: {MODEL_FILE.resolve()}")

