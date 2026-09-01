import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# --------------------------------------------------
# 1. Выбираем устройство: GPU или CPU
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Устройство:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. Загружаем MNIST
# --------------------------------------------------

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# --------------------------------------------------
# 3. Создаём DataLoader
# --------------------------------------------------

batch_size = 64

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


print("Обучающих изображений:", len(train_data))
print("Тестовых изображений:", len(test_data))


# --------------------------------------------------
# 4. Создаём нейронную сеть
# --------------------------------------------------

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

        x = self.network(x)

        return x


# --------------------------------------------------
# 5. Создаём модель и переносим её на GPU
# --------------------------------------------------

model = NeuralNetwork().to(device)

print(model)


# --------------------------------------------------
# 6. Loss function
# --------------------------------------------------

loss_fn = nn.CrossEntropyLoss()


# --------------------------------------------------
# 7. Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 8. Функция обучения
# --------------------------------------------------

def train():

    model.train()

    total_loss = 0

    for batch, (images, labels) in enumerate(train_loader):

        # Переносим данные на GPU
        images = images.to(device)
        labels = labels.to(device)

        # Прямой проход
        predictions = model(images)

        # Считаем ошибку
        loss = loss_fn(predictions, labels)

        # Обнуляем старые градиенты
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Обновляем веса
        optimizer.step()

        total_loss += loss.item()

        if batch % 200 == 0:

            print(
                f"Batch: {batch:4d} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    return average_loss


# --------------------------------------------------
# 9. Функция проверки
# --------------------------------------------------

def test():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            predicted_classes = predictions.argmax(dim=1)

            correct += (
                predicted_classes == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total * 100

    return accuracy


# --------------------------------------------------
# 10. Обучение
# --------------------------------------------------

epochs = 5

for epoch in range(epochs):

    print()
    print("=" * 50)
    print(f"Epoch {epoch + 1}/{epochs}")
    print("=" * 50)

    loss = train()

    accuracy = test()

    print()
    print(f"Average loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")


# --------------------------------------------------
# 11. Сохраняем обученную модель
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "mnist_model.pth"
)

print()
print("Обучение завершено.")
print("Модель сохранена: mnist_model.pth")