import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision

# Путь к папке с данными
data_dir = './FER-2013'

# Трансформации изображений
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Загрузка данных для обучения и валидации
train_dataset = ImageFolder(root=data_dir + '/train', transform=transform,
                            loader=torchvision.datasets.folder.default_loader)
test_dataset = ImageFolder(root=data_dir + '/test', transform=transform,
                           loader=torchvision.datasets.folder.default_loader)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Определение модели
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class EmotionClassifier_second(nn.Module):
    def __init__(self):
        super(EmotionClassifier_second, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(64 * 12 * 12, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x

model = EmotionClassifier()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение первой модели
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / len(train_dataset)

    # Валидация модели
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    test_accuracy = correct_predictions / len(test_dataset)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}')

model_one = test_accuracy
# Сохранение первой модели
#torch.save(model.state_dict(), 'emotion_classifier.pt')

model_second = EmotionClassifier_second()

# Определение функции потерь и оптимизатора для второй модели
optimizer_second = optim.Adam(model_second.parameters(), lr=0.001)

# Обучение второй модели
for epoch in range(num_epochs):
    model_second.train()
    running_loss = 0.0
    correct_predictions = 0

    for images, labels in train_loader:
        optimizer_second.zero_grad()
        outputs = model_second(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_second.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / len(train_dataset)

    # Валидация модели
    model_second.eval()
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_second(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    test_accuracy = correct_predictions / len(test_dataset)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}')

# Сохранение второй модели, если ее точность выше первой модели
if test_accuracy > model_one:
    torch.save(model_second.state_dict(), 'emotion_classifier.pt')
    print("first model is better, so it is saved.")
else:
    print("second model is better, so it is saved.")
    torch.save(model.state_dict(), './emotion_classifier.pt')
    # You can keep the first model or delete the previously saved 'emotion_classifier.pt' file

