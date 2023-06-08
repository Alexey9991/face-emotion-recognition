import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from train import EmotionClassifier

# Определение модели
model = EmotionClassifier()

# Загрузка предварительно обученной модели
state_dict = torch.load('emotion_classifier.pt')
model.load_state_dict(state_dict)
model.eval()

# Загрузка и предобработка изображения
image_path = './data_for_recognition/1.jpg'
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Предсказание класса изображения
with torch.no_grad():
    output = model(input_batch)

# Загрузка файла с классами
with open('emotion_classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Получение предсказанного класса
_, predicted_idx = torch.max(output, 1)
predicted_class = classes[predicted_idx.item()]

# Вывод предсказанного класса
print(f'Predicted class: {predicted_class}')

output_image = image.copy()
draw = ImageDraw.Draw(output_image)
font = ImageFont.load_default().font # Выбор шрифта и размера текста
text = f'Predicted class: {predicted_class}'
text_width, text_height = draw.textsize(text, font=font)
text_position = (10, 10)  # Позиция текста на изображении
text_color = (255, 255, 255)  # Цвет текста (белый)
draw.rectangle((text_position[0], text_position[1], text_position[0] + text_width, text_position[1] + text_height),
               fill=(0, 0, 0))  # Заливка прямоугольника под текстом черным цветом
draw.text(text_position, text, font=font, fill=text_color)

# Сохранение изображения с надписью
output_image_path = './processed_image.jpg'
output_image.save(output_image_path)