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

# Создание изображения с надписью


image_with_text = image.copy()
draw = ImageDraw.Draw(image_with_text)
text = predicted_class
font_size = 24
text_width, text_height = draw.textsize(text)
text_x = (image_with_text.width - text_width) // 2
text_y = (image_with_text.height - text_height) // 2
draw.text((text_x, text_y), text, fill=(255, 255, 255))

# Сохранение изображения с надписью
output_path = './'
image_with_text.save(output_path)
# Вывод пути сохраненного изображения
print(f'Saved image with text: {output_path}')
