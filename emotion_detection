import cv2
import torch
from torchvision.transforms import transforms
import numpy as np

# Загрузка модели классификатора эмоций
model = torch.load('path_to_emotion_model.pt')
model.eval()

# Список классов эмоций
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Загрузка каскадного классификатора для определения лица
face_cascade = cv2.CascadeClassifier('path_to_face_cascade.xml')

# Загрузка изображения
image = cv2.imread('path_to_image.jpg')

# Преобразование изображения в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лица на изображении
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Обработка каждого обнаруженного лица
for (x, y, w, h) in faces:
    # Извлечение области интереса (лица) изображения
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    # Преобразование области интереса в тензор PyTorch
    transform = transforms.Compose([transforms.ToTensor()])
    roi = transform(roi_gray).unsqueeze(0)
    
    # Классификация эмоции
    with torch.no_grad():
        prediction = model(roi)
    emotion_label = emotion_labels[prediction.argmax().item()]
    
    # Отображение результата на изображении
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Отображение и сохранение результата
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
