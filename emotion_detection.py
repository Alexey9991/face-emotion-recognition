import cv2
import torch
from torchvision.transforms import transforms
import numpy as np

# Загрузка модели классификатора эмоций
model = YourEmotionClassifierModel()
model.load_state_dict(torch.load('emotion_classifier.pt'))
model.eval()



# Список классов эмоций
#emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_labels = ['Angry']

# Загрузка каскадного классификатора для определения лица
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Загрузка изображения
image = cv2.imread('data_for_recognition/face-people-portrait-actor-hair-Person-Nicolas-Cage-man-beard-look-male-hairstyle-facial-hair-bristle-579421.jpg')

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
print(emotion_label)
# Отображение и сохранение результата
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
