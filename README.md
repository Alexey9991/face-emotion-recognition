# face-emotion-recognition


# Описание проекта
## Название проекта: face-emotion-recognition
### Цель: Распознавания эмоционального состояния человека по фотографии 

Задачи:
1. Разработка архитектуры системы.
2. Подготовка исходных данных.
3. Анализ существующих моделей.
4. Обучение и оценка моделей на исходном датасете.
5. Выбор наиболее оптимальной модели.
6. Развертывание наилучшей модели.  

Датасет: [Kaggle dataset](https://www.kaggle.com/datasets/msambare/fer2013)  
Проект: [habr-posts-likes-prediction](https://github.com/Alexey9991/face-emotion-recognition)

### Целесообразность использования датасета для решения поставленной задачи:
Датасет содержит более 28 тысяч изображений семи классов различных эмоций человека "angry", "disgust", "fear", "happy", "neutral", "sad", "surprice".
![output](https://user-images.githubusercontent.com/97290990/233084081-9ff502bf-7061-4d73-8ff5-b09c3e0a6873.png)
Данный дасет занимает всего 56 MB объема, в связи с этим будет возможно его обучение и хранение на облаке. Разрешение фотографий 48 на 48 пикселей. Датасет содержит неравномерное количество изображений разных классов.

![plots](https://user-images.githubusercontent.com/97290990/233084609-4ea63ff4-14c3-42ab-85e8-dbb1cf569472.png)

В связи с этим один из классов было решенено удалить и обучить сверточную нейронную сеть для классификации остальных. Была получена диаграмма доказывающая, что решение данной задачи реально

![image](https://user-images.githubusercontent.com/97290990/233087666-3e865432-2af5-48a3-96fc-4c72378bd1ba.png)

![image]([https://user-images.githubusercontent.com/97290990/233087666-3e865432-2af5-48a3-96fc-4c72378bd1ba.png](https://github.com/Alexey9991/face-emotion-recognition/blob/1d8c47f4818823c89f8701cb4080277926f90b1e/%D0%BE%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D1%8B.png))

![image](https://user-images.githubusercontent.com/97290990/233087666-3e865432-2af5-48a3-96fc-4c72378bd1ba.png)
