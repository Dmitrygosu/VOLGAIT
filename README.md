# Распознавание ГРЗ

Программа для автоматического распознавания номерных знаков автомобилей с видео.

## Установка

Windows:
```bash
choco install golang opencv tesseract -y
go mod download
```

Linux:
```bash
sudo apt install golang libopencv-dev tesseract-ocr -y
go mod download
```

## Запуск

```bash
go run main.go -video видео-участникам.mp4 -output results.csv
```

Или скомпилировать и запустить:
```bash
go build -o grz-recognition.exe main.go
.\grz-recognition.exe -video видео-участникам.mp4 -output results.csv
```

## Проверка:

```bash
python validate_results.py results.csv
```

## Обучение модели

```bash
pip install -r requirements.txt
python train_model.py
```

Выберите вариант 2 для загрузки готовой модели YOLOv8.

## Файлы

- main.go - основной код
- train_model.py - обучение модели  
- validate_results.py - проверка результатов
- EXPLANATION.md - пояснительная записка
- INSTALL.md - инструкция по установке

## Требования

- Go 1.21+
- OpenCV 4.5+
- Tesseract OCR 4.0+
- CPU: Intel Core i5 или лучше
- RAM: 4GB минимум
