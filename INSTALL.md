# Инструкция по установке и запуску

## Windows

### Быстрая установка

Если есть Chocolatey:
```powershell
choco install golang opencv tesseract -y
```

### Ручная установка

1. Go - скачать с https://go.dev/dl/ и установить

2. OpenCV - скачать с https://opencv.org/releases/
   - Распаковать в C:\opencv
   - Добавить в переменные окружения:
     - OPENCV_DIR=C:\opencv\build
     - В PATH добавить: C:\opencv\build\x64\vc16\bin

3. Tesseract OCR - скачать с https://github.com/UB-Mannheim/tesseract/wiki
   - Установить
   - Добавить в PATH: C:\Program Files\Tesseract-OCR

### Установка зависимостей Go

```bash
go mod download
```

### Сборка и запуск

Вариант 1 - без компиляции:
```bash
go run main.go -video видео-участникам.mp4 -output results.csv
```

Вариант 2 - скомпилировать:
```bash
go build -o grz-recognition.exe main.go
.\grz-recognition.exe -video видео-участникам.mp4 -output results.csv
```

### Проверка результатов

```bash
python validate_results.py results.csv
```

## Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y golang-go libopencv-dev tesseract-ocr build-essential
go mod download
go build -o grz-recognition main.go
./grz-recognition -video video.mp4 -output results.csv
```

## Обучение модели (опционально)

```bash
pip install -r requirements.txt
python train_model.py
```

Выберите вариант 2 для загрузки готовой модели YOLOv8 (рекомендуется для быстрого старта).

## Возможные проблемы

### OpenCV не найден

Проверьте что OPENCV_DIR установлен:
```bash
echo $env:OPENCV_DIR
```

Должно показать C:\opencv\build

### Tesseract не найден

Проверьте что Tesseract в PATH:
```bash
tesseract --version
```

### Ошибка компиляции

```bash
go clean -cache
go mod tidy
go build -v main.go
```

## Минимальные требования

- CPU: Intel Core i5 7600
- RAM: 4GB
- Go 1.21+
- OpenCV 4.5+
- Tesseract OCR 4.0+
