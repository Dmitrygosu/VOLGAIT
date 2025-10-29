#!/usr/bin/env python3

import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
import shutil
from pathlib import Path

class LicensePlateTrainer:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.dataset_dir = self.project_dir / "dataset"
        self.models_dir = self.project_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def download_datasets(self):
        print("Downloading license plate datasets...")
        
        datasets = [
            {
                "name": "RusCar",
                "url": "https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/raw/main/data.yaml",
                "description": "Russian car license plates dataset"
            }
        ]
        
        print("\nДля обучения модели используйте открытые датасеты:")
        print("1. RU-License-Plates-Dataset (GitHub)")
        print("2. Open Images Dataset v6 с аннотациями номеров")
        print("3. Генерация синтетических номеров")
        
        self.generate_synthetic_data()
        
    def generate_synthetic_data(self):
        print("\nГенерация синтетических данных...")
        
        synthetic_dir = self.dataset_dir / "synthetic"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        (synthetic_dir / "images").mkdir(exist_ok=True)
        (synthetic_dir / "labels").mkdir(exist_ok=True)
        
        russian_letters = "АВЕКМНОРСТУХ"
        latin_equiv = "ABEKMHOPCTYX"
        
        num_samples = 1000
        
        for i in range(num_samples):
            img = np.ones((200, 520, 3), dtype=np.uint8) * 255
            
            letter1 = np.random.choice(list(russian_letters))
            numbers = ''.join([str(np.random.randint(0, 10)) for _ in range(3)])
            letter2 = np.random.choice(list(russian_letters))
            letter3 = np.random.choice(list(russian_letters))
            region = str(np.random.randint(1, 200)).zfill(2)
            
            plate_text = f"{letter1}{numbers}{letter2}{letter3}{region}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, plate_text, (30, 130), font, 3, (0, 0, 0), 5)
            
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
            
            img_path = synthetic_dir / "images" / f"plate_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            label_path = synthetic_dir / "labels" / f"plate_{i:04d}.txt"
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.95 0.9\n")
        
        print(f"Создано {num_samples} синтетических изображений")
        
    def prepare_dataset(self):
        print("Подготовка датасета...")
        
        data_yaml = self.dataset_dir / "data.yaml"
        
        yaml_content = f"""
train: {self.dataset_dir}/synthetic/images
val: {self.dataset_dir}/synthetic/images

nc: 1
names: ['license_plate']
"""
        
        with open(data_yaml, 'w') as f:
            f.write(yaml_content.strip())
        
        print(f"Создан файл конфигурации: {data_yaml}")
        
    def train_yolo(self):
        print("\nНачало обучения YOLOv8...")
        
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data=str(self.dataset_dir / "data.yaml"),
            epochs=100,
            imgsz=640,
            batch=16,
            name='license_plate_detector',
            device='0' if self._has_cuda() else 'cpu',
            patience=20,
            save=True,
            pretrained=True,
            optimizer='Adam',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )
        
        print("\nОбучение завершено!")
        
        best_model_path = Path("runs/detect/license_plate_detector/weights/best.pt")
        if best_model_path.exists():
            model = YOLO(str(best_model_path))
            
            onnx_path = self.models_dir / "yolov8n.onnx"
            model.export(format='onnx', dynamic=False, simplify=True)
            
            exported_onnx = Path("runs/detect/license_plate_detector/weights/best.onnx")
            if exported_onnx.exists():
                shutil.copy(exported_onnx, onnx_path)
                print(f"\nМодель экспортирована: {onnx_path}")
        
    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def download_pretrained_model(self):
        print("\nСкачивание предобученной модели...")
        
        from ultralytics import YOLO
        
        model = YOLO('yolov8n.pt')
        
        onnx_path = self.models_dir / "yolov8n.onnx"
        model.export(format='onnx')
        
        exported = Path("yolov8n.onnx")
        if exported.exists():
            shutil.move(str(exported), str(onnx_path))
            print(f"Базовая модель сохранена: {onnx_path}")
    
    def run(self):
        print("=== Обучение модели распознавания номеров ===\n")
        
        self.download_datasets()
        self.prepare_dataset()
        
        choice = input("\nВыберите действие:\n1. Обучить модель с нуля\n2. Использовать предобученную модель\nВыбор (1/2): ")
        
        if choice == "1":
            self.train_yolo()
        else:
            self.download_pretrained_model()
        
        print("\n=== Подготовка завершена ===")
        print(f"Модель находится в: {self.models_dir}")

if __name__ == "__main__":
    trainer = LicensePlateTrainer()
    trainer.run()

