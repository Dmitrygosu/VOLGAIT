#!/usr/bin/env python3

import csv
import sys
from datetime import timedelta

def parse_time(time_str):
    parts = time_str.split(':')
    minutes = int(parts[0])
    secs_parts = parts[1].split('.')
    seconds = int(secs_parts[0])
    centiseconds = int(secs_parts[1]) if len(secs_parts) > 1 else 0
    
    total_seconds = minutes * 60 + seconds + centiseconds / 100.0
    return total_seconds

def load_ground_truth(filepath):
    detections = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            time_detect = parse_time(row['time_detect'])
            plate = row['plate_num'].replace('#', '')
            detections.append({
                'time_start': parse_time(row['time_start']),
                'time_detect': time_detect,
                'time_end': parse_time(row['time_end']),
                'plate': plate,
                'full_plate': row['plate_num']
            })
    return detections

def load_results(filepath):
    detections = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            detections.append({
                'time': parse_time(row['time']),
                'plate': row['plate_num']
            })
    return detections

def normalize_plate(plate):
    return ''.join(c for c in plate.upper() if c.isalnum())

def compare_plates(plate1, plate2):
    p1 = normalize_plate(plate1)
    p2 = normalize_plate(plate2)
    
    if p1 == p2:
        return 1.0
    
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    
    matches = sum(c1 == c2 for c1, c2 in zip(p1, p2))
    return matches / max(len(p1), len(p2))

def validate_results(ground_truth_file, results_file):
    print("=== Валидация результатов распознавания ГРЗ ===\n")
    
    print(f"Загрузка эталонных данных из {ground_truth_file}...")
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"  Загружено {len(ground_truth)} номеров\n")
    
    print(f"Загрузка результатов из {results_file}...")
    try:
        results = load_results(results_file)
        print(f"  Загружено {len(results)} номеров\n")
    except FileNotFoundError:
        print(f"  Ошибка: файл не найден\n")
        return
    
    print("=" * 80)
    print(f"{'№':<4} {'Эталон':<15} {'Время':<12} {'Результат':<15} {'Время':<12} {'Совп.':<8} {'Статус'}")
    print("=" * 80)
    
    matched = 0
    partial_matched = 0
    total = len(ground_truth)
    
    for i, gt in enumerate(ground_truth, 1):
        gt_time_window = (gt['time_start'], gt['time_end'])
        
        best_match = None
        best_similarity = 0.0
        best_result = None
        
        for result in results:
            if gt_time_window[0] <= result['time'] <= gt_time_window[1]:
                similarity = compare_plates(gt['plate'], result['plate'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_result = result
                    best_match = True
        
        status = ""
        if best_result:
            time_str = format_time(best_result['time'])
            plate_str = best_result['plate']
            
            if best_similarity >= 0.9:
                status = "✓ Полное"
                matched += 1
            elif best_similarity >= 0.7:
                status = "~ Частичное"
                partial_matched += 1
            else:
                status = "✗ Ошибка"
        else:
            time_str = "---"
            plate_str = "---"
            status = "✗ Не найден"
        
        gt_time_str = format_time(gt['time_detect'])
        similarity_str = f"{best_similarity*100:.0f}%"
        
        print(f"{i:<4} {gt['full_plate']:<15} {gt_time_str:<12} {plate_str:<15} {time_str:<12} {similarity_str:<8} {status}")
    
    print("=" * 80)
    print()
    
    print("=== Статистика ===\n")
    print(f"Всего номеров в эталоне:     {total}")
    print(f"Полных совпадений:           {matched} ({matched/total*100:.1f}%)")
    print(f"Частичных совпадений:        {partial_matched} ({partial_matched/total*100:.1f}%)")
    print(f"Не распознано:               {total - matched - partial_matched} ({(total-matched-partial_matched)/total*100:.1f}%)")
    print()
    
    accuracy = (matched + partial_matched * 0.5) / total
    print(f"Точность (weighted):         {accuracy*100:.1f}%")
    print()
    
    if len(results) > 0:
        false_positives = 0
        for result in results:
            found = False
            for gt in ground_truth:
                if gt['time_start'] <= result['time'] <= gt['time_end']:
                    similarity = compare_plates(gt['plate'], result['plate'])
                    if similarity >= 0.7:
                        found = True
                        break
            if not found:
                false_positives += 1
        
        precision = (matched + partial_matched) / len(results) if len(results) > 0 else 0
        print(f"Ложных срабатываний:         {false_positives}")
        print(f"Precision:                   {precision*100:.1f}%")
    
    print()
    
    if accuracy >= 0.8:
        print("✓ ОТЛИЧНО! Система работает хорошо")
    elif accuracy >= 0.6:
        print("~ УДОВЛЕТВОРИТЕЛЬНО. Требуется улучшение")
    else:
        print("✗ НЕУДОВЛЕТВОРИТЕЛЬНО. Требуется значительное улучшение")

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{minutes:02d}:{secs:02d}.{centisecs:02d}"

if __name__ == "__main__":
    ground_truth_file = "test_ground_truth.csv"
    results_file = "results.csv"
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    validate_results(ground_truth_file, results_file)

