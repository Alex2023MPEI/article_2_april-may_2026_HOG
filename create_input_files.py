import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from typing import List, Tuple

def extract_hog_features(image_path: str) -> np.ndarray:
    """
    Извлекает HOG признаки из одного изображения.
    Изображение преобразуется в grayscale и ресайзится к 400x400 (уже должно быть).
    Параметры HOG: orientations=9, pixels_per_cell=(50,50), cells_per_block=(2,2).
    Возвращает вектор признаков (n_features,) типа float64.
    """
    with Image.open(image_path) as img:
        # Конвертируем в grayscale
        img_gray = img.convert(mode='L')
        # Преобразуем в numpy array
        image = np.array(object=img_gray, dtype=np.float64) / 255.0  # нормализация [0,1]
    
    # Извлечение HOG (width=400, height=400)
    features = hog(
        image=image,
        orientations=9,
        pixels_per_cell=(50, 50),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=False,
        feature_vector=True
    );#С такими параметрами n_features=(400/50-2+1)*(400/50-2+1)*2*2*9=1_764
    return features.astype(np.float64)


def build_dataset(
    images_dir: str = "car_bike_dataset_aug",
    output_dir: str = ".",
    split_ratio: float = 0.8
) -> None:
    """
    Создаёт файлы opened_ids.txt, opened_data.npy, opened_target.npy,
    closed_ids.txt, closed_data.npy, closed_target.npy.

    Аргументы:
        images_dir: папка с аугментированными изображениями (bike_*_aug*.jpg, car_*_aug*.jpg)
        output_dir: папка для сохранения выходных файлов
        split_ratio: доля данных в opened (0.8 = 80% в opened, 20% в closed)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем, какие номера исходных изображений (1..2000) попадают в opened
    # Используем остаток от деления на 5: остаток 0 -> closed, иначе opened
    # Так как 1/5 = 0.2, closed = 20%
    total_original = 2000
    closed_numbers = set()
    for num in range(1, total_original + 1):
        if num % 5 == 0:   # 400 номеров
            closed_numbers.add(num)
    
    print(f"Closed numbers count: {len(closed_numbers)} (expected 400)")
    print(f"Opened numbers count: {total_original - len(closed_numbers)} (expected 1600)")
    
    # Списки для opened и closed
    opened_ids: List[str] = []
    opened_data: List[np.ndarray] = []
    opened_targets: List[float] = []
    
    closed_ids: List[str] = []
    closed_data: List[np.ndarray] = []
    closed_targets: List[float] = []
    
    # Аугментации: от 0 до 6 (всего 7)
    aug_indices = range(7)  # 0..6
    
    # Пройдём по всем номерам 1..2000
    for num in range(1, total_original + 1):
        # Определяем, куда попадут все файлы этого номера
        is_closed = num in closed_numbers
        target_list = closed_targets if is_closed else opened_targets
        ids_list = closed_ids if is_closed else opened_ids
        data_list = closed_data if is_closed else opened_data
        
        # Для каждого типа: bike и car
        for prefix, target_val in [("bike", 0.0), ("car", 1.0)]:
            for aug in aug_indices:
                filename = f"{prefix}_{num}_aug{aug}.jpg"
                filepath = os.path.join(images_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: {filepath} not found, skipping")
                    continue
                
                # Извлечение HOG признаков
                try:
                    features = extract_hog_features(filepath)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
                
                # Добавляем в соответствующие списки
                ids_list.append(filename)
                data_list.append(features)
                target_list.append(target_val)
        if (num%20)==0:print(f'{num}/{total_original} файлов для каждого типа (bike и car) обработано');
    # Преобразуем списки в numpy массивы
    def save_split(ids_list, data_list, targets_list, name_prefix):
        # ids.txt
        with open(os.path.join(output_dir, f"{name_prefix}_ids.txt"), 'w') as f:
            for id_ in ids_list:
                f.write(id_ + '\n')
        # data.npy
        if data_list:
            data_arr = np.vstack(data_list)  # (n_samples, n_features)
            np.save(os.path.join(output_dir, f"{name_prefix}_data.npy"), data_arr)
        else:
            np.save(os.path.join(output_dir, f"{name_prefix}_data.npy"), np.empty((0, 0)))
        # target.npy
        targets_arr = np.array(targets_list, dtype=np.float64)
        np.save(os.path.join(output_dir, f"{name_prefix}_target.npy"), targets_arr)
    
    save_split(opened_ids, opened_data, opened_targets, "opened")
    save_split(closed_ids, closed_data, closed_targets, "closed")
    
    print("\nСохранены следующие файлы:")
    for name in ["opened_ids.txt", "opened_data.npy", "opened_target.npy",
                 "closed_ids.txt", "closed_data.npy", "closed_target.npy"]:
        print(f"  {os.path.join(output_dir, name)}")
    print(f"\nСтатистика:")
    print(f"  opened: {len(opened_ids)} образцов (ids), {len(opened_data)} векторов признаков длины {len(opened_data[0])}, {len(opened_targets)} таргетов")
    print(f"  closed: {len(closed_ids)} образцов (ids), {len(closed_data)} векторов признаков длины {len(opened_data[0])}, {len(closed_targets)} таргетов")
    print(f"  Всего: {len(opened_ids) + len(closed_ids)} образцов (ожидается 28000)")


if __name__ == "__main__":
    build_dataset(
        images_dir="car_bike_dataset_aug",
        output_dir=".",
        split_ratio=0.8
    )