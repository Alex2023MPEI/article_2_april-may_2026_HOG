import io,os;
import numpy as np;
from pathlib import Path;
from PIL import Image, ImageFilter, ImageEnhance;
from typing import Dict, List, Tuple;

def get_image_dimensions() -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Считывает все изображения из папок Car_Bike_Dataset_orig/Bike и Car_Bike_Dataset_orig/Car,
    определяет их размеры (ширина, высота) и возвращает два словаря:
    - ключ: имя файла
    - значение: список [width, height]

    Предполагается, что папки Car_Bike_Dataset_orig находятся в той же директории, что и скрипт.
    """
    # Определяем директорию, где находится текущий скрипт
    script_dir:str=os.path.dirname(os.path.abspath(__file__));
    base_dir:str=os.path.join(script_dir, "Car_Bike_Dataset_orig");
    bike_dir:str=os.path.join(base_dir, "Bike");
    car_dir:str=os.path.join(base_dir, "Car");
    # Расширения файлов, которые считаем изображениями
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'};
    bike_dict: Dict[str, List[int]] = {};
    car_dict: Dict[str, List[int]] = {};
    # Функция для обработки одной папки
    def process_folder(folder_path: str) -> Dict[str, List[int]]:
        result = {};
        if not os.path.exists(folder_path):
            print(f"Предупреждение: папка {folder_path} не найдена");
            return result;

        for filename in os.listdir(folder_path):
            # Проверяем расширение файла
            ext = os.path.splitext(filename)[1].lower();
            if ext not in image_extensions:
                continue;
            filepath = os.path.join(folder_path, filename);
            try:
                with Image.open(filepath) as img:
                    width, height = img.size;
                    result[filename] = [width, height];
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}");
        return result;
    # Обрабатываем папки
    bike_dict = process_folder(bike_dir);
    car_dict = process_folder(car_dir);
    # Проверяем количество (необязательно, но полезно)
    if len(bike_dict) != 2000:
        print(f"Внимание: в папке Bike найдено {len(bike_dict)} изображений (ожидалось 2000)");
    if len(car_dict) != 2000:
        print(f"Внимание: в папке Car найдено {len(car_dict)} изображений (ожидалось 2000)");
    return bike_dict, car_dict;

def make_same_size_dataset(new_side: int) -> None:
    """
    Обрабатывает все изображения из папок Car_Bike_Dataset_orig/Bike и Car_Bike_Dataset_orig/Car:
    - приводит большую сторону каждого изображения к new_side (с сохранением пропорций),
    - дополняет меньшую сторону белым фоном до квадрата new_side x new_side,
    - переименовывает в car_{num}.jpg и bike_{num}.jpg (num от 1 до 2000),
    - сохраняет все изображения в одну папку car_bike_dataset_same_size.
    """
    # Определяем пути
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "Car_Bike_Dataset_orig")
    car_dir = os.path.join(base_dir, "Car")
    bike_dir = os.path.join(base_dir, "Bike")
    out_dir = os.path.join(script_dir, "car_bike_dataset_same_size")

    # Создаём выходную папку, если её нет
    os.makedirs(out_dir, exist_ok=True)

    # Расширения изображений, которые будем обрабатывать
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    def process_folder(folder: str, prefix: str) -> None:
        """Обрабатывает одну папку, сохраняя изображения с префиксом prefix_<номер>.jpg"""
        # Получаем список файлов, отфильтровывая по расширениям
        files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]
        files.sort()  # чтобы нумерация была детерминированной
        num_done:int=0;
        num_total:int=len(files);
        for idx, filename in enumerate(files, start=1):
            filepath = os.path.join(folder, filename)
            try:
                with Image.open(filepath) as img:
                    # Переводим в RGB (на случай RGBA или других режимов)
                    img = img.convert('RGB')
                    w, h = img.size
                    # 1. Масштабируем так, чтобы большая сторона стала new_side
                    if w >= h:
                        new_w = new_side
                        new_h = int(round(h * (new_side / w)))
                    else:
                        new_h = new_side
                        new_w = int(round(w * (new_side / h)))
                    # Изменяем размер с высоким качеством
                    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    # 2. Создаём белое полотно и вставляем изображение по центру
                    canvas = Image.new('RGB', (new_side, new_side), (255, 255, 255))
                    x_offset = (new_side - new_w) // 2
                    y_offset = (new_side - new_h) // 2
                    canvas.paste(img_resized, (x_offset, y_offset))
                    # 3. Сохраняем с новым именем
                    out_name = f"{prefix}_{idx}.jpg"
                    out_path = os.path.join(out_dir, out_name)
                    canvas.save(out_path, 'JPEG', quality=95)
                    print(f"Saved: {out_name}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            num_done=num_done+1;
            print(f'Обработано {num_done}/{num_total} файлов в папке [{folder}]');
    # Обрабатываем обе папки (нумерация для каждой начинается с 1)
    print("Processing cars...")
    process_folder(car_dir, 'car')
    print("Processing bikes...")
    process_folder(bike_dir, 'bike')
    print(f"\nDone! All images saved to {out_dir}")
    print(f"Total images: 4000 (2000 cars + 2000 bikes)")

def generate_augmented_dataset(
    input_dir: str = "car_bike_dataset_same_size",
    output_dir: str = "car_bike_dataset_aug",
    image_size: int = 400,
    jpeg_quality: int = 35,
    brightness_range: tuple = (0.6, 1.4),
    noise_std: int = 25
) -> None:
    """
    Создаёт аугментированные версии всех изображений из input_dir.
    
    Для каждого файла (например, bike_5.jpg) создаются:
        bike_5_aug0.jpg   - оригинал
        bike_5_aug1.jpg   - белый шум
        bike_5_aug2.jpg   - размытие по Гауссу
        bike_5_aug3.jpg   - артефакты JPEG (сильное сжатие)
        bike_5_aug4.jpg   - смещение красного канала +20
        bike_5_aug5.jpg   - изменение яркости
        bike_5_aug6.jpg   - горизонтальное отражение
    
    Параметры:
        input_dir: папка с исходными изображениями (400x400)
        output_dir: папка для сохранения аугментированных изображений
        image_size: размер изображения (квадрат)
        jpeg_quality: качество для имитации артефактов (0-100, меньше = сильнее)
        brightness_range: (мин, макс) для множителя яркости
        noise_std: среднеквадратичное отклонение для гауссовского шума (в единицах пикселя 0-255)
    """
    # Создаём выходную папку
    os.makedirs(output_dir, exist_ok=True)
    # Получаем список всех изображений (предполагаем, что все .jpg)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    files.sort()  # детерминированный порядок
    # Функции аугментаций
    def aug_original(img: Image.Image) -> Image.Image:
        """Копия без изменений"""
        return img.copy()
    def aug_white_noise(img: Image.Image) -> Image.Image:
        """Добавление гауссовского шума"""
        np_img = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, noise_std, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    def aug_gaussian_blur(img: Image.Image) -> Image.Image:
        """Размытие по Гауссу (радиус 1.5)"""
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))
    def aug_jpeg_artifacts(img: Image.Image) -> Image.Image:
        """Имитация сильного сжатия JPEG"""
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    def aug_red_shift(img: Image.Image) -> Image.Image:
        """Увеличение красного канала на 20 (насыщение)"""
        np_img = np.array(img, dtype=np.int16)
        np_img[:, :, 0] = np.clip(np_img[:, :, 0] + 20, 0, 255)  # канал R
        return Image.fromarray(np_img.astype(np.uint8))
    def aug_brightness(img: Image.Image) -> Image.Image:
        """Случайное изменение яркости"""
        factor = np.random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    def aug_horizontal_flip(img: Image.Image) -> Image.Image:
        """Зеркальное отражение слева-направо"""
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # Список всех аугментаций (индекс = номер aug)
    augmentations: List = [
        ("original", aug_original),
        ("white_noise", aug_white_noise),
        ("gaussian_blur", aug_gaussian_blur),
        ("jpeg_artifacts", aug_jpeg_artifacts),
        ("red_shift", aug_red_shift),
        ("brightness", aug_brightness),
        ("horizontal_flip", aug_horizontal_flip)
    ]
    # Обрабатываем каждый файл
    total_files = len(files)
    for idx, filename in enumerate(files, 1):
        basename = os.path.splitext(filename)[0]  # например, "bike_5"
        img_path = os.path.join(input_dir, filename)   
        with Image.open(img_path) as img:
            img_rgb = img.convert('RGB')
            # Для каждой аугментации создаём и сохраняем файл
            for aug_idx, (aug_name, aug_func) in enumerate(augmentations):
                out_name = f"{basename}_aug{aug_idx}.jpg"
                out_path = os.path.join(output_dir, out_name)
                try:
                    augmented = aug_func(img_rgb)
                    # Убедимся, что размер остался 400x400 (некоторые аугментации могли его изменить)
                    if augmented.size != (image_size, image_size):
                        augmented = augmented.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    augmented.save(out_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"Ошибка при создании {out_name}: {e}")
        if idx % 100 == 0:
            print(f"Обработано {idx} / {total_files} файлов")
    print(f"\nГотово! Все изображения сохранены в {output_dir}")
    print(f"Всего файлов: {total_files * len(augmentations)}")
    return None;
def is_suitable_for_stats(dtype)->bool:
    """Функция проверяет, доступны ли для типа dtype операции сложения, умножения, деления и сравнение порядка"""
    res:bool=False;
    if dtype in [np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64]:res=True;
    if dtype in [np.float16,np.float32,np.float64]:res=True;
    return res;
def view_all_npy_files_info()->None:
    """Функция выводит информацию обо всех npy файлах в текущей папке (без подпапок)"""
    npy_files = sorted(Path('.').glob('*.npy'));#Находим все .npy файлы только в текущей директории (без подпапок)
    if not npy_files:
        print("В текущей папке не найдено файлов с расширением .npy")
        return
    for file_path in npy_files:
        # Размер файла в байтах
        file_size = file_path.stat().st_size
        try:
            # Загружаем массив
            arr:np.ndarray = np.load(file_path)
            # Метаданные массива
            shape = arr.shape
            dtype = arr.dtype
            ndim=arr.ndim
            num_elements = arr.size  # эквивалентно np.prod(shape)
            print(f"Файл: {file_path.name}");
            print(f"  Размер файла: {file_size} байт");
            print(f"  Shape: {shape}");
            print(f'  Ndim: {ndim}');
            print(f"  Dtype: {dtype}");
            print(f"  Количество элементов: {num_elements}");
            if (num_elements>0)and(is_suitable_for_stats(arr.dtype)):
                print(f'  Минимальный элемент: {np.min(arr)}');
                print(f'  Максимальный элемент: {np.max(arr)}');
                print(f'  Среднее арифметическое элементов: {np.mean(arr)}');
                pass;
        except Exception as e:
            print(f"Файл: {file_path.name}")
            print(f"  Размер файла: {file_size} байт")
            print(f"  ⚠️ Ошибка при чтении: {e}")
        print("-" * 40);
    return None;

if __name__ == "__main__":
    bike_sizes_dict, car_sizes_dict = get_image_dimensions()
    print(f"Количество изображений Bike: {len(bike_sizes_dict)}")
    print(f"Количество изображений Car: {len(car_sizes_dict)}")
    # Вывод первых нескольких записей
    print(f'filename:size[width, height] (for bikes):');
    for name, size in list(bike_sizes_dict.items())[:3]:print(f"{name}:{size}");
    print(f'filename:size[width, height] (for cars):');
    for name, size in list(car_sizes_dict.items())[:3]:print(f"{name}:{size}");
    bike_widths:list[int]=[size[0] for size in bike_sizes_dict.values()];
    bike_heights:list[int]=[size[1] for size in bike_sizes_dict.values()];
    car_widths:list[int]=[size[0] for size in car_sizes_dict.values()];
    car_heights:list[int]=[size[1] for size in car_sizes_dict.values()];
    bike_aspect_ratios:list[float]=[bike_widths[i]/bike_heights[i] for i in range(len(bike_widths))];
    car_aspect_ratios:list[float]=[car_widths[i]/car_heights[i] for i in range(len(car_widths))];
    #print(f'bike_widths: {bike_widths}');
    #print(f'bike_heights: {bike_heights}');
    #print(f'car_widths: {car_widths}');
    #print(f'car_heights: {car_heights}');
    bike_width_min:int=min(bike_widths);
    bike_width_max:int=max(bike_widths);
    bike_width_mean:float=sum(bike_widths)/len(bike_widths);
    bike_height_min:int=min(bike_heights);
    bike_height_max:int=max(bike_heights);
    bike_height_mean:float=sum(bike_heights)/len(bike_heights);
    bike_aspect_ratio_min:int=min(bike_aspect_ratios);
    bike_aspect_ratio_max:int=max(bike_aspect_ratios);
    bike_aspect_ratio_mean:float=sum(bike_aspect_ratios)/len(bike_aspect_ratios);
    car_width_min:int=min(car_widths);
    car_width_max:int=max(car_widths);
    car_width_mean:float=sum(car_widths)/len(car_widths);
    car_height_min:int=min(car_heights);
    car_height_max:int=max(car_heights);
    car_height_mean:float=sum(car_heights)/len(car_heights);
    car_aspect_ratio_min:int=min(car_aspect_ratios);
    car_aspect_ratio_max:int=max(car_aspect_ratios);
    car_aspect_ratio_mean:float=sum(car_aspect_ratios)/len(car_aspect_ratios);
    print(f'Для bike: width от {bike_width_min} до {bike_width_max} [среднее {bike_width_mean}], height от {bike_height_min} до {bike_height_max} [среднее {bike_height_mean}], aspect_ratio (w/h) от {bike_aspect_ratio_min} до {bike_aspect_ratio_max} [среднее {bike_aspect_ratio_mean}]');
    print(f'Для car : width от {car_width_min} до {car_width_max} [среднее {car_width_mean}], height от {car_height_min} до {car_height_max} [среднее {car_height_mean}], aspect_ratio (w/h) от {car_aspect_ratio_min} до {car_aspect_ratio_max} [среднее {car_aspect_ratio_mean}]');
    #Приведение датасета к размеру 400x400 уже сделано
    #make_same_size_dataset(new_side=400);
    #Добавление аугментаций тоже уже сделано
    #generate_augmented_dataset(input_dir="car_bike_dataset_same_size",output_dir="car_bike_dataset_aug",image_size=400,jpeg_quality=35,brightness_range=(0.6,1.4),noise_std=25);
    view_all_npy_files_info();


