import numpy as np
import torch
import cv2
from PIL import Image
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MaskSplitter:
    CATEGORY = "image/masking"
    
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("result_mask1", "result_mask2")
    FUNCTION = "split_masks"
    
    def split_masks(self, mask1, mask2):
        """
        Універсальний розділювач масок, який ділить зону перетину на два сектори
        незалежно від форми та орієнтації перетину.
        
        Параметри:
        mask1, mask2 - вхідні бінарні маски (тензори PyTorch)
        
        Повертає:
        result_mask1, result_mask2 - результуючі маски після розділення
        """
        # Перевірка та обробка розмірностей масок
        l_masks1 = []
        l_masks2 = []
        
        if mask1.dim() == 2:
            mask1 = torch.unsqueeze(mask1, 0)
        if mask2.dim() == 2:
            mask2 = torch.unsqueeze(mask2, 0)
            
        # Створюємо список результуючих масок
        result_masks1 = []
        result_masks2 = []
        
        # Обробляємо кожну маску в батчі
        for i in range(len(mask1)):
            m1 = mask1[i] if i < len(mask1) else mask1[-1]
            m2 = mask2[i] if i < len(mask2) else mask2[-1]
            
            # Конвертуємо тензори в numpy масиви для обробки
            mask1_np = m1.cpu().numpy()
            mask2_np = m2.cpu().numpy()
            
            # Конвертуємо маски до формату uint8 для обробки в OpenCV
            mask1_np = (mask1_np * 255).astype(np.uint8)
            mask2_np = (mask2_np * 255).astype(np.uint8)
            
            # Створюємо копії для результатів
            result_mask1 = mask1_np.copy()
            result_mask2 = mask2_np.copy()
            
            # Знаходимо перетин
            intersection = np.logical_and(mask1_np > 0, mask2_np > 0).astype(np.uint8) * 255
            
            # Якщо немає перетину, повертаємо оригінальні маски
            if np.sum(intersection) == 0:
                result_mask1_tensor = torch.from_numpy(mask1_np / 255.0).float().unsqueeze(0)
                result_mask2_tensor = torch.from_numpy(mask2_np / 255.0).float().unsqueeze(0)
                result_masks1.append(result_mask1_tensor)
                result_masks2.append(result_mask2_tensor)
                continue
            
            # Знаходимо центри мас масок
            try:
                m1 = cv2.moments(mask1_np)
                m2 = cv2.moments(mask2_np)
                
                # Якщо моменти не визначені, використовуємо простий підхід
                if m1["m00"] == 0 or m2["m00"] == 0:
                    raise ValueError("Zero moment detected")
                
                cx1, cy1 = int(m1["m10"] / m1["m00"]), int(m1["m01"] / m1["m00"])
                cx2, cy2 = int(m2["m10"] / m2["m00"]), int(m2["m01"] / m2["m00"])
            except Exception as e:
                print(f"Error calculating moments: {e}")
                # Резервний спосіб - геометричний центр
                h, w = mask1_np.shape
                cx1, cy1 = w // 4, h // 2
                cx2, cy2 = 3 * w // 4, h // 2
            
            # Створюємо маску розділення
            divider_mask = np.zeros_like(mask1_np)
            
            # Координати пікселів перетину
            y_coords, x_coords = np.where(intersection > 0)
            
            # Обчислюємо відстані для кожного пікселя перетину
            for j in range(len(y_coords)):
                y, x = y_coords[j], x_coords[j]
                
                # Відстані до центрів масок
                dist1 = np.sqrt((x - cx1)**2 + (y - cy1)**2)
                dist2 = np.sqrt((x - cx2)**2 + (y - cy2)**2)
                
                # Якщо піксель ближче до першого центру, позначаємо його
                if dist1 <= dist2:
                    divider_mask[y, x] = 1
            
            # Розділяємо маски
            result_mask1[np.logical_and(intersection > 0, divider_mask == 0)] = 0
            result_mask2[np.logical_and(intersection > 0, divider_mask == 1)] = 0
            
            # Конвертуємо назад у тензори PyTorch і додаємо розмірність батчу
            result_mask1_tensor = torch.from_numpy(result_mask1 / 255.0).float().unsqueeze(0)
            result_mask2_tensor = torch.from_numpy(result_mask2 / 255.0).float().unsqueeze(0)
            
            result_masks1.append(result_mask1_tensor)
            result_masks2.append(result_mask2_tensor)
        
        # Об'єднуємо результати і повертаємо їх як кортеж
        final_mask1 = torch.cat(result_masks1, dim=0)
        final_mask2 = torch.cat(result_masks2, dim=0)
        
        return (final_mask1, final_mask2)



class PushBubblesToZones:
    CATEGORY = "image/masking"
    
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "filter_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2})
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("result_mask1", "result_mask2")
    FUNCTION = "push_bubbles"
    
    def push_bubbles(self, mask1, mask2, iterations=10, filter_size=3):
        """
        Розділяє перетин двох масок, ніби 'бульбашки' переміщуються в свої зони.
        
        Параметри:
        mask1, mask2 - вхідні бінарні маски (тензори PyTorch)
        iterations - максимальна кількість ітерацій
        filter_size - розмір фільтра для видалення ізольованих зон
        
        Повертає:
        result_mask1, result_mask2 - результуючі маски після розділення
        """
        # Перевірка та обробка розмірностей масок
        if mask1.dim() == 2:
            mask1 = torch.unsqueeze(mask1, 0)
        if mask2.dim() == 2:
            mask2 = torch.unsqueeze(mask2, 0)
            
        # Створюємо список результуючих масок
        result_masks1 = []
        result_masks2 = []
        
        # Обробляємо кожну маску в батчі
        for i in range(len(mask1)):
            m1 = mask1[i] if i < len(mask1) else mask1[-1]
            m2 = mask2[i] if i < len(mask2) else mask2[-1]
            
            # Конвертуємо тензори в numpy масиви для обробки
            mask1_np = m1.cpu().numpy()
            mask2_np = m2.cpu().numpy()
            
            # Конвертуємо маски до формату uint8 для обробки в OpenCV
            mask1_np = (mask1_np * 255).astype(np.uint8)
            mask2_np = (mask2_np * 255).astype(np.uint8)
            
            # Створюємо копії для результатів
            result_mask1 = mask1_np.copy()
            result_mask2 = mask2_np.copy()
            
            # Знаходимо перетин
            intersection = np.logical_and(mask1_np > 0, mask2_np > 0).astype(np.uint8) * 255
            
            # Якщо немає перетину, повертаємо оригінальні маски
            if np.sum(intersection) == 0:
                result_mask1_tensor = torch.from_numpy(mask1_np / 255.0).float().unsqueeze(0)
                result_mask2_tensor = torch.from_numpy(mask2_np / 255.0).float().unsqueeze(0)
                result_masks1.append(result_mask1_tensor)
                result_masks2.append(result_mask2_tensor)
                continue
            
            # Знаходимо координати пікселів у зоні перетину
            y, x = np.where(intersection > 0)
            
            # Поки є пікселі в зоні перетину і не перевищено кількість ітерацій
            iter_count = 0
            while len(x) > 0 and iter_count < iterations:
                new_intersection = np.zeros_like(intersection)
                
                for j in range(len(x)):
                    xi, yi = x[j], y[j]
                    
                    # Визначаємо, куди виштовхувати
                    if mask1_np[yi, xi] > 0:
                        # Шукаємо найближчу точку в зоні mask1
                        moved = False
                        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            nx, ny = xi + dx, yi + dy
                            if 0 <= nx < result_mask1.shape[1] and 0 <= ny < result_mask1.shape[0]:
                                if result_mask1[ny, nx] > 0 and result_mask2[ny, nx] == 0:
                                    # Знайдено безпечну зону в mask1
                                    result_mask1[yi, xi] = 255
                                    moved = True
                                    break
                        
                        if not moved:
                            # Якщо не знайдено безпечну зону, просто очищаємо
                            result_mask1[yi, xi] = 0
                    
                    if mask2_np[yi, xi] > 0:
                        # Шукаємо найближчу точку в зоні mask2
                        moved = False
                        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            nx, ny = xi + dx, yi + dy
                            if 0 <= nx < result_mask2.shape[1] and 0 <= ny < result_mask2.shape[0]:
                                if result_mask2[ny, nx] > 0 and result_mask1[ny, nx] == 0:
                                    # Знайдено безпечну зону в mask2
                                    result_mask2[yi, xi] = 255
                                    moved = True
                                    break
                        
                        if not moved:
                            # Якщо не знайдено безпечну зону, просто очищаємо
                            result_mask2[yi, xi] = 0
                
                # Оновлюємо перетин
                new_intersection = np.logical_and(result_mask1 > 0, result_mask2 > 0).astype(np.uint8) * 255
                
                # Якщо перетин не змінився, виходимо з циклу
                if np.array_equal(new_intersection, intersection):
                    break
                
                intersection = new_intersection
                y, x = np.where(intersection > 0)
                iter_count += 1
            
            # Фільтрація ізольованих зон, якщо filter_size > 1
            if filter_size > 1:
                kernel = np.ones((filter_size, filter_size), np.uint8)
                result_mask1 = cv2.morphologyEx(result_mask1, cv2.MORPH_OPEN, kernel)
                result_mask1 = cv2.morphologyEx(result_mask1, cv2.MORPH_CLOSE, kernel)
                result_mask2 = cv2.morphologyEx(result_mask2, cv2.MORPH_OPEN, kernel)
                result_mask2 = cv2.morphologyEx(result_mask2, cv2.MORPH_CLOSE, kernel)
            
            # Конвертуємо назад у тензори PyTorch і додаємо розмірність батчу
            result_mask1_tensor = torch.from_numpy(result_mask1 / 255.0).float().unsqueeze(0)
            result_mask2_tensor = torch.from_numpy(result_mask2 / 255.0).float().unsqueeze(0)
            
            result_masks1.append(result_mask1_tensor)
            result_masks2.append(result_mask2_tensor)
        
        # Об'єднуємо результати і повертаємо їх як кортеж
        final_mask1 = torch.cat(result_masks1, dim=0)
        final_mask2 = torch.cat(result_masks2, dim=0)
        
        return (final_mask1, final_mask2)


class GravityMaskSplitter:
    CATEGORY = "image/masking"
    
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "gravity_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "center_strength": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "protect_core": ("BOOLEAN", {"default": True}),
                "core_radius": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05})
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("result_mask1", "result_mask2")
    FUNCTION = "gravity_split"
    
    def gravity_split(self, mask1, mask2, gravity_strength=1.0, center_strength=2.0, 
                      iterations=10, protect_core=True, core_radius=0.2):
        """
        Розділяє маски за принципом гравітаційного тяжіння.
        
        Параметри:
        mask1, mask2 - вхідні бінарні маски (тензори PyTorch)
        gravity_strength - сила тяжіння до своїх пікселів
        center_strength - сила тяжіння до центру маси
        iterations - кількість ітерацій алгоритму
        protect_core - чи захищати центральну частину маски
        core_radius - радіус захищеної центральної частини (відносна частина від розміру об'єкта)
        
        Повертає:
        result_mask1, result_mask2 - результуючі маски після розділення
        """
        # Перевірка та обробка розмірностей масок
        if mask1.dim() == 2:
            mask1 = torch.unsqueeze(mask1, 0)
        if mask2.dim() == 2:
            mask2 = torch.unsqueeze(mask2, 0)
            
        # Створюємо список результуючих масок
        result_masks1 = []
        result_masks2 = []
        
        # Обробляємо кожну маску в батчі
        for i in range(len(mask1)):
            m1 = mask1[i] if i < len(mask1) else mask1[-1]
            m2 = mask2[i] if i < len(mask2) else mask2[-1]
            
            # Конвертуємо тензори в numpy масиви для обробки
            mask1_np = m1.cpu().numpy()
            mask2_np = m2.cpu().numpy()
            
            # Конвертуємо маски до формату uint8 для обробки в OpenCV
            mask1_np = (mask1_np * 255).astype(np.uint8)
            mask2_np = (mask2_np * 255).astype(np.uint8)
            
            # Створюємо копії для результатів
            result_mask1 = mask1_np.copy()
            result_mask2 = mask2_np.copy()
            
            # Знаходимо перетин
            intersection = np.logical_and(mask1_np > 0, mask2_np > 0).astype(np.uint8) * 255
            
            # Якщо немає перетину, повертаємо оригінальні маски
            if np.sum(intersection) == 0:
                result_mask1_tensor = torch.from_numpy(mask1_np / 255.0).float().unsqueeze(0)
                result_mask2_tensor = torch.from_numpy(mask2_np / 255.0).float().unsqueeze(0)
                result_masks1.append(result_mask1_tensor)
                result_masks2.append(result_mask2_tensor)
                continue
            
            # Знаходимо центри мас масок
            try:
                m1_moments = cv2.moments(mask1_np)
                m2_moments = cv2.moments(mask2_np)
                
                # Якщо моменти не визначені, використовуємо простий підхід
                if m1_moments["m00"] == 0 or m2_moments["m00"] == 0:
                    raise ValueError("Zero moment detected")
                
                cx1, cy1 = int(m1_moments["m10"] / m1_moments["m00"]), int(m1_moments["m01"] / m1_moments["m00"])
                cx2, cy2 = int(m2_moments["m10"] / m2_moments["m00"]), int(m2_moments["m01"] / m2_moments["m00"])
            except Exception as e:
                print(f"Error calculating moments: {e}")
                # Резервний спосіб - геометричний центр
                h, w = mask1_np.shape
                cx1, cy1 = w // 4, h // 2
                cx2, cy2 = 3 * w // 4, h // 2
            
            # Обчислюємо захищені зони (ядра) масок, якщо потрібно
            core_mask1 = np.zeros_like(mask1_np)
            core_mask2 = np.zeros_like(mask2_np)
            
            if protect_core:
                # Знаходимо всі контури для визначення розміру об'єкта
                contours1, _ = cv2.findContours(mask1_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours2, _ = cv2.findContours(mask2_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Обчислюємо середній радіус для кожної маски
                if len(contours1) > 0:
                    max_contour1 = max(contours1, key=cv2.contourArea)
                    area1 = cv2.contourArea(max_contour1)
                    radius1 = int(np.sqrt(area1 / np.pi) * core_radius)
                    cv2.circle(core_mask1, (cx1, cy1), radius1, 255, -1)
                
                if len(contours2) > 0:
                    max_contour2 = max(contours2, key=cv2.contourArea)
                    area2 = cv2.contourArea(max_contour2)
                    radius2 = int(np.sqrt(area2 / np.pi) * core_radius)
                    cv2.circle(core_mask2, (cx2, cy2), radius2, 255, -1)
            
            # Створюємо маски потенціалів тяжіння для обох масок
            h, w = mask1_np.shape
            gravity_field1 = np.zeros((h, w), dtype=np.float32)
            gravity_field2 = np.zeros((h, w), dtype=np.float32)
            
            # Знаходимо всі пікселі в масках (без перетину)
            y1, x1 = np.where(np.logical_and(mask1_np > 0, mask2_np == 0))
            y2, x2 = np.where(np.logical_and(mask2_np > 0, mask1_np == 0))
            
            # Зона перетину для обробки
            intersection_coords = np.where(intersection > 0)
            y_intersect, x_intersect = intersection_coords
            
            # Для кожної ітерації
            for _ in range(iterations):
                # Оновлюємо gravity_field для кожної маски
                gravity_field1.fill(0)
                gravity_field2.fill(0)
                
                # Вплив всіх пікселів першої маски на силове поле
                for py, px in zip(y1, x1):
                    for iy, ix in zip(y_intersect, x_intersect):
                        dist = np.sqrt((px - ix)**2 + (py - iy)**2)
                        if dist > 0:
                            gravity_field1[iy, ix] += gravity_strength / (dist ** 2)
                
                # Вплив всіх пікселів другої маски на силове поле
                for py, px in zip(y2, x2):
                    for iy, ix in zip(y_intersect, x_intersect):
                        dist = np.sqrt((px - ix)**2 + (py - iy)**2)
                        if dist > 0:
                            gravity_field2[iy, ix] += gravity_strength / (dist ** 2)
                
                # Вплив центрів мас
                for iy, ix in zip(y_intersect, x_intersect):
                    dist1 = np.sqrt((cx1 - ix)**2 + (cy1 - iy)**2)
                    dist2 = np.sqrt((cx2 - ix)**2 + (cy2 - iy)**2)
                    
                    if dist1 > 0:
                        gravity_field1[iy, ix] += center_strength / (dist1 ** 2)
                    if dist2 > 0:
                        gravity_field2[iy, ix] += center_strength / (dist2 ** 2)
                
                # Розподіляємо пікселі перетину відповідно до полів тяжіння
                new_intersection = np.zeros_like(intersection)
                
                for iy, ix in zip(y_intersect, x_intersect):
                    # Перевіряємо, чи піксель знаходиться в захищеній зоні
                    if core_mask1[iy, ix] > 0:
                        result_mask1[iy, ix] = 255
                        result_mask2[iy, ix] = 0
                    elif core_mask2[iy, ix] > 0:
                        result_mask1[iy, ix] = 0
                        result_mask2[iy, ix] = 255
                    else:
                        # Порівнюємо сили тяжіння
                        if gravity_field1[iy, ix] > gravity_field2[iy, ix]:
                            result_mask1[iy, ix] = 255
                            result_mask2[iy, ix] = 0
                        else:
                            result_mask1[iy, ix] = 0
                            result_mask2[iy, ix] = 255
                
                # Оновлюємо перетин для наступної ітерації
                new_intersection = np.logical_and(result_mask1 > 0, result_mask2 > 0).astype(np.uint8) * 255
                
                # Якщо перетин зник, виходимо з циклу
                if np.sum(new_intersection) == 0:
                    break
                
                intersection = new_intersection
                y_intersect, x_intersect = np.where(intersection > 0)
            
            # Завершальна перевірка і усунення залишкових перетинів
            final_intersection = np.logical_and(result_mask1 > 0, result_mask2 > 0)
            if np.any(final_intersection):
                for iy, ix in zip(*np.where(final_intersection)):
                    if gravity_field1[iy, ix] > gravity_field2[iy, ix]:
                        result_mask2[iy, ix] = 0
                    else:
                        result_mask1[iy, ix] = 0
            
            # Конвертуємо назад у тензори PyTorch і додаємо розмірність батчу
            result_mask1_tensor = torch.from_numpy(result_mask1 / 255.0).float().unsqueeze(0)
            result_mask2_tensor = torch.from_numpy(result_mask2 / 255.0).float().unsqueeze(0)
            
            result_masks1.append(result_mask1_tensor)
            result_masks2.append(result_mask2_tensor)
        
        # Об'єднуємо результати і повертаємо їх як кортеж
        final_mask1 = torch.cat(result_masks1, dim=0)
        final_mask2 = torch.cat(result_masks2, dim=0)
        
        return (final_mask1, final_mask2)


class WatershedMaskSplitter:
    CATEGORY = "image/masking"
    
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "filter_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2})
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("result_mask1", "result_mask2")
    FUNCTION = "watershed_split"
    
    def watershed_split(self, mask1, mask2, filter_size=3):
        # Перевірка та обробка розмірностей масок
        if mask1.dim() == 2:
            mask1 = torch.unsqueeze(mask1, 0)
        if mask2.dim() == 2:
            mask2 = torch.unsqueeze(mask2, 0)
            
        result_masks1 = []
        result_masks2 = []
        
        for i in range(len(mask1)):
            m1 = mask1[i] if i < len(mask1) else mask1[-1]
            m2 = mask2[i] if i < len(mask2) else mask2[-1]
            
            # Конвертація в numpy
            mask1_np = (m1.cpu().numpy() * 255).astype(np.uint8)
            mask2_np = (m2.cpu().numpy() * 255).astype(np.uint8)
            
            # Знаходимо перетин
            intersection = np.logical_and(mask1_np > 0, mask2_np > 0).astype(np.uint8) * 255
            
            # Якщо немає перетину, повертаємо оригінальні маски
            if np.sum(intersection) == 0:
                result_masks1.append(m1.unsqueeze(0))
                result_masks2.append(m2.unsqueeze(0))
                continue
            
            # Створюємо маркери для водорозділу
            markers = np.zeros_like(mask1_np, dtype=np.int32)
            markers[mask1_np > 0] = 1  # Маркер для першої маски
            markers[mask2_np > 0] = 2  # Маркер для другої маски
            
            # Підготовка градієнтного зображення для водорозділу
            # Об'єднуємо маски для градієнта
            combined_mask = np.maximum(mask1_np, mask2_np)
            
            # Застосовуємо фільтр Собеля для виявлення градієнта
            sobelx = cv2.Sobel(combined_mask, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(combined_mask, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(sobelx**2 + sobely**2)
            
            # Нормалізуємо градієнт до діапазону [0, 255]
            gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Застосовуємо водорозділ
            watershed_result = cv2.watershed(cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), markers)
            
            # Створюємо результуючі маски
            result_mask1 = np.zeros_like(mask1_np)
            result_mask2 = np.zeros_like(mask2_np)
            
            result_mask1[watershed_result == 1] = 255
            result_mask2[watershed_result == 2] = 255
            
            # Фільтрація ізольованих зон
            if filter_size > 1:
                kernel = np.ones((filter_size, filter_size), np.uint8)
                result_mask1 = cv2.morphologyEx(result_mask1, cv2.MORPH_OPEN, kernel)
                result_mask1 = cv2.morphologyEx(result_mask1, cv2.MORPH_CLOSE, kernel)
                result_mask2 = cv2.morphologyEx(result_mask2, cv2.MORPH_OPEN, kernel)
                result_mask2 = cv2.morphologyEx(result_mask2, cv2.MORPH_CLOSE, kernel)
            
            # Конвертуємо назад у тензори
            result_mask1_tensor = torch.from_numpy(result_mask1 / 255.0).float().unsqueeze(0)
            result_mask2_tensor = torch.from_numpy(result_mask2 / 255.0).float().unsqueeze(0)
            
            result_masks1.append(result_mask1_tensor)
            result_masks2.append(result_mask2_tensor)
        
        final_mask1 = torch.cat(result_masks1, dim=0)
        final_mask2 = torch.cat(result_masks2, dim=0)
        
        return (final_mask1, final_mask2)

# Реєстрація ноди
NODE_CLASS_MAPPINGS = {
    "Mask Splitter": MaskSplitter,
    "Push Bubbles To Zones": PushBubblesToZones,
    "Gravity Mask Splitter": GravityMaskSplitter,
    "Watershed Mask Splitter": WatershedMaskSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mask Splitter": "Mask Splitter",
    "Push Bubbles To Zones": "Push Bubbles To Zones",
    "Gravity Mask Splitter": "Gravity Mask Splitter",
    "Watershed Mask Splitter": "Роздільник масок (Водорозділ)",
}
