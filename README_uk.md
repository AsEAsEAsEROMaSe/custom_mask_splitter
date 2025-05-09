# Розділювач Масок для ComfyUI

[Read in English | Читати англійською](README.md)

Набір нод для ComfyUI, що забезпечують розширені операції маніпуляції масками, спеціально розроблені для розділення та обробки масок, що перетинаються.

## Особливості

- **Розділювач Масок (Mask Splitter)**: Розділяє дві маски, що перетинаються, розділяючи зону перетину на основі відстані від кожного пікселя до центру маси кожної маски.
- **Виштовхування Бульбашок до Зон (Push Bubbles To Zones)**: Розділяє маски, що перетинаються, імітуючи виштовхування "бульбашок" до відповідних зон.
- **Гравітаційний Розділювач Масок (Gravity Mask Splitter)**: Використовує гравітаційне силове поле для розділення масок, що перекриваються, з налаштовуваними параметрами сили.
- **Розділювач Масок Водорозділом (Watershed Mask Splitter)**: Реалізує алгоритм водорозділу для розділення масок, що перетинаються, корисний для складних форм.
- **Вставка Зображення в Маску (Image Mask Inserter)**: Вставляє зображення в білу область маски, масштабуючи його для точного вписування.

## Встановлення

1. Клонуйте цей репозиторій у директорію custom_nodes вашого ComfyUI:
```
cd ComfyUI/custom_nodes/
git clone https://github.com/yourusername/custom_mask_splitter.git
```

2. Перезапустіть ComfyUI або перезавантажте веб-інтерфейс.

## Використання

Після встановлення ноди з'являться в категорії "image/masking" у браузері нод ComfyUI.

### Розділювач Масок
Приймає дві маски як вхідні дані і розділяє їх область перетину на основі відстані до центру маси кожної маски.

### Виштовхування Бульбашок до Зон
Розділяє маски, що перетинаються, імітуючи бульбашки, які виштовхуються до своїх відповідних зон, з налаштовуваною кількістю ітерацій та розміром фільтра.

### Гравітаційний Розділювач Масок
Забезпечує фізично-заснований підхід до розділення масок з наступними параметрами:
- **сила гравітації (gravity_strength)**: Сила притягання до відповідних пікселів маски
- **сила центру (center_strength)**: Сила притягання до центру маси
- **ітерації (iterations)**: Кількість ітерацій алгоритму
- **захист ядра (protect_core)**: Чи захищати центральну частину маски
- **радіус ядра (core_radius)**: Радіус захищеної центральної області

### Розділювач Масок Водорозділом
Використовує алгоритм водорозділу для розділення масок, що перекриваються, з параметром розміру фільтра для видалення ізольованих областей.

### Вставка Зображення в Маску
Вставляє зображення в білу область маски з можливістю вибору різних методів інтерполяції.

## Приклади

[Додайте приклади зображень/робочих процесів тут]

## Ліцензія

Цей проект поширюється під ліцензією GNU General Public License v3.0 - дивіться файл [LICENSE](LICENSE) для деталей.

Це розширення використовує ту ж ліцензію, що й ComfyUI. Воно використовує різні бібліотеки (PyTorch, NumPy, OpenCV, PIL), які сумісні з цією ліцензією.

## Подяки

[Додайте подяки, якщо потрібно] 