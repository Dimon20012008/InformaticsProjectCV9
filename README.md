# Кривой Чертила

### Описание

"Кривой чертила" - игра, где надо обводить контур на экране за счёт рисования пальцем в воздухе.

### Установка

1. Скопировать все файлы проекта в проект в PyCharm. Для этого можно скачать .zip файл нажав "Code -> Download ZIP" и
   распаковать его в проекте.
2. Установить библиотеки:
   ```bash
   pip install -r requirements.txt
   ```

### Использование

1. Убедитесь, что камера на закрыта шторкой
2. Запустите файл main.py. Время запуска составляет около 20 секунд.
3. Появится окно с изображением веб-камеры. Убедитесь, что освещение подходящее (нужны белые тона) и программа
   отрисовывает кружок на указательном пальце правой руки:

![Wrong](resources/tests/test_blue_lightning.jpg)
![Wrong](resources/tests/test_dim_lightning.jpg)
**Примеры неправильного освещения**<br><br>
![Right](resources/tests/test_bright_lightning1.jpg)
![Right](resources/tests/test_bright_lightning2.jpg)
**Пример правильного освещения**

В кадре не должно быть две руки, иначе программа будет реагировать на обе:
4. Наведите палец на любую точку контура и нажмите на любую клавишу клавиатуры (например, на пробел), чтобы начать
   рисовать. Если при нажатии программа не обнаружила палец, то стартовой точкой будет первая точка, которую программа
   обнаружила.
5. Обведите контур. Чем ближе, тем выше будет результат. Близость можно понять по цвету контура - зелёный это близко,
   красный - очень далеко.

![Bad gameplay](resources/tests/test_contour_color_showcase_screenshot.jpg)

6. В конце покажется результат в процентах. Чем он ближе к 100%, тем лучше.
7. Теперь появился новый контур и игру можно начать сначала.
8. Для выхода из игры можно просто закрыть окно.

### Примеры с результатами

![Path1](resources/tests/test_gameplay1.jpg)
Пример 1
![Result1](resources/tests/test_result1.jpg)
Результат 1
![Path2](resources/tests/test_gameplay2.jpg)
Пример 2
![Result2](resources/tests/test_result2.jpg)
Результат 2
![Path3](resources/tests/test_gameplay3.jpg)
Пример 3
![Result3](resources/tests/test_result3.jpg)
Результат 3
![Path4](resources/tests/test_gameplay4.jpg)
Пример 4
![Result4](resources/tests/test_result4.jpg)
Результат 4
### Готовность проекта

Проект завершен. 

