# Команда 22
* Болотов М.В.
* Пахомов Д.Е.
* Султанов Э.М.
* Шибакова А.А.

 
# Проект "Прогнозирование длительности поездок в такси"
## Описание проделанной работы

## Постановка задачи

**Бизнес-задача:**
Определить характеристики и с их помощью спрогнозировать длительность поездки такси.

**Техническая задача:**
Построить модель машинного обучения, которая на основе предложенных характеристик клиента будет предсказывать числовой признак - время поездки такси. Решается задача регрессии.

#### Основные Цели Проекта:

1. **Формирование набора данных:**
   - Сформировать набор данных на основе нескольких источников информации.

2. **Разведывательный анализ данных (EDA):**
   - Спроектировать новые признаки с помощью Feature Engineering и выявить наиболее значимые при построении модели.

3. **Исследование данных:**
   - Исследовать предоставленные данные и выявить закономерности.

4. **Построение моделей:**
   - Построить несколько моделей и выбрать из них ту, которая показывает наилучший результат по заданной метрике.

5. **Процесс предсказания:**
   - Спроектировать процесс предсказания длительности поездки для новых данных.

## 1. Знакомство с данными, базовый анализ и расширение данных

**Знакомство с данными и базовый анализ:**
- Ознакомление с предоставленными данными о 1.5 миллионах поездок и 11 характеристиках.
- Данные были классифицированы по нескольким группам: данные о клиенте и таксопарке, временные характеристики, географическая информация, прочие признаки, и целевой признак.
- Проведён базовый анализ состояния данных, включая проверку на наличие пропущенных значений (результат: пропущенных значений нет).

**Статистический анализ данных:**
- Определены временные рамки данных: с 2016-01-01 по 2016-06-30.
- Количество уникальных таксопарков: 2.
- Максимальное количество пассажиров: 9.
- Средняя длительность поездки: 959 секунд.
- Медианная длительность поездки: 662 секунды.
- Минимальная длительность поездки: 1 секунда.
- Максимальная длительность поездки: 3526282 секунды.

**Расширение набора данных:**
- Реализованы функции для добавления новых признаков к исходному набору данных:
  - `add_datetime_features()`: добавлены столбцы `pickup_date`, `pickup_hour`, `pickup_day_of_week`.
  - `add_holiday_features()`: добавлен столбец `pickup_holiday`.
  - `add_osrm_features()`: добавлены столбцы `total_distance`, `total_travel_time`, `number_of_steps`.
  - `add_geographical_features()`: добавлены столбцы `haversine_distance`, `direction`.
  - `add_cluster_features()`: добавлен столбец `geo_cluster`.
  - `add_weather_features()`: добавлены столбцы `temperature`, `visibility`, `wind_speed`, `precip`, `events`.
  - `fill_null_weather_data()`: заполнены пропуски в столбцах с погодными условиями и информацией из OSRM API.

**Анализ данных после расширения:**
- Проанализированы поездки в праздничные дни и добавлены соответствующие признаки.
- Проанализированы данные из OSRM и добавлены соответствующие признаки.
- Вычислены медианные величины различных признаков после заполнения пропусков.
- Найдены очевидные выбросы в данных, используя анализ длительности и средней скорости поездок.

**Выбросы:**
- Определены выбросы по длительности поездок (длительность более 24 часов).
- Определены выбросы по скорости (скорость более 300 км/ч).
Всего выбросов:
- По признаку длительности поездки: 4.
- По признаку скорости: 407.

 ## 2. Разведывательный анализ данных (EDA)

В ходе анализа данных о поездках такси в Нью-Йорке были построены различные визуализации для выявления зависимостей и закономерностей. 

- С 00:00 по 5:00 количество заказов такси минимально, что может быть связано с тем, что в это время люди предпочитают оставаться дома или пользоваться другими видами транспорта.
- С 13:00 по 18:00 наблюдается пик медианной длительности поездок. Это может быть связано с тем, что в это время дня на дорогах больше транспортных пробок или увеличивается спрос на такси из-за пиковой нагрузки на общественный транспорт.
- Наибольшее количество поездок совершается в выходные дни - субботу и воскресенье, что может быть связано с тем, что люди чаще пользуются такси в выходные для поездок на отдых или по делам.
- Медианная длительность поездки в течение недели колеблется, достигая пика в среду. Это может быть связано с тем, что в середине недели люди чаще совершают более длительные поездки, возможно, из-за работы или других обстоятельств.
- Наибольшее количество поездок совершается в пятницу, что может быть связано с окончанием рабочей недели и желанием людей провести время на отдыхе или встречах с друзьями.
- В воскресенье медианная длительность поездок наименьшая, что может быть связано с тем, что в выходные дни люди предпочитают более короткие поездки для отдыха и развлечений.

Для анализа данных о длительности поездок в зависимости от времени суток и дня недели была построена сводная таблица. В таблице по строкам указаны часы начала поездки (pickup_hour), по столбцам - день недели (pickup_day_of_week), а в ячейках указана медианная длительность поездки (trip_duration).

По результатам анализа выяснилось, что самые продолжительные поездки (в медианном смысле) наблюдаются с понедельника по пятницу в промежутке с 8 до 18 часов. Наибольшая медианная длительность поездки наблюдалась в четверг в 14 часов дня.

Для визуализации данных была построена тепловая карта, на которой можно наглядно увидеть зависимость длительности поездки от времени суток и дня недели.

Для построения диаграмм рассеяния были использованы данные о координатах начала и завершения поездок такси в Нью-Йорке. Для удобства анализа были построены две scatter-диаграммы.

На первой диаграмме представлено географическое расположение точек начала поездок (pickup_longitude, pickup_latitude).

На второй диаграмме показано географическое расположение точек завершения поездок (dropoff_longitude, dropoff_latitude).

Из диаграмм видно, что большинство точек начала и завершения поездок сосредоточены в центре Нью-Йорка, что соответствует ожидаемому распределению. Однако, 2 кластера из десяти не попали на диаграммы, что может свидетельствовать о том, что эти точки находятся за границами Нью-Йорка.

## 3. Отбор и преобразование признаков
 
### Исключение неинформативных признаков:

- Идентифицирован и исключен уникальный для каждой поездки признак, не несущий полезной информации.
- Объяснено понятие утечки данных и исключен признак "dropoff_datetime" из обучающего набора данных.
- После исключения указанных признаков, в таблице осталось 25 столбцов.

### Кодирование признаков:

- Проведено кодирование признаков "vendor_id" и "store_and_fwd_flag" в соответствии с заданием.
- Рассчитано среднее по закодированным столбцам "vendor_id" (0.53) и "store_and_fwd_flag" (0.006).

### One-Hot Encoding:

- Создана таблица "data_onehot" с закодированными признаками "pickup_day_of_week", "geo_cluster" и "events".
- Сгенерировано 18 бинарных столбцов с помощью One-Hot Encoding.

### Выбор признаков с помощью SelectKBest:

- Применен метод SelectKBest с score_func=f_regression для отбора 25 признаков.

### Нормализация предикторов:

- Произведена нормализация предикторов в обучающей и валидационной выборках с помощью MinMaxScaler.
- Рассчитано среднее арифметическое для первого предиктора из валидационной выборки (0.54).

## 4. Этап моделирования

В ходе работы были построены и оценены различные модели машинного обучения для прогнозирования длительности поездки такси. В качестве метрики качества использовался RMSLE (Root Mean Squared Log Error).

### Линейная регрессия

Первой была построена модель линейной регрессии. Метрика RMSLE на тренировочной и валидационной выборках составила 0.54.

### Дерево решений

Затем была построена модель дерева решений. Метрика RMSLE на тренировочной выборке составила 0, а на валидационной выборке - 0.54. Были замечены признаки переобучения модели. После подбора оптимальной глубины дерева (12), метрика RMSLE на тренировочной выборке составила 0.41, а на валидационной выборке - 0.43.

### Случайный лес

Была построена модель случайного леса с гиперпараметрами: n_estimators = 200, max_depth = 12, criterion = 'squared_error', min_samples_split = 20, random_state = 42. Метрика RMSLE на тренировочной выборке составила 0.40, а на валидационной выборке - 0.41.

### Градиентный бустинг

Была построена модель градиентного бустинга над деревьями решений с гиперпараметрами: learning_rate = 0.5, n_estimators = 100, max_depth = 6, min_samples_split = 30, random_state = 42. Метрика RMSLE на тренировочной выборке составила 0.37, а на валидационной выборке - 0.39.

Из всех построенных моделей наилучший результат показала модель градиентного бустинга над деревьями решений.

### Важность факторов

Была проведена оценка важности факторов для модели градиентного бустинга. Топ-3 наиболее значимых фактора для предсказания длительности поездки в логарифмическом масштабе: total_distance, total_travel_time, pickup_hour.


![img](https://drive.google.com/uc?export=view&id=1Q1YBuFJjOHh9CCTa34Ta5va7tO61_bGH)



### Медианная абсолютная ошибка

Для лучшей из построенных моделей была рассчитана медианная абсолютная ошибка (MeAE) предсказания длительности поездки такси на валидационной выборке. Значение метрики MeAE составило 1.8 минуты.

### Предсказание для тестового набора данных

Было сделано предсказание для отложенного тестового набора данных. Перед созданием прогноза для тестовой выборки были произведены все необходимые манипуляции с данными.

### Экстремальный градиентный бустинг

В завершение работы была построена модель экстремального градиентного бустинга (XGBoost). В среде Colab полученные результаты RMSLE составили 0.40269.

### Автоматическая оптимизация гиперпараметров с помощью фреймворка Optuna
Для автоматической оптимизации гиперпараметров и улучшения качества метрики RMSLE был использован фреймворк Optuna. На основе гиперпараметров, полученных с помощью Optuna, в среде Google Colab результаты RMSLE составили 0.38543, а на платформе Kaggle - 0.39423.

![img](https://drive.google.com/uc?export=view&id=1BRmiZXm1LYvScsxqe7t35yDdJ7OGF2d7)

![img](https://drive.google.com/uc?export=view&id=1Zjnk4vLOIx0j1eBkPwu7hYj16_CPu8JQ)

В ходе работы были построены и оценены различные модели машинного обучения для прогнозирования длительности поездки такси. Наилучший результат показала модель Экстремального градиентного бустинга (XGBoost).

![img](https://drive.google.com/uc?export=view&id=1MvysKUhKym4X5BQsN2sSJcCvjWb5Ohea)

## 5. Процесс предсказания
После завершения процесса формирования предсказания длительности поездок на тестовой выборке, был создан submission-файл и отправлен на платформу Kaggle для оценки значения метрики RMSLE на тестовой выборке. Полученные на платформе Kaggle результаты RMSLE составляют 0.39423.


![img](https://drive.google.com/uc?export=view&id=1N6zTN09yImW3erDPhSpOQgGR7MLmVzgo)


Было создано веб-приложение, использующее обученную модель экстремального градиентного бустинга (XGBoost) для прогнозирования общей продолжительности поездки такси в Нью-Йорке.
Веб-приложение развернуто на Hugging Face Spaces и доступно по следующей [ссылке](https://huggingface.co/spaces/Emil25/practicum_2).

![img](https://drive.google.com/uc?export=view&id=1GQLezy86ldo1Gwp7vrHHahuQaF03wTfr)
