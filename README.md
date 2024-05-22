# Проект "Прогнозирование Длительности Поездок на Такси"
## Описание проделанной работы

## 1. Постановка задачи

**Бизнес-задача:**
Определить характеристики и с их помощью спрогнозировать длительность поездки такси.

**Техническая задача:**
Построить модель машинного обучения, которая на основе предложенных характеристик клиента будет предсказывать числовой признак - время поездки такси. Решается задача регрессии.

#### Основные Цели Проекта:

1. **Формирование набора данных:**
   - Собрать данные из нескольких источников информации.

2. **Feature Engineering:**
   - Создать новые признаки на основе имеющихся данных и выявить наиболее значимые при построении модели.

3. **Исследование данных:**
   - Провести анализ предоставленных данных и выявить закономерности.

4. **Построение моделей:**
   - Разработать несколько моделей и выбрать наилучшую по заданной метрике.

5. **Процесс предсказания:**
   - Спроектировать процесс предсказания времени длительности поездки для новых данных.

## 2. Знакомство с данными, базовый анализ и расширение данных

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

 ## 3. Разведывательный анализ данных (EDA):

В ходе анализа данных о поездках такси в Нью-Йорке были построены различные визуализации для выявления зависимостей и закономерностей. 

## Визуализации

### Распределение количества поездок в зависимости от часа дня

Первая визуализация отображает распределение количества поездок в зависимости от часа дня. Из графика видно, что с 00:00 по 5:00 количество заказов такси минимально, что может быть связано с тем, что в это время люди предпочитают оставаться дома или пользоваться другими видами транспорта.

### Зависимость медианной длительности поездки от часа дня

Вторая визуализация показывает зависимость медианной длительности поездки от часа дня. На основе этого графика можно сделать вывод, что с 13:00 по 18:00 наблюдается пик медианной длительности поездок. Это может быть связано с тем, что в это время дня на дорогах больше транспортных пробок или увеличивается спрос на такси из-за пиковой нагрузки на общественный транспорт.

### Зависимость количества поездок и медианной длительности поездки от дня недели

#### Количество поездок по дням недели

Первая визуализация показала, что наибольшее количество поездок совершается в выходные дни - субботу и воскресенье, что может быть связано с тем, что люди чаще пользуются такси в выходные для поездок на отдых или по делам.

#### Медианная длительность поездки по дням недели

Вторая визуализация показала, что медианная длительность поездки в течение недели колеблется, достигая пика в среду. Это может быть связано с тем, что в середине недели люди чаще совершают более длительные поездки, возможно, из-за работы или других обстоятельств.

### Зависимость количества поездок и медианной длительности поездки от дня недели

#### Количество поездок по дням недели

Первая визуализация показала, что наибольшее количество поездок совершается в пятницу, что может быть связано с окончанием рабочей недели и желанием людей провести время на отдыхе или встречах с друзьями.

#### Медианная длительность поездки по дням недели

Вторая визуализация показала, что в воскресенье медианная длительность поездок наименьшая, что может быть связано с тем, что в выходные дни люди предпочитают более короткие поездки для отдыха и развлечений.

## Анализ данных о длительности поездок

Для анализа данных о длительности поездок в зависимости от времени суток и дня недели была построена сводная таблица. В таблице по строкам указаны часы начала поездки (pickup_hour), по столбцам - день недели (pickup_day_of_week), а в ячейках указана медианная длительность поездки (trip_duration).

### Результаты анализа

По результатам анализа выяснилось, что самые продолжительные поездки (в медианном смысле) наблюдаются с понедельника по пятницу в промежутке с 8 до 18 часов. Наибольшая медианная длительность поездки наблюдалась в четверг в 14 часов дня.

### Тепловая карта

Для визуализации данных была построена тепловая карта, на которой можно наглядно увидеть зависимость длительности поездки от времени суток и дня недели.

### Диаграммы рассеяния

Для построения диаграмм рассеяния были использованы данные о координатах начала и завершения поездок такси в Нью-Йорке. Для удобства анализа были построены две scatter-диаграммы.

#### Географическое расположение точек начала поездок

На первой диаграмме представлено географическое расположение точек начала поездок (pickup_longitude, pickup_latitude).

#### Географическое расположение точек завершения поездок

На второй диаграмме показано географическое расположение точек завершения поездок (dropoff_longitude, dropoff_latitude).

### Выводы

Из диаграмм видно, что большинство точек начала и завершения поездок сосредоточены в центре Нью-Йорка, что соответствует ожидаемому распределению. Однако, 2 кластера из десяти не попали на диаграммы, что может свидетельствовать о том, что эти точки находятся за границами Нью-Йорка.

## 4. Отбор и преобразование признаков
 
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
- В список отобранных признаков вошли: 'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'pickup_hour', 'pickup_holiday', 'total_distance', 'total_travel_time', 'number_of_steps', 'haversine_distance', 'temperature', 'pickup_day_of_week_1', 'pickup_day_of_week_2', 'pickup_day_of_week_3', 'pickup_day_of_week_4', 'pickup_day_of_week_5', 'pickup_day_of_week_6', 'geo_cluster_1', 'geo_cluster_3', 'geo_cluster_5', 'geo_cluster_7', 'geo_cluster_9'.

### Нормализация предикторов:

- Произведена нормализация предикторов в обучающей и валидационной выборках с помощью MinMaxScaler.
- Рассчитано среднее арифметическое для первого предиктора из валидационной выборки (0.54).
