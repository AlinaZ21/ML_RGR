import streamlit as st

st.title("Информация о наборе данных")
st.header("Тематика датасета: дожди в Австралии")
st.header("Описание признаков:")
st.write("- Date: Дата наблюдения")
st.write("- MinTemp: Минимальная температура в градусах Цельсия")
st.write("- MaxTemp: Максимальная температура в градусах Цельсия")
st.write("- Rainfall: Количество осадков, выпавших за сутки в мм")
st.write("- Evaporation: Так называемое испарение поддона класса А (мм) за 24 часа до 9 утра")
st.write("- Sunshine: Количество часов яркого солнечного света в сутках")
st.write("- WindGustSpeed: Скорость (км/ч) самого сильного порыва ветра за 24 часа до полуночи.")
st.write("- WindSpeed9am: Скорость ветра в 9 часов утра")
st.write("- WindSpeed3pm: Скорость ветра в 3 часа дня")
st.write("- Humidity9am: Влажность в 9 часов утра")
st.write("- Humidity3pm: Влажность в 3 дня")
st.write("- Pressure9am: Давление в 9 часов утра")
st.write("- Pressure3pm: Давление в 3 часа дня")
st.write("- Cloud9am: Погода в 9 часов утра")
st.write("- Cloud3pm: Погода в 3 часа дня")
st.write("- Temp9am: Погода в 9 часов утра")
st.write("- Temp3pm: Температура в 3 часа дня")
st.write("- RainToday: Был ли дождь сегодня")
st.write("- RainTomorrow: будет ли дождь завтра завтра")
st.write("- Location: Общее название места расположения метеостанции")
st.write("- WindGustDir: Направление самого сильного порыва ветра за сутки до полуночи")
st.write("- WindDir9am: Направление ветра в 9 часов утра")
st.write("- WindDir3pm: Направление ветра в 3 часа дня")
st.header("Предобработка данных")
st.write("Датасет создан для пресказывания дождя завтра в Австралии: 1 - дождь будет, 0 - дождя не будет")
st.write("Наибольшее влияние на погоду оказывает время года, то есть месяц. Поэтому вместо даты был оставлен только номер месяца")
st.write("В датасете присутствовали категориальные признаки. Они были переведены в бинарные:")
st.write("- в столбцах Rain today и Rain Tomorrow замена No, Yes на 0,1;")
st.write("- применён BinaryEncoder.")
st.write("Пропущенные значения и явные дубликаты были удалены")
st.write("Числовые признаки были масштабированны. Был устранен дисбаланс классов.")