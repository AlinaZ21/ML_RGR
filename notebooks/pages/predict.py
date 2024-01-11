import streamlit as st
import pandas as pd

def predict_word(pred):
    if pred == 1: return "Завтра будет дождь"
    else: return "Завтра дождя не будет"

uploaded_file = st.file_uploader("Выберите файл с датасетом", type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Загруженный датасет:", df)

else:
    df = pd.read_csv('data/Australian_Rains.csv')
    st.write("Загруженный датасет 'Australian Rains':", df)

with open('models/ML1.pkl', 'rb') as file:
        ML1 = pickle.load(file)

with open('models/ML2.pkl', 'rb') as file:
    ML2 = pickle.load(file)

with open('models/ML3.pkl', 'rb') as file:
    ML3 = pickle.load(file)

with open('models/ML4.pkl', 'rb') as file:
    ML4= pickle.load(file)

with open('models/ML5.pkl', 'rb') as file:
    ML5 = pickle.load(file)

from tensorflow.keras.models import load_model
ML6 = load_model('models/ML6.keras')

x = pd.DataFrame(df.drop(['RainTomorrow'], axis = 1))
y = pd.DataFrame(df['RainTomorrow'])


st.title("Получить предсказания дождя")
st.header("Заполните данные на основе которых будет сделано предсказание")

st.header("Date")
Date = st.selectbox("месяц", [1,2,3,4,5,6,7,8,9,10,11,12])

st.header("Location")
locations = ['Cobar', 'CoffsHarbour', 'Moree', 'NorfolkIsland', 'Sydney',
    'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Canberra', 'Sale',
    'MelbourneAirport', 'Melbourne', 'Mildura', 'Portland', 'Watsonia',
    'Brisbane', 'Cairns', 'Townsville', 'MountGambier', 'Nuriootpa',
    'Woomera', 'PerthAirport', 'Perth', 'Hobart', 'AliceSprings',
    'Darwin']
Location = st.selectbox("Город", locations)

st.header("MinTemp")
MinTemp = st.number_input("Число:", value=18.4)

st.header("MaxTemp")
MaxTemp = st.number_input("Число:", value=29.7)

st.header("Rainfall")
Rainfall = st.number_input("Число:", value=2)

st.header("Evaporation")
Evaporation = st.number_input("Число:", value=7.4)

st.header("Sunshine")
Sunshine = st.number_input("Число:", value=10.7)

st.header("WindGustDir")
dirs=['SSW', 'S', 'NNE', 'WNW', 'N', 'SE', 'ENE', 'NE', 'E', 'SW', 'W',
    'WSW', 'NNW', 'ESE', 'SSE', 'NW']
WindGustDir = st.selectbox("Направление", dirs)

st.header("WindGustSpeed")
WindGustSpeed = st.number_input("Число:", value=30)

st.header("WindDir9am")
dirs2=['ENE', 'SSE', 'NNE', 'WNW', 'NW', 'N', 'S', 'SE', 'NE', 'W', 'SSW',
    'E', 'NNW', 'ESE', 'WSW', 'SW']
WindDir9am = st.selectbox("Направление", dirs2)

st.header("WindDir3pm")
dirs3=['SW', 'SSE', 'NNW', 'WSW', 'WNW', 'S', 'ENE', 'N', 'SE', 'NNE',
    'NW', 'E', 'ESE', 'NE', 'SSW', 'W']
WindDir3pm = st.selectbox("Направление", dirs3)

st.header("WindSpeed9am")
WindSpeed9am = st.number_input("Число:", value=11)

st.header("WindSpeed3pm")
WindSpeed3pm = st.number_input("Число:", value=7)

st.header("Humidity9am")
Humidity9am = st.number_input("Число:", value=27)

st.header("Humidity3pm")
Humidity3pm = st.number_input("Число:", value=9)

st.header("Pressure9am")
Pressure9am = st.number_input("Число:", value=1012.6)

st.header("Pressure3pm")
Pressure3pm = st.number_input("Число:", value=1010.1)

st.header("Cloud9am")
Cloud9am = st.number_input("Число:", value=0.1)

st.header("Cloud3pm")
Cloud3pm = st.number_input("Число:", value=1)

st.header("Temp9am")
Temp9am = st.number_input("Число:", value=29.8)

st.header("Temp3pm")
Temp3pm = st.number_input("Число:", value=36.4)

st.header("RainToday")
RainToday = st.number_input("Число:", value=0, min_value=0, max_value=1)

data = pd.DataFrame({'Date': [Date],
                    'Location': [Location],
                    'MinTemp': [MinTemp],
                    'MaxTemp': [MaxTemp],
                    'Rainfall': [Rainfall],
                    'Evaporation': [Evaporation],
                    'Sunshine': [Sunshine],
                    'WindGustDir': [WindGustDir],
                    'WindGustSpeed': [WindGustSpeed],
                    'WindDir9am': [WindDir9am],
                    'WindDir3pm': [WindDir3pm],
                    'WindSpeed9am': [WindSpeed9am],
                    'WindSpeed3pm': [WindSpeed3pm],
                    'Humidity9am': [Humidity9am],    
                    'Humidity3pm': [Humidity3pm],   
                    'Pressure9am': [Pressure9am],   
                    'Pressure3pm': [Pressure3pm],   
                    'Cloud9am': [Cloud9am],   
                    'Cloud3pm': [Cloud3pm],   
                    'Temp9am': [Temp9am],   
                    'Temp3pm': [Temp3pm],   
                    'RainToday': [RainToday],  
                    'RainTomorrow': [0]       
                    })



with open('models/Binary_Encoder.pkl', 'rb') as file:
    BinEncod = pickle.load(file)

data_category = BinEncod.fit_transform(df.select_dtypes(include=['object'])).astype(int)
data_num = data.select_dtypes(exclude=['object'])
data = pd.concat([data_num, pd.DataFrame(data_category)], axis=1)

x = pd.DataFrame(df.drop(['RainTomorrow'], axis = 1))

button_clicked = st.button("Предсказать погоду")

if button_clicked:

    st.header("Logistir regression:")
    pred =[]
    log_reg_pred = ML1.predict(x)[0]
    pred.append(int(log_reg_pred))
    st.write(predict_word(log_reg_pred))

    st.header("Gradient boosting:")
    gradient_pred = ML3.predict(x)[0]
    pred.append(int(gradient_pred))
    st.write(predict_word(gradient_pred))

    st.header("Bagging:")
    bagging_pred = ML4.predict(x)[0]
    pred.append(int(bagging_pred))
    st.write(predict_word(bagging_pred))

    st.header("Stacking:")
    stacking_pred = ML5.predict(x)[0]
    pred.append(int(stacking_pred))
    st.write(predict_word(stacking_pred))

    st.header("NN:")
    nn_pred = round(ML6.predict(x)[0][0])
    pred.append(nn_pred)
    st.write(predict_word(nn_pred))

    st.title("Точность моделей:")
    
    from sklearn.metrics import accuracy_score

    st.header("Logistic regression:")
    log_reg_pred = ML1.predict(x)
    st.write(f"{accuracy_score(y, log_reg_pred)}")

    st.header("Gradient boosting:")
    gradient_pred = ML3.predict(x)
    st.write(f"{accuracy_score(y, gradient_pred)}")

    st.header("Bagging:")
    bagging_pred = ML4.predict(x)
    st.write(f"{accuracy_score(y, bagging_pred)}")

    st.header("Stacking:")
    stacking_pred = ML5.predict(x)
    st.write(f"{accuracy_score(y, stacking_pred)}")

    st.header("NN:")
    nn_pred = [np.argmax(pred) for pred in ML6.predict(x, verbose=None)]
    st.write(f"{accuracy_score(y, nn_pred)}")