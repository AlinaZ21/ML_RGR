import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import tree

df = pd.read_csv(r'C:/Users/AlinaZ/WebApplication/data/Australian_Rains.csv')

st.write("Загруженный датасет 'Australian Rains'")

st.header("Гистограммы")

columns = ["MaxTemp","Rainfall","WindGustSpeed"]

for col in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)[col], bins=100, kde=True)
    plt.title(f'Гистограмма для {col}')
    st.pyplot(plt)


st.header("Тепловая карта с корреляцией между основными признаками")

plt.figure(figsize=(12, 8))
columns_warm = ['WindGustSpeed','Humidity3pm','Rainfall', 'Cloud3pm', 'RainToday', 'RainTomorrow']
df_warm = df[columns_warm]
sns.heatmap(df_warm.corr(), annot=True, cmap='coolwarm')
plt.title('Тепловая карта с корреляцией')
st.pyplot(plt)


st.header("Ящики с усами ")

outlier = df[["RainTomorrow","Cloud3pm","WindGustSpeed"]]
Q1 = outlier.quantile(0.25)
Q3 = outlier.quantile(0.75)
IQR = Q3-Q1
data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) | (outlier > (Q3 + 1.5 * IQR))).any(axis=1)]

index_list = list(data_filtered.index.values)
data_filtered = df[df.index.isin(index_list)]

data_filtered.boxplot(by='RainTomorrow', column='WindGustSpeed', grid=True)
data_filtered.boxplot(by='RainTomorrow', column='Cloud3pm', grid=True)
st.pyplot(plt)

st.header("Дерево решений")

x=pd.DataFrame(df.drop(['RainTomorrow'],axis=1))
with open(r'C:/Users/AlinaZ/WebApplication/models/DTC.pkl', 'rb') as file:
        DTC = pickle.load(file)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(DTC,feature_names=x.columns.values.tolist(), class_names=["Yes", "No"] , filled=True)
st.pyplot(plt)

st.header("Круговая диаграмма в какой месяц сколько измерений было сделано (в процентах от общего количества измерений)")

plt.figure(figsize=(8, 8))
df['Date'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Date')
plt.ylabel('')
st.pyplot(plt)






