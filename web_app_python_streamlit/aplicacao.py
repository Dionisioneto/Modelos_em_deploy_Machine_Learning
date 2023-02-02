## Bibliotecas 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier # Classificador

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

## Título

st.write("""
         Previsão de Diabetes \n
         Wep App construído sobre algoritmos de Machine Learning para a predição de pacientes com diabetes. \n
         Fonte: PIMA - India (Kaggle)
         """)

# dataset
dados = pd.read_csv('diabetes.csv')

# Cabeçalho
st.subheader("Informações dos dados")


# Nome do usuário
user_input = st.sidebar.text_input("Diabetes seu nome")

st.write("Paciente: ", user_input)

## Treinamento do modelo

X = dados.drop(["Outcome"], 1) ## features
y = dados["Outcome"]  ## target

## Separação entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12)

## Capturação dos dados do usuário com a função

def dados_usuario():
    pregnancies = st.sidebar.slider("Gravidez", 0, 15, 1) ## label, barra de deslizar, mínimo valor, máximo valor.
    age = st.sidebar.slider("Idade", 15, 100, 21)
    glucose = st.sidebar.slider("Glicose", 0, 200, 110)
    blood_preassure = st.sidebar.slider("Pressão sanguínea", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Espessura da pele", 0, 99, 20)
    insulin = st.sidebar.slider("Insulina", 0, 900, 30)
    bmi = st.sidebar.slider("Índice de Massa Coporal", 0.0, 70.0, 15.0)
    dpf = st.sidebar.slider("Histórico familiar de diabetes", 0.0, 3.0, 0.0)

    ## Dicionário para o armazenamento de informações
    user_data = {"gravidez": pregnancies,
                "idade": age,
                "glicose": glucose,
                "pressao_sanguinea": blood_preassure,
                "espessura_pele": skin_thickness,
                "insulina": insulin,
                "massa_corporal": bmi,
                "historico_familiar": dpf}

    ## dataframe para armazenar os dados

    features_input = pd.DataFrame(user_data, index = [0])

    return features_input

user_input_variables = dados_usuario()

## Gráfico das informações do indivíduo
graf = st.bar_chart(user_input_variables)


st.subheader("Dados do usuário")
st.write(user_input_variables)


## Treinamento do modelo
clf = RandomForestClassifier(n_estimators=500, random_state=12, min_samples_split=5, max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## Acurácia do modelo
st.subheader("Acurácia do modelo")
st.write(accuracy_score(y_test, y_pred)*100)

## Previsão

prediction = clf.predict(np.array(user_input_variables).reshape(1,-1))

st.subheader("Previsão")
st.write(prediction)

