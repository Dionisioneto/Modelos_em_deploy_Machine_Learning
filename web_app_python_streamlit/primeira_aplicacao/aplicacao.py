## Bibliotecas 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier # Classificador

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
