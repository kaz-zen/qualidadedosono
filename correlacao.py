import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def mapearValoresCategoricos(dataframe):
    gender_mapping = {'Male': 0, 'Female': 1}
    occupation_mapping = {'Accountant': 0, 'Doctor': 1, 'Engineer': 2, 'Lawyer': 3, 'Manager': 4, 'Nurse': 5, 'Sales Representative': 6, 'Salesperson': 7, 'Scientist': 8, 'Software Engineer': 9, 'Teacher': 10}
    bmi_mapping = {'Normal': 0, 'Normal Weight': 1, 'Obese': 2, 'Overweight': 3}
    blood_pressure_mapping = {'115/75': 0, '115/78': 1, '117/76': 2, '118/75': 3, '115/76': 4, '119/77': 5, '120/80': 6, '121/79': 7, '122/80': 8, '125/80': 9, '125/82': 10, '126/83': 11, '128/84': 12, '128/85': 13, '129/84': 14, '130/85': 15, '130/86': 16, '131/86': 17, '132/87': 18, '135/88': 19, '136/89': 20, '137/89': 21, '135/90': 22, '139/91': 23, '140/90': 24, '140/95': 25, '142/92': 26}
    sleep_disorder_mapping = {'Insomnia': 0, 'No disorder': 1, 'Sleep Apnea': 2}

    dataframe['Gender'] = dataframe['Gender'].map(gender_mapping)
    dataframe['Occupation'] = dataframe['Occupation'].map(occupation_mapping)
    dataframe['BMI Category'] = dataframe['BMI Category'].map(bmi_mapping)
    dataframe['Blood Pressure'] = dataframe['Blood Pressure'].map(blood_pressure_mapping)
    dataframe['Sleep Disorder'] = dataframe['Sleep Disorder'].map(sleep_disorder_mapping)

    return dataframe

def normalizarColunas(dataframe):
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
    return normalized_data

def removerColunaId(dataframe):
    return dataframe.drop(columns=['Person ID'])

def calcularMatrizCorrelacao(dataframe):
    return dataframe.corr()

def plotarMatrizCorrelacaoHeatmap(correlation_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.show()

def plotarGraficoCorrelacao(dataframe):
    dataMapeada = mapearValoresCategoricos(dataframe)
    dataNormalizada = normalizarColunas(dataMapeada)
    dfSemId = removerColunaId(dataNormalizada)
    matrizCorrelacao = calcularMatrizCorrelacao(dfSemId)
    plotarMatrizCorrelacaoHeatmap(matrizCorrelacao)

