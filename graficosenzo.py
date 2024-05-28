import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #scikit-learn
from sklearn.preprocessing import StandardScaler

import tratamentododataset as tratamento

def plotarGraficoDispersao(df ,x, y):
    plt.scatter(df[x], df[y])

    plt.title(f'Gráfico de Dispersão: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()


df=tratamento.retornarDataframeTratado()

plotarGraficoDispersao(df, 'Age', 'Quality of Sleep')