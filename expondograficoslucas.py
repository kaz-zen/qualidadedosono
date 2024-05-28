import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tratamentododataset as tratamento
from statsmodels.graphics.mosaicplot import mosaic

def plotarGraficoDispersao(df ,x, y):
    plt.scatter(df[x], df[y])

    plt.title(f'Gráfico de Dispersão: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()

def plotarGraficoMosaico(df, x, y):
    mosaic(df, [x, y], label_rotation=90)

    plt.title(f'Gráfico de Mosaico: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()

def plotarGraficoPizzaDisturbioSono(df):
    contagem = df['Occupation'].value_counts()
    plt.pie(contagem, labels = contagem.index, autopct='%1.1f%%')

    sleepDisorderName = df['Sleep Disorder'].iloc[0]
    plt.title('Profissões com '+ sleepDisorderName)
    plt.show()
    
df = tratamento.retornarDataframeTratado()

plotarGraficoDispersao(df, 'Stress Level', 'Quality of Sleep')

df_apneia = df[df['Sleep Disorder'] == 'Sleep Apnea']
plotarGraficoPizzaDisturbioSono(df_apneia)

df_insomnia = df[df['Sleep Disorder'] == 'Insomnia']
plotarGraficoPizzaDisturbioSono(df_insomnia)

df_none = df[df['Sleep Disorder'] == 'No disorder']
plotarGraficoPizzaDisturbioSono(df_none)


