import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tratamentododataset as tratamento
from statsmodels.graphics.mosaicplot import mosaic


def retornarDataframeTratado():
    url = "Dataset\\Sleep_health_and_lifestyle_dataset.csv"
    df = pd.read_csv(url)
    dfTratado = tratarDataframe(df)
    return dfTratado

def tratarDataframe(df):
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('No disorder')
    return df

def plotarHistogramas(df, x, hue):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=x, hue=hue, multiple="stack", kde=True)
    plt.title(f'Histograma: {x} com {hue}')
    plt.xlabel(x)
    plt.ylabel('Frequência')
    plt.show()

def plotarBoxPlot(df, coluna):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[coluna])
    plt.title(f'Box Plot of {coluna}')
    plt.show()

# Obtendo o DataFrame tratado
df = retornarDataframeTratado()

# Plotando gráficos box plot para "Quality of Sleep" e "Sleep Duration"
plotarBoxPlot(df, 'Quality of Sleep')
plotarBoxPlot(df, 'Sleep Duration')
plotarHistogramas(df, 'Physical Activity Level', 'Sleep Disorder')