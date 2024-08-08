import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #scikit-learn
from sklearn.preprocessing import StandardScaler
import tratamentododataset as tratamento
import correlacao as correlacao  
import graficos as graficos

df = tratamento.retornarDataframeTratado()

correlacao.plotarGraficoCorrelacao(df)
graficos.plotarGraficos(df)

print(df.head())