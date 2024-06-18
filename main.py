import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #scikit-learn
from sklearn.preprocessing import StandardScaler
import tratamentododataset as tratamento

df = tratamento.retornarDataframeTratado()



print(df.head())