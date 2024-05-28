import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import main as main

# Gráfico Ocupação x Qualidade do sono
plt.figure(figsize=(12, 6))
sns.barplot(x='Occupation', y='Quality of Sleep', data=main.df, ci=None)
plt.title('Qualidade do Sono por Ocupação')
plt.xlabel('Ocupação')
plt.ylabel('Qualidade do Sono')
plt.xticks(rotation=45)
plt.show()

# Gráfico de dispersão entre 'Sleep Duration' e 'Age'
plt.figure(figsize=(12, 6))
sns.barplot(x='Age', y='Sleep Duration', data=main.df, ci=None)
plt.title('Relação entre Duração do Sono e Idade')
plt.xlabel('Idade')
plt.ylabel('Duração do Sono')
plt.show()