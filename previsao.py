import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import main as main



# Mapear valores categóricos para números
occupation_mapping = {
    'Accountant': 0, 'Doctor': 1, 'Engineer': 2, 'Lawyer': 3, 'Manager': 4, 
    'Nurse': 5, 'Sales Representative': 6, 'Salesperson': 7, 'Scientist': 8, 
    'Software Engineer': 9, 'Teacher': 10
}

main.df['Occupation'] = main.df['Occupation'].map(occupation_mapping)

# Transformar outras colunas categóricas em números
label_encoder = LabelEncoder()
main.df['Gender'] = label_encoder.fit_transform(main.df['Gender'])
main.df['BMI Category'] = label_encoder.fit_transform(main.df['BMI Category'])
main.df['Blood Pressure'] = label_encoder.fit_transform(main.df['Blood Pressure'])
main.df['Sleep Disorder'] = label_encoder.fit_transform(main.df['Sleep Disorder'])

# Separar as features (X) e o alvo (y)
X = main.df[['Sleep Duration', 'Occupation', 'Quality of Sleep']]
y = main.df['Age']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar as features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')