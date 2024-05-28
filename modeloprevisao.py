import main as main
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Codificar a coluna 'profissao'
le = LabelEncoder()
main.df['Occupation'] = le.fit_transform(main.df['Occupation'])
main.df['Sleep Disorder'] = le.fit_transform(main.df['Sleep Disorder'])

# Definir características e alvos
X = main.df[['Age', 'Occupation', 'Stress Level', 'Sleep Disorder']]
y = main.df[['Sleep Duration', 'Quality of Sleep']]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(f'R^2 Score: {r2}')

print(f'Mean Squared Error: {mse}')
