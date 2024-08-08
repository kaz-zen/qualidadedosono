import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Função para carregar o dataset
def load_dataset(url):
    df = pd.read_csv(url)
    return df

# Função para preencher valores ausentes
def fill_missing_values(df):
    df['Sleep Disorder'].fillna('No disorder', inplace=True)
    return df

# Função para mapear valores categóricos
def map_occupation(df):
    occupation_mapping = {
        'Accountant': 0, 'Doctor': 1, 'Engineer': 2, 'Lawyer': 3, 'Manager': 4, 
        'Nurse': 5, 'Sales Representative': 6, 'Salesperson': 7, 'Scientist': 8, 
        'Software Engineer': 9, 'Teacher': 10
    }
    df['Occupation'] = df['Occupation'].map(occupation_mapping)
    return df, occupation_mapping

# Função para transformar colunas categóricas em números
def encode_categorical_columns(df):
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['BMI Category'] = label_encoder.fit_transform(df['BMI Category'])
    df['Blood Pressure'] = label_encoder.fit_transform(df['Blood Pressure'])
    df['Sleep Disorder'] = label_encoder.fit_transform(df['Sleep Disorder'])
    return df, label_encoder

# Função para definir características e alvos
def define_features_and_target(df):
    X = df[['Age', 'Occupation', 'Stress Level', 'Sleep Disorder']]
    y = df[['Sleep Duration', 'Quality of Sleep']]
    return X, y

# Função para dividir os dados em treino e teste
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Função para treinar o modelo
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Função para fazer previsões e avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

# Função para fazer previsões baseadas em entrada do usuário
def predict_sleep(model, age, occupation, stress_level, sleep_disorder, occupation_mapping, label_encoder):
    occupation_encoded = occupation_mapping[occupation]
    sleep_disorder_encoded = label_encoder.transform([sleep_disorder])[0]
    
    input_data = pd.DataFrame([[age, occupation_encoded, stress_level, sleep_disorder_encoded]], 
                              columns=['Age', 'Occupation', 'Stress Level', 'Sleep Disorder'])
    
    prediction = model.predict(input_data)
    return prediction