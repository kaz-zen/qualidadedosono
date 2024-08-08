from modelo_previsao_final import (
    load_dataset, fill_missing_values, map_occupation,
    encode_categorical_columns, define_features_and_target,
    split_data, train_model, evaluate_model, predict_sleep
)

url = "https://drive.google.com/uc?id=11BAOuVd8eBgy4bQ1XSfpPBs0sPI4HYzq"
    
# Executar o pipeline completo
df = load_dataset(url)
df = fill_missing_values(df)
df, occupation_mapping = map_occupation(df)
df, label_encoder = encode_categorical_columns(df)
X, y = define_features_and_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)
mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    
# Exibir resultados
print(f'R^2 Score: {r2}')
print(f'Mean Squared Error: {mse}')
    
# Exemplo de uso da função de previsão
age = 25
occupation = 'Software Engineer'
stress_level = 3
sleep_disorder = 'No disorder'

predicted_sleep_duration, predicted_quality_of_sleep = predict_sleep(model, age, occupation, stress_level, sleep_disorder, occupation_mapping, label_encoder)[0]
print(f'Predicted Sleep Duration: {predicted_sleep_duration}')
print(f'Predicted Quality of Sleep: {predicted_quality_of_sleep}')