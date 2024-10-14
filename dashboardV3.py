import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Configurar el título del dashboard
st.title('CRYPTO Trend Prediction')
user_input_ticket = st.text_input('Enter Stock Ticket', 'BTC-USD')
user_input_StartDate = st.text_input('Enter start date', '2021-01-01')



# Descargar datos
df = yf.download(user_input_ticket, user_input_StartDate)
df = df.drop(columns=['Adj Close'])

# Variables para entrenamiento
cols = list(df)[0:5]
df_for_training = df[cols].astype(float)

# Normalizar el dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Crear datos de entrenamiento
trainX, trainY = [], []
n_future = 1   # Número de días a predecir
n_past = 14  # Número de días pasados a usar para la predicción

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(trainX, trainY, epochs=60, batch_size=20, validation_split=0.1, verbose=1)

# Predecir
n_days_for_prediction = 15
predict_period_dates = pd.date_range(list(df.index)[-1], periods=n_days_for_prediction, freq='1d').tolist()
prediction = model.predict(trainX[-n_days_for_prediction:])

# Inversar la normalización
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]

# Crear un DataFrame para las predicciones
forecast_dates = [time_i.date() for time_i in predict_period_dates]
df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

# Ajustar pronósticos
diferencia = df['Open'].iloc[-1] - df_forecast['Open'].iloc[0]
df_forecast['Open'] += diferencia  # Ajustar todos los valores pronosticados

# Gráfico combinado
st.subheader('Predicción de Precios y Datos Históricos')

# Crear subgráficos usando Plotly
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.1, subplot_titles=("Precio de Cierre", "Volumen", "RSI", "MACD"))


# Gráfico de precios
fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name='Precio Real', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Open'], mode='lines', name='Precio Predicho', line=dict(color='green')), row=1, col=1)

# Gráfico de volumen
volume_color = ['green' if close >= open else 'red' for close, open in zip(df['Close'], df['Open'])]
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volumen', marker_color=volume_color), row=2, col=1)


#_________________________

# Selección de rango de fechas
st.sidebar.subheader("Filtrar por rango de fechas")
start_date = st.sidebar.date_input("Fecha de inicio", df.index.min())
end_date = st.sidebar.date_input("Fecha de fin", df.index.max())



# Filtrar los datos según el rango de fechas
df_filtered = df.loc[start_date:end_date]


# Cálculo del RSI
window_length = 14
delta = df_filtered['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
average_gain = gain.rolling(window=window_length).mean()
average_loss = loss.rolling(window=window_length).mean()
rs = average_gain / average_loss
df_filtered['RSI'] = 100 - (100 / (1 + rs))
fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['RSI'], mode='lines', name='RSI'), row=3, col=1)

# Cálculo del MACD
exp12 = df_filtered['Close'].ewm(span=12, adjust=False).mean()
exp26 = df_filtered['Close'].ewm(span=26, adjust=False).mean()
df_filtered['MACD'] = exp12 - exp26
df_filtered['Signal Line'] = df_filtered['MACD'].ewm(span=9, adjust=False).mean()

# Gráfico del MACD
fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=4, col=1)
fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Signal Line'], mode='lines', name='Signal Line', line=dict(color='red')), row=4, col=1)

# Cálculo de las Bollinger Bands
df_filtered['SMA20'] = df_filtered['Close'].rolling(window=20).mean()
df_filtered['Upper Band'] = df_filtered['SMA20'] + (df_filtered['Close'].rolling(window=20).std() * 2)
df_filtered['Lower Band'] = df_filtered['SMA20'] - (df_filtered['Close'].rolling(window=20).std() * 2)

# Gráfico de las Bollinger Bands
fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Upper Band'], mode='lines', name='Upper Band', line=dict(color='gray')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Lower Band'], mode='lines', name='Lower Band', line=dict(color='gray')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['SMA20'], mode='lines', name='SMA20', line=dict(color='orange')), row=1, col=1)

#__________________________

# Ajustar el diseño
fig.update_layout(height=1200, title_text="Análisis de Precio y Volumen", showlegend=True)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig, use_container_width=True)


