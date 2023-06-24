import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st



start = '2007-01-01'
end = '2028-06-27'  # Update to 5 years ahead of the current end date




st.title('Stock Trend Prediction')

user_input = st.text_input("Enter Stock Ticker",'....')

df = yf.download(user_input,start,end)

#describe data

st.subheader('Data from 2010 - 2023')
st.write(df.describe())



st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Clsoing Price  vs Time chart with 100MA')
ma100 =df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Clsoing Price  vs Time chart with 200MA')
ma100 =df.Close.rolling(100).mean()
ma200 =df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


#spiliting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


data_training_array=scaler.fit_transform(data_training)



model = load_model('keras_model.h5')


past_100_data = data_training.tail(100)
final_df = pd.concat([past_100_data, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    

x_test,y_test =np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted =y_predicted * scale_factor
y_test = y_test * scale_factor


# final
st.subheader('Prediction vs Orginal')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')  # Select the first time step
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



# Get the latest data point for the next day prediction
latest_data = input_data[-1]
latest_data = latest_data.reshape(1, latest_data.shape[0], 1)  # Reshape to (1, sequence_length, num_features)

# Predict the next day's price
next_day_prediction = model.predict(latest_data)

# Scale the prediction back to the original range
next_day_prediction = next_day_prediction * scale_factor

# Compute the next day's close, open, high, and low values based on the prediction
next_day_close = next_day_prediction[0][0]
next_day_open = df['Close'][-1]  # Use the last known close value as the next day's open
next_day_high = max(next_day_open, next_day_close)
next_day_low = min(next_day_open, next_day_close)

# final
st.subheader('Prediction vs Original (Including Next Day)')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(range(len(y_test), len(y_test) + 2), [y_test[-1], next_day_close], 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
xtick_labels = list(df.index[-len(y_test):]) + [df.index[-1] + pd.Timedelta(days=1)]
xtick_locations = list(df.index[-len(y_test):]) + [df.index[-1] + pd.Timedelta(days=1)]
plt.xticks(xtick_locations, xtick_labels)
plt.legend()
st.pyplot(fig2)



