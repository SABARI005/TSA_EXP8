### Developed By: SABARI S
### Register No:212222240085
### Date: 


# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
py
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load your dataset
file_path = '/mnt/data/data_date.csv'  # Use your file path
data = pd.read_csv(file_path)

# Display dataset information
print("Shape of the dataset:", data.shape)
print("First 50 rows of the dataset:")
print(data.head(50))

# Filter data for 'India'
country_data = data[data['Country'] == 'India'].copy()

# Replace 'AQI Value' with the actual column to analyze
plt.plot(country_data['AQI Value'].head(50))
plt.title('First 50 values of the "AQI Value" column')
plt.xlabel('Index')
plt.ylabel('AQI Value')
plt.show()

# Calculate rolling mean with window size 5
rolling_mean_5 = country_data['AQI Value'].rolling(window=5).mean()

print("First 10 values of the rolling mean with window size 5:")
print(rolling_mean_5.head(10))

# Calculate rolling mean with window size 10
rolling_mean_10 = country_data['AQI Value'].rolling(window=10).mean()

# Plot original data and fitted value using rolling mean
plt.plot(country_data['AQI Value'], label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)')
plt.title('Original Data and Fitted Value (Rolling Mean)')
plt.xlabel('Index')
plt.ylabel('AQI Value')
plt.legend()
plt.show()

# AutoRegressive (AR) model with lag order of 13
lag_order = 13
model = AutoReg(country_data['AQI Value'], lags=lag_order)
model_fit = model.fit()

# Plot ACF and PACF
plot_acf(country_data['AQI Value'])
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(country_data['AQI Value'])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Generate predictions
predictions = model_fit.predict(start=lag_order, end=len(country_data)-1)

# Calculate and print Mean Squared Error (MSE)
mse = mean_squared_error(country_data['AQI Value'][lag_order:], predictions)
print('Mean Squared Error (MSE):', mse)

# Plot predictions vs original data
plt.plot(country_data['AQI Value'][lag_order:], label='Original Data')
plt.plot(predictions, label='Predictions', color='red')
plt.title('AR Model Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('AQI Value')
plt.legend()
plt.show()


```

### OUTPUT:

![{02B0DC78-9479-4B53-BC0F-03A8FBFE79AE}](https://github.com/user-attachments/assets/9fe50f7e-539d-4a50-895b-b1d9a1b32c81)


![{33846C93-1493-4A8B-823F-4450794B8CCB}](https://github.com/user-attachments/assets/51115b99-a10f-4434-a281-91fd3043a623)
![{1961C532-4230-4480-8BE1-5C17EE54B9E6}](https://github.com/user-attachments/assets/1cb04d82-b004-4b43-845b-4937c7748e5d)

![{9E0D05BF-5F6E-4A66-8F3E-512DCA30AC04}](https://github.com/user-attachments/assets/76561cf3-9e4f-4423-87c8-a9f436d86eba)



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
