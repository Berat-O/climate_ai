# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10, 6



# Replace 'your_column1' and 'your_column2' with the actual column names you want to select
columns_to_select = ['DateTime', 'MaxTemp']

# Read the data from 'Istanbul Weather Data.csv' into a DataFrame
dftrain = pd.read_csv('Istanbul Weather Data.csv', usecols=columns_to_select)


dftrain['DateTime'] = pd.to_datetime(dftrain['DateTime'], format="%d.%m.%Y") #convert from string to datetime

indexedDataset = dftrain.set_index(['DateTime'])





# Plot the data
plt.xlabel('Date')
plt.ylabel('Temp')
plt.plot(indexedDataset)

#plt.show()


#Determine rolling statistics
rolmean = indexedDataset.rolling(window=12).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level
rolstd = indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)

#Plot rolling statistics
orig = plt.plot(indexedDataset, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
#plt.show()


#Perform Augmented Dickey–Fuller test:
print('Results of Dickey Fuller Test:')
dftest = adfuller(indexedDataset['MaxTemp'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)

#Estimating trend
indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)
#plt.show()

#The below transformation is required to make series stationary
movingAverage = indexedDataset_logScale.rolling(window=12).mean()
movingSTD = indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage, color='red')
#plt.show()


datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage


#Remove NAN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
print(datasetLogScaleMinusMovingAverage.head(2))


def test_stationarity(timeseries):
    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    #plt.show()
    
    #Perform Dickey–Fuller test:
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries['MaxTemp'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(datasetLogScaleMinusMovingAverage)

exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')
#plt.show()

# crate error
datasetLogScaleMinusExponentialMovingAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
datasetLogScaleMinusExponentialMovingAverage.dropna(inplace=True)

datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)
#plt.show()























# Explore the data
# Uncomment and run the following lines to investigate the dataset
# print(dftrain.columns)
# print(dftrain.dtypes)
# print(dftrain.head(3))
# print(dftrain["Condition"].unique())
# print(dftrain['Condition'].value_counts())
# print(dftrain.info()
# print(dftrain["AvgWind"].nunique())
# print(dftrain[dftrain.Condition == "Fog"])
# print(dftrain.groupby("Condition").get_group("Fog"))
# print(dftrain[dftrain.AvgWind == 4])

# Find missing values in the dataset
# Uncomment and run the following line to see the count of missing values in each column
# print(dftrain.isnull().sum())

# Rename the column 'Condition' to 'Weather Conditions'
# Uncomment and run the following line to rename the column
# dftrain.rename(columns={"Condition": "Weather Conditions"}, inplace=True)

# Calculate the standard deviation of 'AvgPressure'
# Uncomment and run the following line to calculate the standard deviation
# print(dftrain.AvgPressure.std())

# Calculate the variance of 'AvgHumidity'
# Uncomment and run the following line to calculate the variance
# print(dftrain.AvgHumidity.var())

# Plot the count of each weather condition in a horizontal bar chart
# Uncomment the following lines to create a bar chart
# dftrain.Condition.value_counts().plot(kind='barh')
# plt.show()

# Define the columns to be plotted
#plot_cols = ['MaxTemp', 'MinTemp', 'AvgHumidity']

# Plot the selected columns with date_time as the index
#plot_features = dftrain[plot_cols]
#plot_features.index = date_time
#_ = plot_features.plot(subplots=True)

# Plot a subset of the data (first 480 rows)
#plot_features = dftrain[plot_cols][:480]
#plot_features.index = date_time[:480]
#_ = plot_features.plot(subplots=True)

# Display the plots
#plt.show()




