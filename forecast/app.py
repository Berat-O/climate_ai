# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
from sklearn.naive_bayes import GaussianNB

# Read the data from 'Istanbul Weather Data.csv' into a DataFrame
dftrain = pd.read_csv('Istanbul Weather Data.csv')

# Convert the 'DateTime' column to a datetime format and set it as the index
date_time = pd.to_datetime(dftrain.pop('DateTime'), format='mixed')

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
plot_cols = ['MaxTemp', 'MinTemp', 'AvgHumidity']

# Plot the selected columns with date_time as the index
plot_features = dftrain[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

# Plot a subset of the data (first 480 rows)
plot_features = dftrain[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)

# Display the plots
plt.show()



