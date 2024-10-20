# flood-prediction

# Project Proposal:
Flood Prediction

# Goal/Description of Project: 
Our project aims to predict the likelihood of flooding in South Asia over an extended period of time in years. By analyzing environmental factors in the Indian Ocean as it has been significantly impacted by climate change recently (water temperature, temperature, tides, wind direction, wave height, storm probability/intensity), we will develop a model that forecasts whether or not flooding is likely to occur. 

# Data Collection: 
Tides: Data on tidal levels from local government or marine monitoring websites
Wind Direction and Speed: Sourced from meteorological websites or APIs (e.g. OpenWeather)
Wave Height: Data from coastal monitoring stations or NOAA.
Storm Probability/Intensity: Hurricane or storm data from government weather agencies or historical datasets.
Water Temperature: Data from NOAA.
Data Collection Frequency: Since we plan to predict over an extended period of time in years, monthly data collection would give enough of a detailed description of changes in environmental factors.
Missing Values: To handle missing values, if there are a lot of missing values in a certain environmental factor, we will drop the column. We can also take the mean of the features to replace null values or use a K-Nearest Neighbors implementation. 

Collection methods include web scraping, API integration, and leveraging open-source datasets from organizations like NOAA and regional environmental monitoring agencies.
https://www.weather.gov/documentation/services-web-api 

# Modeling Approach:
We plan to model the data using a combination of machine learning techniques:
Logistic Regression: for binary classification (flood/no flood)

Decision Trees: to capture complex interactions between variables such as storm intensity and wave height

We will use accuracy, precision, recall, and F1 scores to measure correct predictions, the percentages of correct positive predictions out of total predicted positives and out of actual positives, and compute the mean of precision and recall respectively for classification models.

Deep Learning (RNNs): if the dataset supports time-series analysis to predict flood patterns based on sequences of weather data

For time-series analysis, we can use metrics like mean absolute error (MAE) and root mean squared error (RMSE) to measure the average magnitude of errors between predicted and actual values and to penalize larger errors and assess our model performance respectively. 

We plan to use numpy, matplotlib, pandas 

# Data Visualization: 
Heatmaps: which are commonly used for weather datasets. Helps for visualizing flood probability across different areas 
Decision Tree: plot the decision tree if we decide to use one
Scatter Plots: to plot the correlation between two variables (ex. wave height vs. water temperature)

# Test plan:
Withhold 30% of the data as a test set (70/30 rule)
Train the model on data collected from previous years (2010-2015, 2015-2020)
Test the model’s performance on new data from subsequent time periods or years to validate its predictive capabilities 

This approach will ensure that the model generalizes well and can make accurate predictions on unseen data
