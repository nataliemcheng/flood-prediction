# Midterm Report:
YOUTUBE LINK: https://youtu.be/3ExyLXjNoZY
# Data Processing:

1. Data Loading and Overview: 
- The dataset contains various features related to environmental and demographic factors in India such as rainfall, temperature, humidity, river discharge, water level, elevation, and historical flood occurrences. The binary target variable is "Flood Occurred" (1 = flood, 0 = no flood).

2. Data Encoding: 
- Categorical features such as “Land Cover” and “Soil Type” were one-hot encoded to convert them into numerical form, allowing the model to process these variables. One category per variable was dropped to avoid multicollinearity, resulting in dummy variables for each categorical option.

3. Data Splitting:
- The processed data was split into training (70%) and testing (30%) sets to evaluate the model's performance on unseen data. By doing so, the model can learn the patterns of the data but also ensure that the model does not overfit and can generalize beyond the training set. 

# Data Modeling Methods Used:

1. Logistic Regression:
- We began by applying logistic regression, a basic classification model suitable for binary outcomes. Logistic regression attempts to find the best fit between the features and the binary target by estimating probabilities. This model estimates the probability of an event by fitting the data to a logistic function, enabling us to distinguish between flood and no-flood scenarios based on various predictor variables. Logistic Regression is particularly beneficial for interpretability, as it provides insight into how each feature impacts the likelihood of flooding in a given area. This approach was chosen initially to provide a baseline for more complex models. 
- After training the logistic regression model on the training set, we evaluated its performance using accuracy, precision, recall, and F1 scores. We also visualized the confusion matrix and analyzed feature importance by examining the coefficients, which indicate how each feature impacts the probability of a flood occurrence.

2. Additional Modeling Explorations/Next Steps:
- Moving forward, attempting other models like Random Forest and Gradient Boosting could potentially improve the accuracy by capturing the more complex relationships within the data. We predict that the accuracy could improve because they do not assume linear relationships between the data like Logistic Regression. 
- We can also consider hyperparameter tuning for models like XGBoost to further explore the complex and non-linear relationships in the data. 


# Preliminary Results:

1. Logistic Regression Model:
- The logistic regression model achieved an accuracy of approximately 50%, which suggests that it was unable to capture clear patterns in the data for predicting flood occurrences or the target variable possibly has little to no linear correlation with the features. Precision and recall scores were also around 0.50, indicating that the model’s performance was close to random.
- The confusion matrix highlights our findings that the model struggled to distinguish between flood and no-flood cases effectively.

2. Feature Importance:
- Analysis of the logistic regression coefficients showed that certain features, such as water level, rainfall, and river discharge, had the most significant impact on predictions. These features align with known flood indicators, suggesting that they are important in modeling flood occurrences.
- This insight provides a basis for potential feature engineering, such as creating interaction terms (e.g., combining rainfall and water level), which could improve the model's predictive power.

- Accuracy: The logistic regression model achieved approximately 50% accuracy, which suggests limited ability to find linear relationships in the data.
- Confusion Matrix Interpretation: Misclassifications between flood and no-flood predictions, which shows we need to improve our model, try new models, and tune hyperparameters.
  
3. Next Steps for Improvement:
The preliminary results indicate that logistic regression alone may be insufficient for this dataset, so further work could include:
- Advanced feature engineering to capture non-linear relationships.
- Trying time-series or neural network models if sequential data or temporal patterns become available.
- Potentially tuning more complex models like XGBoost or LightGBM with refined hyperparameters.

______________________________________________________________________________________________________
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
