# Flood Prediction in South Asia: A Machine Learning Approach

# How to Build and Run the Code

To reproduce the results, follow the steps below:
  1. Clone the Repository: Download or clone the repository containing the code and dataset.


Install Dependencies: Ensure Python 3.x is installed on your system. Install all required libraries using the Makefile provided. Run the following command:

 make install
 
   2. This will install dependencies such as:
      
- pandas
- scikit-learn
- seaborn
- matplotlib
- xgboost
- folium

Run the Code: Use the following command to execute the script and generate results:

 make run

3. This will:

- Load the dataset
- Process and visualize the data
- Train models (Logistic Regression, Random Forest, XGBoost)
- Fine-tune the Logistic Regression model
- Generate evaluation metrics and visualizations.

Then either use one of the following commands "make convert" or "make convert-pdf" in order exporting the notebook as an HTML or PDF file

4. Dependencies: The main libraries used in the project are listed in the requirements.txt file. To install them manually, run:

 pip install -r requirements.txt

5. Dataset: Ensure the dataset (flood_risk_dataset_india.csv) is in the same directory as the code. This dataset contains environmental factors such as rainfall, water levels, humidity, land cover types, and flood occurrences.

# Visualizations of Data

To better understand the dataset and relationships between features, the following visualizations were created:

1. Feature Distributions:
- The distributions of Water Level (m) and Rainfall (mm) were plotted, showing their variation with respect to flood occurrences.
- These visualizations provide insights into how these features influence flood risk.

2. Heatmap of Flood Probability:
- A heatmap visualizes the combined effect of rainfall and water level bins on flood probability. It highlights critical ranges where flood risk increases.

3. Flood Occurrence by Land Cover Type:
- A bar chart was generated to explore the relationship between land cover types (e.g., urban, forest, desert) and flood occurrences.

4. Geospatial Visualization:
- An interactive map shows the geographical distribution of flood occurrences across South Asia, with clusters identifying high-risk areas.

# Description of Data Processing and Modeling

1. Data Preprocessing:
- Categorical features (Land Cover and Soil Type) were one-hot encoded to convert them into numerical format.
- To address class imbalance, the dataset was balanced using RandomUnderSampler, ensuring equal representation of flood and no-flood cases.
- Numerical features were standardized using StandardScaler to ensure all variables were on the same scale.
- Recursive Feature Elimination (RFE) was applied to select the top 10 most predictive features for the model, reducing dimensionality.

2. Modeling:
- Three machine learning models were trained: Logistic Regression, Random Forest, and XGBoost.
- Logistic Regression performed best with an accuracy of 50.40% and a ROC-AUC score of 51.08%, making it the final model.
- The Logistic Regression model was fine-tuned using GridSearchCV, optimizing hyperparameters (C for regularization strength and solver type). The best configuration achieved an accuracy of 50.27% and a ROC-AUC score of 51.08%.

3. Reason for Choosing Logistic Regression:
- Despite the low accuracy, Logistic Regression showed the most consistent performance and was interpretable for this problem.
- Fine-tuning focused on improving generalization without overfitting, which is critical for datasets with limited predictive signals.


# Results

The following results summarize the model performance and insights gained:

1. Model Evaluation:
- Logistic Regression: Accuracy = 50.40%, ROC-AUC = 51.08%
- Random Forest: Accuracy = 48.58%, ROC-AUC = 48.94%
- XGBoost: Accuracy = 50.22%, ROC-AUC = 49.70%

2. Key Observations:
- Logistic Regression achieved the highest accuracy and ROC-AUC, making it the final model.
- The dataset’s low accuracy could be attributed to:
    - Limited Predictive Power: Environmental features may not fully capture the complexity of flood occurrence.
    - Data Quality: The dataset may contain noise or insufficient resolution for certain features.
    - Model Assumptions: Logistic Regression assumes linear relationships, which may not entirely align with real-world flood dynamics.
- Why We Chose Logistic Regression:
    - Logistic Regression was selected because it achieved the highest accuracy (50.40%) and ROC-AUC score (51.08%) compared to Random Forest and XGBoost.
    - Logistic Regression is interpretable and suitable for binary classification tasks like this one. It also allows for easier fine-tuning and analysis of feature importance.

3. Visualizations:
- The heatmap revealed that certain ranges of Rainfall and Water Level increase flood risk.
- Geospatial analysis identified specific regions in South Asia prone to flooding, aligning with historical flood patterns.


# Why Accuracy Was Low

1. Dataset Limitations:
- The dataset may lack sufficient predictive power. For instance, some features like rainfall or water level alone may not fully explain flood occurrences.
- Potential noise or inaccuracies in the dataset could be affecting model performance.
- Missing important environmental factors, such as elevation, river discharge, or land slope, may hinder the model’s ability to capture flood patterns.

2. Linear Assumptions of Logistic Regression:
- Logistic Regression assumes a linear relationship between features and the target variable, which may not align with the complex and non-linear nature of flood dynamics.

3. Small Sample Size:
- While undersampling balanced the classes, it also reduced the number of samples available for training, possibly leading to overfitting or underperformance.

# Future Improvements

To improve model performance and better understand flood risks:

1. Enhanced Data Collection:
- Include additional features such as elevation, river flow rates, or satellite imagery for better predictive power.
- Ensure higher resolution and accuracy in collected environmental data.
- Collect additional features like elevation, river discharge, satellite imagery, or seasonal data to capture more predictive signals.
- Increase the dataset size or resolution to reduce noise and improve generalization.
- Collaborate with experts in hydrology or climate science to identify and include key variables affecting flood risk.

2. Modeling Techniques:
- Experiment with non-linear models (e.g., Support Vector Machines or Neural Networks) to capture complex interactions between features.
- Use ensemble methods like Gradient Boosting for improved performance.

3. Temporal Analysis:
- Analyze trends over time by incorporating historical flood data and using time-series forecasting techniques.

4. Explore Non-Linear Models:
- Use algorithms like Support Vector Machines, Gradient Boosting, or Neural Networks to model non-linear relationships between features and flood occurrences.

5. Temporal Analysis:
- Incorporate time-series data to analyze trends and seasonal variations in floods, which could improve predictions.

6. Feature Engineering:
- Create interaction terms (e.g., Rainfall × Water Level) or polynomial features to capture more complex relationships.

# Conclusion
This project demonstrated the application of machine learning to predict flood risks in South Asia. Despite low accuracy, the findings highlight the potential for environmental data to inform flood prediction models. The interactive visualizations provide valuable insights for policymakers and researchers, and the framework can be further improved with additional data and modeling techniques.

______________________________________________________________________________________________________
# Midterm Report:
YOUTUBE LINK: https://www.youtube.com/watch?v=OuV1QngPClg
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
