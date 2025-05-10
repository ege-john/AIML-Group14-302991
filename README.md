# Cargo Capacity Prediction 

The purpose of this README file is to give the reader an insight about the project by listing the purpose, methods and results of the project in detail.

## AI&ML Group 14 Team Members

- *Member 1 (“Captain”): Ege John Isik
- *Member 2: Muhammet Emin Albayram
- *Member 3: Adasu Akel

The explanations of the given code and its purpose is given in the following sections.

---

## Description of Project

The purpose of the project is to design a machine learning model to predict the desired data “Cargo Capacity (kg)” in the “Aeropolis.csv” data. The goal is to create a system which predicts the possible cargo capacity in a given situation with many parameters. This model can be used in order to optimize the operations of the cargo flights. During the process the models which are proper for the type of data are used to train the model. The type of the problem is described, and the models are chosen according to this. Once the models are trained, a comparison is made by the code and the best model is chosen. The performance of the model is evaluated, and a result of this evaluation is given to the user. 

The dataset Aeropolis is made of 20 variables such as Cargo Capacity, Air Temperature, Weather Status, Package Type and etc. It is made out of the data of cargo flights made and their environmental factors. The following is the description of the variables in data: 

![Features](Visualization/Dataset%20Features.png)

This data can be used to describe the behavior of cargo capacity in different situations. The given code analyzes the correlation between features of the dataset and creates models for predicting the cargo capacity of a new flight by looking at other features. 

### Output Comparison

The output of the given code is the following comparison:

![Comparison](Visualization/Model%20Comparison%20Table.png)

The output shows us that the R² score of the linear regression model is greater than other models. This means that the accuracy of predictions made by linear regression model is higher than others. Also, RMSE score of linear regression is less than others. Having a smaller RMSE score shows that this model makes less mistakes when predicting the cargo capacity. The results are described in the Results section below with more details.

The "Images" folder included in the repository contains the following illustrations:

- *Heatmap: Showing the correlation between each feature
- *Distribution of Cargo Capacity
- *Distribution of All Features: Showing the skewness
- *Missing Values Illustration
- *Comparison of Evaluation of Models

---

## Description of Models, Features, Algorithms

By inspecting the given data we concluded that this is a regression problem. The aim for the models is to predict cargo capacity in future flights. Cargo capacity feature contains numerical data. The data is continuous and our aim is to find the expected value of this continuous function at given set of other features such as Wind Speed. This is why it is a regression problem. 

The models that are used during the project are **Linear Regression, Random Forest and Support Vector Regression**.

### Linear Regression

Linear regression is a baseline model for regression problems. It is used to assume a linear relationship between other features and cargo capacity. Even though it is a simple model, it is useful for making predictions for continuous numerical values efficiently. It is a model which works fast and efficiently in most cases. 

### Random Forest

It is an ensemble learning model which is working with combining many decision trees to make predictions. The Aeropolis data is made of both numerical and categorical values. This is why random forest is an efficient model to implement on this data because this model can handle data with wide variety of inputs. Also, it can be used for non-linear relationships and it is one of the reasons why we used it for our machine learning model. Even though it can have a higher computational cost compared to different models, it has high accuracy and efficiency. 

### Support Vector Regressor (SVR)

Support Vector Regressor is a kernel model which is useful for non-linear relationships. It uses the technique of increasing the dimension of relationship in order to categorize the data. It can work highly efficiently with complex data. When there are relationships without a clear linear relationship the support vector regressor is one of the most useful models. 

---

## Recreating Project Environment

To recreate the environment used during coding, follow these steps:

1. Ensure that Conda is installed.
   
2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>

3. Create the environment:
    ```bash
    conda env create -f environment.yml
    
4. Activate the environment:
    ```bash
    conda activate <environment-name>

---

## Flowchart Showing The Steps of The Code

![Steps](Visualization/Flowchart%20Showing%20Steps.png)

## Code Inspection

The first part of the code imports the necessary libraries. After that it loads the data with using pandas and samples it. The sample is chosen as 1% in order to keep the training process fast while preventing underfitting. However, the user can adjust the value based on the desire and goals of the analysis. 

In the second section, there is a part which shows a preview of the dataset. 

Then, the explanatory data analysis start. First, the code checks the number of unique values for categorical features. Later, it shows the statistics of each feature. By looking at these the user can see the mean, standard deviation, minimum value and maximum value of a feature in dataset.

Then a visualized explanatory data analysis is done with generating a heatmap, regression plot and distribution plot. 

The code checks the missing values on the dataset and visualizes the result to do a proper analysis. The result shows that almost 10% of all features are missing. This can lead to problems during the training of the models. So, the code should impute the missing values in the following sections.

![MissingValue](Visualization/1-Graph%20of%20Missing%20Values.png)

The heatmap illustrates the correlation between each feature in dataset. The aim is to see how much a change in one feature can affect the other feature. We can easily see that Wind Speed and Cargo Capacity is highly correlated while other parameters have correlations close to zero. This means that a change in Wind Speed can increase or decrease the value of Cargo Capacity. Also, there is a correlation between Cargo Capacity and Air Temperature which is close to zero but still higher than the other correlations. All the other features have correlations with Cargo Capacity close to zero. We can conclude that a change in Air Temperature can have a slight effect on Cargo Capacity, while Wind Speed can have a large effect. 

![Heatmap](Visualization/2-Correlation%20Heatmap.png)

In order to inspect the correlation of Wind Speed and Cargo Capacity more, the code creates a regression plot. By looking at the plot it is easily noticeable that the changes in one of the variables affect the other one. 

![Regression](Visualization/3-Regression%20Plot%20of%20WindSpeed%20vs%20CargoCapacity.png)

Then the distributions of numerical columns are plotted to understand and see the skewness of data. This process is made in order to decide whether the code should impute the missing values with mean or median. 

![Distribution](Visualization/4-Distribution%20of%20Features.png)

By looking at the results, it is obvious that all the numerical data except Cleaning Liquid Usage has skewness near to zero. This means that taking the mean of these features to fill the missing values doesn’t make a bias on the learning process because the graphs of the distribution of features are symmetrical. However, the median imputing is used to fill the missing values of Cleaning Liquid Usage to prevent misinformation. 

This is because the Cleaning Liquid Usage graph is skewed to the left side. If the mean of Cleaning Liquid Usage would be used to impute the missing data there could be a misleading because of skewness. Other than that, the categorical missing values are filled by looking at the most frequent data. After that the data is split into training and test sets. 

The models (Linear Regression, Random Forest and Support Vector Regressor) are defined and trained. Afterwards each model is tested on the test set and their R², RMSE and Execution Time are calculated. 

Later, hyperparameter tuning is made by the code for optimizing Random Forest and SVR. The GridSearchCV is used for random forest optimization. It is used because of its high efficiency in optimization. For the SVR optimization Random Search optimization method is used to prevent long optimizing time. 

Finally, the code gives an output including the R², RMSE and Execution Time of all models and the optimized models so that the user can compare them. Also, a graphical visualization of evaluations is given at the end.


![Evaluation](Visualization/5-Model%20Comparison%20Graphs.png)

---

## Experimental Design

### Main Purpose

The project’s main goal is to evaluate the performance of various machine learning models in forecasting the aeropolis vehicles cargo capacity using operational and climatic characteristics. 

The experiment aims to:

1. Compare the predictive accuracy of different models.
2. Identify the machine learning model that achieves the best balance between computational efficiency and prediction accuracy.
3. Demonstrate the effect of preprocessing techniques (e.g., handling missing values, scaling, encoding) on the models' performance.

This aligns with the project's broader objective of developing a robust predictive pipeline for cargo capacity that could be applied in operational logistics and resource planning for aeropolis vehicles.

### Baselines

In this experiment, we used three machine learning methods as baselines to compare performance:

- *Linear Regression*:
  - A simple, interpretable model that serves as a baseline for regression problems.
  - Provides an efficient approach, making it an ideal baseline to evaluate the improvements on the performance offered by more complex models.

- *Random Forest*:
  - A robust, tree-based ensemble learning method that captures non-linear relationships and interactions between features.
  - Frequently used in regression tasks, providing a strong baseline for comparison against other algorithms.

- *Support Vector Regressor (SVR)*:
  - Included to represent models that can capture complex relationships using kernels.
  - It does prediction with the way of increasing the dimensions of correlations for dealing with non-linear data.

These baselines were chosen because they represent a spectrum of complexity and are widely used for similar predictive modeling tasks, providing a meaningful context for evaluating the project's results.

---

## Evaluation Metrics

We used the following evaluation metrics to measure our models performance. These metrics were chosen because they provide complementary insights into the accuracy and effectiveness of regression models in predicting cargo capacity.

1. *Root Mean Squared Error (RMSE)*:
RMSE was chosen because it measures the average prediction error and penalizes large errors more heavily, which is critical for accurate prediction. It provides a clear measure of how different the predictions are from the actual values. Large deviations in predicted cargo capacity can cause operational errors in resource allocation and logistics planning. RMSE ensures that the model priorities minimizing inaccuracies. Our main goal for this project is to develop a reliable prediction system for cargo capacity, RMSE directly aligns with this objective by focusing on reducing large prediction errors.

2. *R-Squared (R²)*:
R² was chosen because it measures how much of the variability in cargo capacity is explained by the model. It provides an indication of how well the model captures the underlying relationships in the data. Since the project aims to leverage operational and meteorological features for prediction, it helps the model to evaluate how well the features are integrated to explain the target variable.

By using both of these evaluation metrics, we ensured a comprehensive evaluation of model performance. Together, these metrics align with the projects objective to build an accurate and reliable predictive pipeline for cargo capacity, ensuring robust performance across different evaluation criteria.

---

## Results

### Main Findings

The project of Aeropolis dataset gives us an insight into the usage of machine learning models for explanatory data analysis and data prediction. The given code implements the three important machine learning models which are **Linear Regression, Random Forest, and Support Vector Regression**. It compares the accuracies, optimizes the models and gives their score of R² and RMSE for a better comparison. 

The result of the code is the following:

![Comparison](Visualization/Model%20Comparison%20Table.png)

By comparing the R² values we can conclude that linear regression has a higher R² value than the other models. This shows that accuracy of linear regression is higher. Also RMSE of linear regression is less than other models. This is because the difference between actual values and the predictions made by linear regression is less than the other models’ predictions. By comparing the execution times, we can say that **Linear Regression is the fastest working algorithm**. We can conclude that the correlation between variables is mostly linear. In this case even though Random Forest and SVR uses more complex methods, they might not have an advantage here because there is not so much non-linearity. 

As the SVR is a model which works better on smaller datasets, the results show us that it underperformed on this large dataset. However, optimized SVR performs better than not optimized SVR but it takes much longer time.

The increasing of R² score and decreasing in RMSE after optimization of Random Forest and SVR shows that hyperparameter tuning worked well. While SVR model was performing worse than Random Forest before optimization, its R² score and RMSE gave better results than Random Forest after optimization.

Finally, R² scores of all models are around 0.7. This shows that the models can explain almost 70 percent of the variance in the Cargo Capacity. It is a decent result for a machine learning algorithm for data prediction. Also, we can conclude that preprocessing of the data with sampling, imputing missing values and scaling worked well for the aim of the project. 

## Conclusions

In conclusion, we have successfully developed and evaluated a machine learning pipeline for predicting cargo capacity in aeropolis vehicles using meteorogical and operational data. By comparing Linear Regression, Random Forest and Support Vector Regressor (SVR), we found that Linear Regression provided the most accurate predictions, with low error rates and deep explanation ability. The use of preprocessing techniques –such as imputation, scaling and encoding– played a critical role in ensuring the reliability of the predictions. The results demonstrate the potential of machine learning to improve efficiency and operational decision-making in real-world scenarios.

However, there are questions that remain open for further exploration. The models were tested on limited sample of data, and it is unclear how well they generalize to more large and complex datasets. For instance, the models performance might be impacted by external factors not captured in the dataset, such as unforeseen weather changes, which can significantly impact the accuracy of the prediction made by the model. During the optimization of Support Vector Regression model, the usage of gridsearchcv rather than random search may have cause the ignoring of more optimal parameter groups. Although it works slow, using GridSearchCV for SVR optimization in future projects can ensure better results. 

Future work should aim to address current limitations by using real-time data streams and testing on more comprehensive datasets to improve model robustness. Expanding the analysis to include explainable AI techniques could provide further insights into feature importance and model decision-making, laying the foundation for more effective and understandable predictive models.
