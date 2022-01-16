
# Diabetes Prediction

This is a Machine Learning model trained to predict whether or not a patient has diabetes based some diagnostic measurements.
The dataset used is from NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases).

#### Exploratory Data Analysis
Tableau was used to analyze the dataset to get some insights.
Some statistical analyses were also done to confirm the distribution of the data, to check for outliers and to check correlations among the variables.
A snapshot of the analysis dashboard is shown below:

#### Data Preprocessing & Feature Engineering
Some key features and variables had null values where were treated with different methods, including filling the null values with the mean/mode of the distribution as appropriate.
No data was deleted on account of missing values.
Random Forest classifier was used to extract the feature imporatnce of all the predictor variables.
Top 5 features were eventually used for training the model.



#### Model Training & Evaluation
Various classification models, including Logistic Regression, Random Forest Classifier, SVM and KNN classifiers were used to train the data with minimal parameter tuning.
Performances of the various models were evaluated.
KNN model gave the best performance with 76% accuracy and a AUC score of 83%.

#### Model Deployment
The trained KNN model was deploy on Heroku platform using Flask as the web framework to serve the web page to allow input of test data, and for the model to display predicted result.

#### Improvements
* Model performance can be enhanced using more training data.
* Quality of dataset used is also key to model performance - dataset with fewer nulls.
* Boosted models can also enhance performance.
## Features

- Data analysis was done using both Tableau and Jupyter Notebook.
- Python - Numpy, Pandas, and Sklearn - was used for the Machine Learning project.
- Flask, a Python web framework was used to serve the web page.
- HTML, CSS, Jinja2 template were also used for the web page

## Authors

- [@PrinceDuru](https://github.com/PrinceDuru)


## Demo
<p><img src="https://github.com/PrinceDuru/KNN_Diabetes_Prediction/blob/master/Demo.gif" /></p>


## Run Locally

Clone the project

```bash
  git clone https://github.com/PrinceDuru/KNN_Diabetes_Prediction
```



