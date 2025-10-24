
# TASK 2 REGRESSION 
You are given a very simple dataframe with 3 variables w, x and y. You need to predict y, given w and x.

## 🚀 NOTEBOOK OVERVIEW
The notebook performs a complete end-to-end workflow:

1. Dataset loading and inspection

2. Exploratory Data Analysis (EDA) and visualization

3. Model training and evaluation

4. Performance Evaluation
 
## 📊 DATASET DESCRIPTION

This Dataset contains: 
• x: 1st input feature, continious 

• w: 2nd input feature, contains 10 unique value

• y: output

## 🔍 EXPLORATORY DATA ANALYSIS (EDA)

This is the main step in this task. it reveals so much about the dataset. i've plotted all points wrt y and it didnt make any sense. so then i've separated all the unique values of w and then plotted the graph between x and y for all the unique values. 

and after plotting for all unique values of w, you can clearly see that all the graphs resembles a sine or cosine wave. 
after that, i tried to do some feature engineering by combining w and x (w*x) and then plotted the graph and it reveals another great feature. the solution model was very clear after this. 

## 🧠 MODEL DEVELOPMENT

The notebook builds a regression model to predict y using the features w and x.

### Steps:

1. Data preprocessing

• Splitting into training and test sets

2. Model training

• Implemented a regression algorithm

• i've fitted a sine wave

3. Evaluation

• Metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), or R² Score

## 🌟 FUTURE IMPROVEMENTS

i wanted to implement a PINN model in this and compare the result with the regression one, i couldn't do it because of time :(

## 🧑‍💻 AUTHOR

Amey Bhagat

📧 amey.241ds009@gmail.com




