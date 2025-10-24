
# 🧠 ANN Challenge: Machine Fault Detection(ML LEAGUE TASK 2)

The goal of this challenge is to detect machine faults in a simulated factory environment using an Artificial Neural Network (ANN).
Given 6 noisy sensor readings per sample, the task is to predict whether a machine is in a faulty (1) or normal (0) state.

## 🧹 Data Preprocessing

### Missing Value Imputation

Forward fill (ffill) used for missing values in Sensor_5.
````markdown
df['Sensor_5'] = df['Sensor_5'].fillna(method='ffill')

````
### Feature Engineering
New features were derived to capture non-linear sensor interactions:
````markdown
def feature_engineering(df):
    df["s1_s3"] = df["Sensor_5"] * df["Sensor_3"]
    df['sensor_min']  = df[['Sensor_1','Sensor_2','Sensor_3','Sensor_4','Sensor_5','Sensor_6']].min(axis=1)
    df['s2_times_s6'] = df['Sensor_2'] * df['Sensor_6']
    df['sensor_mean'] = df[['Sensor_1','Sensor_2','Sensor_3','Sensor_4','Sensor_5','Sensor_6']].mean(axis=1)
    return df
````

### Feature Scaling

Standardized all numeric columns using StandardScaler.


## ⚙️ Model 

The core model is a deep feed-forward neural network, dynamically optimized using Optuna.

#### Base Structure:

• Variable number of dense layers: 5–12

• Each layer:
ReLU activation

• Batch Normalization

• Dropout (0.2–0.6 range)

• Final output layer: Sigmoid activation (binary classification)
````markdown
model = tf.keras.Sequential()
for i in range(n_layers):
    model.add(Dense(num_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))
````


## 🎯 Hyperparameter Optimization (Optuna)

Optuna automatically searched for the best combination of:

• n_layers: 5–12

• n_units: 64–128 per layer

• dropout: 0.2–0.6

• learning_rate: 1e-5–1e-2 (log scale)

• batch_size: [32, 64, 128]

• epochs: [100, 150]

Objective Function: Maximize validation F1-score across 5-fold stratified cross-validation.

