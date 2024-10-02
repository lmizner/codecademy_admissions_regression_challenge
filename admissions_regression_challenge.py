import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

# Load Dataset
admissions_data = pd.read_csv('admissions_data.csv')
print(admissions_data.head())
print(admissions_data.describe())

# Split data into labels and features with "Chance of Admit" as label
labels = admissions_data.iloc[:, -1]
features = admissions_data.iloc[:, 0:-1]

print(labels.shape)
print(features.shape)

# Split into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 42)

# Standardize numerical features
numerical_features = features.select_dtypes(include = ['float', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder = 'passthrough')

features_trained_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

# Build Neural Network model
model = Sequential()
model.add(layers.InputLayer(input_shape = (features_trained_scaled.shape[1],)))
model.add(layers.Dense(64, activation = 'swish'))
model.add(layers.Dropout(0.1)) # adding dropout layer to mitigate overfitting
model.add(layers.Dense(32, activation = 'swish'))
model.add(layers.Dropout(0.1)) # adding dropout layer to mitigate overfitting
model.add(layers.Dense(16, activation = 'swish'))
model.add(layers.Dense(1))

print(model.summary())

# Initialize optimizer and compile the model
# optimizer = RMSprop(learning_rate = 0.001)
# optimizer = Adam(learning_rate = 0.00075) #(R2_score = 83%)
optimizer = Adagrad(learning_rate = 0.01)  #(R2_score = 84%)
model.compile(loss = 'mse', metrics = ['mae'], optimizer = optimizer)

# Add early stopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)

# Fit and evaluate model
history = model.fit(features_trained_scaled, labels_train, epochs = 100, batch_size = 16, validation_split = 0.2, verbose = 1)
mse_result, mae_result = model.evaluate(features_test_scaled, labels_test, verbose = 0, callbacks = [early_stopping])
print(f'MSE: {mse_result}, MAE: {mae_result}')


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

  # Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

# used to keep plots from overlapping each other  
fig.tight_layout()
fig.savefig('static/images/my_plots.png')

predicted_values = model.predict(features_test_scaled) 
print('R^2 score:', r2_score(labels_test, predicted_values))