
# Importing the required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics
import eeglib
import antropy as ant
import tensorflow as tf
from sklearn import preprocessing, svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras import Input, Model, Sequential
from keras.layers import Dense, Flatten, GRU

# Reading the csv file representing the database

database_path = r'C:\DODDY\PROIECT LICENȚĂ - ANXIETY DETECTION\EEG_DATABASE\EEG_Anxiety_database.csv'

database = pd.read_csv(database_path, header=None, skiprows=1)

database.rename(columns={23040: 'Channel'}, inplace=True)
database.rename(columns={23041: 'SAM_label'}, inplace=True)
database.rename(columns={23042: 'HAM_label'}, inplace=True)

# print(database)

# plotting the distribution of patients according to SAM/HAM tests

data = database.to_numpy()

normal_SAM = normal_HAM = light_SAM = light_HAM = moderate_SAM = moderate_HAM = severe_SAM = severe_HAM = 0

for x in range(322):
    if data[x][23041] == 'NORMAL':
        normal_SAM += 1
    elif data[x][23041] == 'LIGHT_ANXIETY':
        light_SAM += 1
    elif data[x][23041] == 'MODERATE_ANXIETY':
        moderate_SAM += 1
    else:
        severe_SAM += 1

for x in range(322):
    if data[x][23042] == 'NORMAL':
        normal_HAM += 1
    elif data[x][23042] == 'LIGHT_ANXIETY':
        light_HAM += 1
    elif data[x][23042] == 'MODERATE_ANXIETY':
        moderate_HAM += 1
    else:
        severe_HAM += 1

data_dict = {'Normal_SAM': normal_SAM, 'Light_Anxiety_SAM': light_SAM, 'Moderate_Anxiety_SAM': moderate_SAM,
             'Severe_Anxiety_SAM': severe_SAM, 'Normal_HAM': normal_HAM, 'Light_Anxiety_HAM': light_HAM,
             'Moderate_Anxiety_HAM': moderate_HAM, 'Severe_Anxiety_HAM': severe_HAM}

diagnostic_SAM = list(data_dict.keys())[0:4]
no_of_patients_SAM = list(data_dict.values())[0:4]

diagnostic_HAM = list(data_dict.keys())[4:8]
no_of_patients_HAM = list(data_dict.values())[4:8]

plt.figure(figsize=(10, 5))
plt.bar(diagnostic_SAM, no_of_patients_SAM, color='red', width=0.5)
plt.title('Data distribution according to SAM label', fontstyle='italic')
plt.xlabel('Diagnostic', weight='bold')
plt.ylabel('Number of patients', weight='bold')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(diagnostic_HAM, no_of_patients_HAM, color='blue', width=0.5)
plt.title('Data distribution according to HAM label', fontstyle='italic')
plt.xlabel('Diagnostic', weight='bold')
plt.ylabel('Number of patients', weight='bold')
plt.show()

plt.figure(figsize=(10, 5))
X = ['Normal', 'Light Anxiety', 'Moderate Anxiety', 'Severe Anxiety']
SAM_data = [normal_SAM, light_SAM, moderate_SAM, severe_SAM]
HAM_data = [normal_HAM, light_HAM, moderate_HAM, severe_HAM]

X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, SAM_data, 0.4, label='SAM test')
plt.bar(X_axis + 0.2, HAM_data, 0.4, label='HAM test')

plt.xticks(X_axis, X)
plt.title("Data distribution - comparison", fontstyle='italic')
plt.xlabel('Diagnostic', weight='bold')
plt.ylabel('Number of patients', weight='bold')
plt.legend()
plt.show()

# plotting each case of anxiety as a corresponding EEG signal

situations = []

while len(situations) < 4:
    signal = np.random.randint(0, 322)
    x = data[signal][23042]
    if x in situations:
        continue
    else:
        situations.append(x)
        plt.figure()
        y = data[signal][0: 23040]
        plt.plot(y)
        plt.title(data[signal][23040] + '\n' + data[signal][23042])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

# Label encoding

# print(database['SAM_label'].unique())
# print(database['HAM_label'].unique())

label_encoder = preprocessing.LabelEncoder()

database['SAM_label'] = label_encoder.fit_transform(database['SAM_label'])

# MODERATE ANXIETY -> 0, NORMAL -> 1, SEVERE ANXIETY -> 2
# we have no cases of LIGHT ANXIETY

database['HAM_label'] = label_encoder.fit_transform(database['HAM_label'])

# LIGHT ANXIETY -> 0, MODERATE ANXIETY -> 1, NORMAL -> 2, SEVERE ANXIETY -> 3

# print("The database after the coding process is:\n", database)

data = database.to_numpy()

# Feature Extraction

# A) Time domain features

# Media, Standard deviation
# Hjorth parameters -> Activity, Mobility, Complexity
# Petrosian Fractal Dimension

mean_vector = []
activity_vector = []
standard_deviation_vector = []
mobility_vector = []
complexity_vector = []
petrosian_fractal_dimension_vector = []

for x in range(322):

    mean_vector.append(statistics.mean(data[x][0:23040]))
    activity_vector.append(statistics.variance(data[x][0:23040]))
    standard_deviation_vector.append(statistics.stdev(data[x][0:23040]))
    mobility, complexity = ant.hjorth_params(data[x][0:23040])
    mobility_vector.append(mobility)
    complexity_vector.append(complexity)
    petrosian_fractal_dimension_vector.append(eeglib.features.PFD(data[x][0:23040]))

mean_vector = np.array(mean_vector).reshape(-1, 1)
activity_vector = np.array(activity_vector).reshape(-1, 1)
standard_deviation_vector = np.array(standard_deviation_vector).reshape(-1, 1)
mobility_vector = np.array(mobility_vector).reshape(-1, 1)
complexity_vector = np.array(complexity_vector).reshape(-1, 1)
petrosian_fractal_dimension_vector = np.array(petrosian_fractal_dimension_vector).reshape(-1, 1)

dataset_time = np.ones((322, 6))

for x in range(322):

    dataset_time[x][0] *= mean_vector[x]
    dataset_time[x][1] *= activity_vector[x]
    dataset_time[x][2] *= standard_deviation_vector[x]
    dataset_time[x][3] += mobility_vector[x]
    dataset_time[x][4] += complexity_vector[x]
    dataset_time[x][5] += petrosian_fractal_dimension_vector[x]

X_time = dataset_time[:, 0:6]

# B) Frequency domain features

# Band Power

FREQ_BANDS = {"delta": [1, 4],
              "theta": [4, 8],
              "alpha": [8, 13],
              "beta": [13, 32],
              "gamma": [32, 64],
              }

delta_band_vector = []
theta_band_vector = []
alpha_band_vector = []
beta_band_vector = []
gamma_band_vector = []

for x in range(322):

    semnal = data[x][0:23040]
    band_power = eeglib.features.bandPower(semnal, bandsLimits=FREQ_BANDS, freqRes=128, normalize=False)
    delta_band_vector.append(band_power['delta'])
    theta_band_vector.append(band_power['theta'])
    alpha_band_vector.append(band_power['alpha'])
    beta_band_vector.append(band_power['beta'])
    gamma_band_vector.append(band_power['gamma'])

delta_band_vector = np.array(delta_band_vector).reshape(-1, 1)
theta_band_vector = np.array(theta_band_vector).reshape(-1, 1)
alpha_band_vector = np.array(alpha_band_vector).reshape(-1, 1)
beta_band_vector = np.array(beta_band_vector).reshape(-1, 1)
gamma_band_vector = np.array(gamma_band_vector).reshape(-1, 1)

dataset_freq = np.ones((322, 5))

for x in range(322):

    dataset_freq[x][0] *= delta_band_vector[x]
    dataset_freq[x][1] *= theta_band_vector[x]
    dataset_freq[x][2] *= alpha_band_vector[x]
    dataset_freq[x][3] *= beta_band_vector[x]
    dataset_freq[x][4] *= gamma_band_vector[x]

X_freq = dataset_freq[:, 0:5]

# split the dataset into input features and label what we want to predict

X = np.concatenate((X_time, X_freq), axis=1)
Y_SAM = data[:, 23041]
Y_HAM = data[:, 23042]

# data normalization

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

# splitting the data set into a training set and a test set

X_train_SAM, X_test_SAM, Y_train_SAM, Y_test_SAM = train_test_split(X_scale, Y_SAM, test_size=0.2)

X_train_SAM = np.asarray(X_train_SAM).astype(np.float32)
Y_train_SAM = np.asarray(Y_train_SAM).astype(np.float32)
X_test_SAM = np.asarray(X_test_SAM).astype(np.float32)
Y_test_SAM = np.asarray(Y_test_SAM).astype(np.float32)

Y_train_SAM = tf.keras.utils.to_categorical(Y_train_SAM, num_classes=3)
Y_test_SAM = tf.keras.utils.to_categorical(Y_test_SAM, num_classes=3)

# print(X_train_SAM.shape, X_test_SAM.shape, Y_train_SAM.shape, Y_test_SAM.shape)

X_train_HAM, X_test_HAM, Y_train_HAM, Y_test_HAM = train_test_split(X_scale, Y_HAM, test_size=0.2)

X_train_HAM = np.asarray(X_train_HAM).astype(np.float32)
Y_train_HAM = np.asarray(Y_train_HAM).astype(np.float32)
X_test_HAM = np.asarray(X_test_HAM).astype(np.float32)
Y_test_HAM = np.asarray(Y_test_HAM).astype(np.float32)

Y_train_HAM = tf.keras.utils.to_categorical(Y_train_HAM, num_classes=4)
Y_test_HAM = tf.keras.utils.to_categorical(Y_test_HAM, num_classes=4)

# print(X_train_HAM.shape, X_test_HAM.shape, Y_train_HAM.shape, Y_test_HAM.shape)

# Neural network definition and training for SAM and HAM classification


def create_model(x_train, number_of_classes):

    # input layer

    inputs = Input(shape=(x_train.shape[1], 1))

    # hidden layer using LSTM(GRU)

    gru = GRU(256, return_sequences=True)(inputs)

    # flatten gru layer into vector form

    flatten = Flatten()(gru)

    # output layer

    outputs = Dense(number_of_classes, activation='softmax')(flatten)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def train_model(model, x_train, y_train, x_test, y_test):

    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy'],
        metrics=['accuracy']
    )

    history = model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test)

    return history, loss, accuracy


SAM_model = create_model(X_train_SAM, number_of_classes=3)
print(SAM_model.summary())

SAM_history, SAM_loss, SAM_accuracy = train_model(SAM_model, X_train_SAM, Y_train_SAM, X_test_SAM, Y_test_SAM)

print(f"The test loss for the SAM model is: {SAM_loss * 100}%")
print(f"The test accuracy for the SAM model is: {SAM_accuracy * 100}%")
print('\n')

plt.figure(figsize=(10, 5))
plt.plot(SAM_history.history['loss'])
plt.plot(SAM_history.history['val_loss'])
plt.title('Loss of the SAM model')
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='best')
plt.show()
plt.figure(figsize=(10,5))
plt.plot(SAM_history.history['accuracy'])
plt.plot(SAM_history.history['val_accuracy'])
plt.title('SAM model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='best')
plt.show()

y_pred_SAM = SAM_model.predict(X_test_SAM)
y_pred_SAM = np.argmax(y_pred_SAM, axis=1)
y_train_SAM = np.argmax(Y_train_SAM, axis=1)
y_test_SAM = np.argmax(Y_test_SAM, axis=1)

cm = confusion_matrix(y_test_SAM, y_pred_SAM)
clr = classification_report(y_test_SAM, y_pred_SAM)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title("SAM model confusion matrix")
plt.show()

print("The SAM classification report is:\n---------------------------------\n", clr)
print('\n')

HAM_model = create_model(X_train_HAM, number_of_classes=4)
print(HAM_model.summary())

HAM_history, HAM_loss, HAM_accuracy = train_model(HAM_model, X_train_HAM, Y_train_HAM, X_test_HAM, Y_test_HAM)

print(f"The test loss for the HAM model is: {HAM_loss * 100}%")
print(f"The test accuracy for the HAM model is: {HAM_accuracy * 100}%")
print('\n')

plt.figure(figsize=(10, 5))
plt.plot(HAM_history.history['loss'])
plt.plot(HAM_history.history['val_loss'])
plt.title('Loss of the HAM model')
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='best')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(HAM_history.history['accuracy'])
plt.plot(HAM_history.history['val_accuracy'])
plt.title('HAM model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='best')
plt.show()

y_pred_HAM = HAM_model.predict(X_test_HAM)
y_pred_HAM = np.argmax(y_pred_HAM, axis=1)
y_train_HAM = np.argmax(Y_train_HAM, axis=1)
y_test_HAM = np.argmax(Y_test_HAM, axis=1)

cm = confusion_matrix(y_test_HAM, y_pred_HAM)
clr = classification_report(y_test_HAM, y_pred_HAM)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title("HAM model confusion matrix")
plt.show()

print("The HAM classification report is:\n---------------------------------\n", clr)
print('\n')

# Implementation of different classification neural networks for training and testing our dataset

# SVM -> Support Vector Machine

# SAM model

SVM_model_SAM = svm.SVC(kernel='linear')
SVM_model_SAM.fit(X_train_SAM, y_train_SAM)
pred_svm_SAM = SVM_model_SAM.predict(X_test_SAM)
svm_accuracy = accuracy_score(y_test_SAM, pred_svm_SAM)
print(f"The accuracy for the SVM_SAM model is: {svm_accuracy * 100}%\n")

cm = confusion_matrix(y_test_SAM, pred_svm_SAM)
clr = classification_report(y_test_SAM, pred_svm_SAM, zero_division=1)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title("SVM_SAM confusion matrix")
plt.show()
print("The SVM classification report for the SAM model is:\n---------------------------------------------------\n", clr)
print('\n')

# HAM model

SVM_model_HAM = svm.SVC(kernel='linear')
SVM_model_HAM.fit(X_train_HAM, y_train_HAM)
pred_svm_HAM = SVM_model_HAM.predict(X_test_HAM)
svm_accuracy = accuracy_score(y_test_HAM, pred_svm_HAM)
print(f"The accuracy for the SVM_HAM model is: {svm_accuracy * 100}%\n")

cm = confusion_matrix(y_test_HAM, pred_svm_HAM)
clr = classification_report(y_test_HAM, pred_svm_HAM, zero_division=1)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Purples')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title("SVM_HAM model confusion matrix")
plt.show()
print("The SVM classification report for the HAM model is:\n---------------------------------------------------\n", clr)
print('\n')

# DT -> Decision Tree

# SAM Model

dt_model_SAM = DecisionTreeClassifier(class_weight='balanced', max_depth=None)
dt_model_SAM.fit(X_train_SAM, y_train_SAM)
pred_dt_SAM = dt_model_SAM.predict(X_test_SAM)
dt_accuracy = accuracy_score(y_test_SAM, pred_dt_SAM)
print(f"The accuracy for the Decision Tree SAM model is: {dt_accuracy * 100}%\n")

cm = confusion_matrix(y_test_SAM, pred_dt_SAM)
clr = classification_report(y_test_SAM, pred_dt_SAM)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='PuRd')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title("Decision Tree SAM confusion matrix")
plt.show()
print("The Decision Tree classification report for the SAM model is:\n-----------------------------------------\n", clr)
print('\n')

# HAM model

dt_model_HAM = DecisionTreeClassifier(class_weight='balanced', max_depth=None)
dt_model_HAM.fit(X_train_HAM, y_train_HAM)
pred_dt_HAM = dt_model_HAM.predict(X_test_HAM)
dt_accuracy = accuracy_score(y_test_HAM, pred_dt_HAM)
print(f"The accuracy for the Decision Tree HAM model is: {dt_accuracy * 100}%\n")

cm = confusion_matrix(y_test_HAM, pred_dt_HAM)
clr = classification_report(y_test_HAM, pred_dt_HAM)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title("Decision Tree HAM confusion matrix")
plt.show()
print("The Decision Tree classification ratio for the HAM model is:\n-----------------------------------------\n", clr)
print('\n')

# RF -> Random Forest

# SAM model

rf_model_SAM = RandomForestClassifier(class_weight='balanced', random_state=1, max_depth=None, n_estimators=100,
                                      max_features=None)
rf_model_SAM.fit(X_train_SAM, y_train_SAM)
pred_rf_SAM = rf_model_SAM.predict(X_test_SAM)
rf_accuracy = accuracy_score(y_test_SAM, pred_rf_SAM)
print(f"The accuracy for the Random Forest SAM model is: {rf_accuracy * 100}%\n")

cm = confusion_matrix(y_test_SAM, pred_rf_SAM)
clr = classification_report(y_test_SAM, pred_rf_SAM)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title("Random Forest SAM confusion matrix")
plt.show()
print("The Random Forest classification report for the SAM model is:\n-----------------------------------------\n", clr)
print('\n')

# HAM model

rf_model_HAM = RandomForestClassifier(class_weight='balanced', random_state=1, max_depth=None, n_estimators=100,
                                      max_features=None)
rf_model_HAM.fit(X_train_HAM, y_train_HAM)
pred_rf_HAM = rf_model_HAM.predict(X_test_HAM)
rf_accuracy = accuracy_score(y_test_HAM, pred_rf_HAM)
print(f"The accuracy for the Random Forest HAM model is: {rf_accuracy * 100}%\n")

cm = confusion_matrix(y_test_HAM, pred_rf_HAM)
clr = classification_report(y_test_HAM, pred_rf_HAM)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='YlOrRd')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title("Random Forest HAM confusion matrix")
plt.show()
print("The Random Forest classification report for the HAM model is:\n-----------------------------------------\n", clr)
