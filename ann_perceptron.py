# Author    Jon-Paul Boyd
# What      Artificial Neural Networks Perceptron Example
# Dataset   Kaggle - Titanic: Machine Learning From Disaster

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
dataset_full = dataset_train.append(dataset_test)


# Impute nulls
dataset_full = dataset_full.fillna({
            'Cabin': 'NoCabin'
             })
dataset_full.fillna(0, inplace=True)
    
# Encoding
label = LabelEncoder()
dataset_full['Gender_Code'] = label.fit_transform(dataset_full['Sex'])
dataset_full.drop(['Sex'], axis = 1, inplace = True)

label = LabelEncoder()
dataset_full['Ticket_Code'] = label.fit_transform(dataset_full['Ticket'])
dataset_full.drop(['Ticket'], axis = 1, inplace = True)

label = LabelEncoder()
dataset_full['Cabin_Code'] = label.fit_transform(dataset_full['Cabin'])
dataset_full.drop(['Cabin'], axis = 1, inplace = True)


# Drop features
dataset_full.drop(['Name'], axis = 1, inplace = True)
dataset_full.drop(['Embarked'], axis = 1, inplace = True)


# Set train and test
X_train_full = dataset_full[:891]
y_train = dataset_train.iloc[:, 1].values
X_train = X_train_full.drop(['Survived'], axis = 1)
X_test_full = dataset_full[891:]
X_test = X_test_full.drop(['Survived'], axis = 1)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
sc_test = StandardScaler()
X_train = sc_train.fit_transform(X_train)
X_test = sc_test.fit_transform(X_test)


# Configure simple network
classifier = Sequential()

# Input layer with 9 inputs, hidden layer with 1 neuron
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu', input_dim = 9))

# Output layer - sigmoid goodf for binary classification
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Binary cross entropy good for binary classification
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# Make prediction (convert to binary)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Output network model image and config
classifier.summary()
classifier.get_config()
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model.png')