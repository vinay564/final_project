# importing lib
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import warnings
from pickle import dump
import matplotlib
matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import encodeDatasets, MyModel, evaluation

warnings.filterwarnings('ignore')

# loading dataset
dataset = pd.read_csv('train1.csv')

# displaying top-5 rows and properties of the dataset
print("Top-5 rows of the dataset")
display(dataset.head())

row, col = dataset.shape
print(f"\nThe number of rows and columns in the dataset: {row} and {col}")

print('\nDescriptive analysis of the dataset')
display(dataset.describe())

print(f'\nColumns in the datasets are : {list(dataset.columns)}')

# preprocessing datasets
# checking for the missing values
preprocessDataframe = dataset.isnull().sum(axis=0).reset_index()
preprocessDataframe.columns = ['column', 'count']
preprocessDataframe = preprocessDataframe.loc[preprocessDataframe['count'] > 0].sort_values(by='count')
print('\nDisplaying columns with number of count')
display(preprocessDataframe)

# adding column price
dataset['price'] = np.exp(dataset['log_price'])

# dropping unwanted columns
col = ['id', 'neighbourhood', 'name', 'description', 'thumbnail_url', 'amenities', 'first_review',
       'last_review', 'host_since', 'host_has_profile_pic', 'latitude', 'longitude', 'instant_bookable']
dataset.drop(col, axis=1, inplace=True)

# filling dataframe
dataset.update(dataset[['bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'zipcode', 'host_identity_verified',
                        'host_response_rate']].fillna(0))
dataset.loc[dataset.zipcode == ' ', 'zipcode'] = 0
dataset['zipcode'] = dataset.zipcode.apply(lambda r: str(r).replace('-', '').replace('\r', '').replace('\n', ''))
dataset['zipcode'] = dataset.zipcode.apply(lambda r: ''.join(re.findall(r'\d+', str(r))))

dataset['host_response_rate'] = dataset.host_response_rate.apply(lambda r: int(str(r).strip('%')))
dataset['host_identity_verified'] = np.where(dataset.host_identity_verified == 't', 1, 0)

# checking for the missing values post preprocessing
print('\nDisplaying columns with number of count post preprocessing')
display(dataset.isna().any())

# data visualization
# plotting price in the cities
sns.barplot(x='city', y='log_price', data=dataset)
plt.title('Price of the property with City')
plt.show()

# plotting histogram of the house
dataset[['accommodates', 'bathrooms', 'bedrooms', 'beds']].hist(figsize=(8, 6))
plt.suptitle('Histogram distribution of the House attributes')
plt.show()

# plotting total count for each category
dataset['property_type'].value_counts().nlargest(5).plot(kind='bar')
plt.title('property_type')
plt.xticks(rotation=0)
plt.show()

dataset['room_type'].value_counts().plot(kind='bar')
plt.title('room_type')
plt.xticks(rotation=0)
plt.show()

dataset['review_scores_rating'].value_counts().nlargest(5).plot(kind='bar')
plt.title('review_scores_rating')
plt.xticks(rotation=0)
plt.show()

# correlation of the dataset
sns.heatmap(dataset.pivot_table(values='log_price', index='property_type', columns='city'))
plt.show()

sns.heatmap(dataset.pivot_table(values='log_price', index='property_type', columns='accommodates'))
plt.show()

# saving the preprocessed dataset
dataset.to_csv('preprocessedDataset.csv', index=False)

# label encoding the dataset
cols = ['city', 'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'host_response_rate']
encodedData = encodeDatasets(dataset, cols)

print('Encoded dataset')
display(encodedData.head())

# separating features
X = encodedData.loc[:, 'property_type':'beds']
print(X.columns)
x = np.asarray(X).astype(np.float32)
y = encodedData['log_price']

# Scaling
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=list(X.columns))
dump(scaler, open(os.path.join('model', 'scaler.pkl'), 'wb'))

# printing input and output features
print('input and output features are')
display(X.head())
display(y.head())

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)

print(f"The shapes of the train dataset is {X_train.shape}")
print(f"The shapes of the test dataset is {X_test.shape}")

# defining the model
RFR = MyModel(name="RFR")
ridge = MyModel(name="ridge")
KNN = MyModel(name="KNN")

# Training the model - Random forest regressor
print("Training the model - Random forest regressor")
RFR.fit(X_train, y_train)

# Training the model - Ridge regressor
print("Training the model - Ridge regressor")
ridge.fit(X_train, y_train)

# Training the model - KNN regressor
print("Training the model - KNN regressor")
KNN.fit(X_train, y_train)

# saving model
if not os.path.isdir('model'):
    os.mkdir(os.path.join('model'))

pickle.dump(RFR, open(os.path.join('model', 'RFR_AirbnbModel.model'), 'wb'))
pickle.dump(ridge, open(os.path.join('model', 'ridge_AirbnbModel.model'), 'wb'))
pickle.dump(KNN, open(os.path.join('model', 'KNN_AirbnbModel.model'), 'wb'))

# evaluating model
evaluation(RFR, X_test, y_test)

evaluation(ridge, X_test, y_test)

evaluation(KNN, X_test, y_test)
