import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_classif

data = pd.read_csv("ElecDeviceRatingPrediction_Milestone2.csv")
new_data = data

ram_categories = ['4 GB', '8 GB', '16 GB', '32 GB']
ssd_categories = ['0 GB', '128 GB', '256 GB', '512 GB', '1024 GB', '2048 GB', '3072 GB']
hdd_categories = ['0 GB', '512 GB', '1024 GB', '2048 GB']
graphic_card_gb_categories = ['0 GB', '2 GB', '4 GB', '6 GB', '8 GB']
warranty_categories = ['No warranty', '1 year', '3 years', '2 years']
generation_categories = ['4th', '7th', '8th', '9th', '10th', '11th', '12th']
# rating_categories = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

label_encoder = LabelEncoder()
scaler = StandardScaler()

ordinal_encoder_ram = OrdinalEncoder(categories=[ram_categories])
ordinal_encoder_ssd = OrdinalEncoder(categories=[ssd_categories])
ordinal_encoder_hdd = OrdinalEncoder(categories=[hdd_categories])
ordinal_encoder_graphic_card_gb = OrdinalEncoder(categories=[graphic_card_gb_categories])
ordinal_encoder_warranty = OrdinalEncoder(categories=[warranty_categories])
ordinal_encoder_generation = OrdinalEncoder(categories=[generation_categories])
# ordinal_encoder_rating = OrdinalEncoder(categories=[rating_categories])
for column in new_data.columns:
    unique_values = new_data[column].unique()
    print(f"Unique values for column '{column}':")
    print(unique_values)
    print()

# Replacing Not Available in generation column
new_data['processor_gnrtn'] = new_data['processor_gnrtn'].replace("Not Available", pd.NA)
mode_value = new_data['processor_gnrtn'].mode()[0]
new_data['processor_gnrtn'] = new_data['processor_gnrtn'].fillna(mode_value)

# ram-gb ssd hdd graphic-card warranty generation --> Ordinal Encoding

new_data['ram_gb'] = ordinal_encoder_ram.fit_transform(new_data[['ram_gb']])
new_data['ssd'] = ordinal_encoder_ssd.fit_transform(new_data[['ssd']])
new_data['hdd'] = ordinal_encoder_hdd.fit_transform(new_data[['hdd']])
new_data['graphic_card_gb'] = ordinal_encoder_graphic_card_gb.fit_transform(new_data[['graphic_card_gb']])
new_data['warranty'] = ordinal_encoder_warranty.fit_transform(new_data[['warranty']])
new_data['processor_gnrtn'] = ordinal_encoder_generation.fit_transform(new_data[['processor_gnrtn']])
# new_data['rating'] = ordinal_encoder_rating.fit_transform(new_data[['rating']])

# From float to int64
new_data['ram_gb'] = new_data['ram_gb'].astype('int64')
new_data['ssd'] = new_data['ssd'].astype('int64')
new_data['hdd'] = new_data['hdd'].astype('int64')
new_data['graphic_card_gb'] = new_data['graphic_card_gb'].astype('int64')
new_data['warranty'] = new_data['warranty'].astype('int64')
new_data['processor_gnrtn'] = new_data['processor_gnrtn'].astype('int64')
# new_data['rating'] = new_data['rating'].astype('int64')

# Touch msoffice weight --> One Hot Encoding
new_data = pd.get_dummies(new_data, columns=['Touchscreen'])
new_data = pd.get_dummies(new_data, columns=['msoffice'])
new_data = pd.get_dummies(new_data, columns=['weight'])
# new_data = pd.get_dummies(new_data, columns=['rating'])

new_data['Touchscreen_No'] = new_data['Touchscreen_No'].map({True: 1, False: 0})
new_data['Touchscreen_Yes'] = new_data['Touchscreen_Yes'].map({True: 1, False: 0})

new_data['msoffice_No'] = new_data['msoffice_No'].map({True: 1, False: 0})
new_data['msoffice_Yes'] = new_data['msoffice_Yes'].map({True: 1, False: 0})

new_data['weight_Casual'] = new_data['weight_Casual'].map({True: 1, False: 0})
new_data['weight_Gaming'] = new_data['weight_Gaming'].map({True: 1, False: 0})
new_data['weight_ThinNlight'] = new_data['weight_ThinNlight'].map({True: 1, False: 0})

# new_data['rating_Bad Rating'] = new_data['rating_Bad Rating'].map({True: 1, False: 0})
# new_data['rating_Good Rating'] = new_data['rating_Good Rating'].map({True: 1, False: 0})
# rating_Bad Rating  rating_Good Rating
new_data.drop(columns=['msoffice_No'], inplace=True)
new_data.drop(columns=['Touchscreen_No'], inplace=True)
# new_data.drop(columns=['rating_Bad Rating'], inplace=True)

# Brand , Processor brand, processor name, ram type, os --> Label Encoding
new_data['brand'] = label_encoder.fit_transform(new_data['brand'])
new_data['processor_brand'] = label_encoder.fit_transform(new_data['processor_brand'])
new_data['processor_name'] = label_encoder.fit_transform(new_data['processor_name'])
new_data['ram_type'] = label_encoder.fit_transform(new_data['ram_type'])
new_data['os'] = label_encoder.fit_transform(new_data['os'])
new_data['rating'] = label_encoder.fit_transform(new_data['rating'])
# new_data['rating'] = new_data['rating'].astype(bool)

# Scaling
x_before_scaling = new_data.drop(columns=['rating'])

scaled_data = scaler.fit_transform(x_before_scaling)
scaled_df = pd.DataFrame(scaled_data, columns=x_before_scaling.columns)

y = new_data['rating']
x_scaled = scaled_df

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.20, shuffle=True, random_state=10)

mutual_info_scores = mutual_info_classif(X_train, y_train)

# Train

k = 5
selector = SelectKBest(mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_train, y_train)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_feature_indices]  # Assuming X_train is a DataFrame

# Print selected features
print("Selected Features Train:")
print(selected_features)



# Test

k = 5
selector_test = SelectKBest(mutual_info_classif, k=k)
X_selected_test = selector_test.fit_transform(X_test, y_test)

# Get the indices of the selected features
selected_feature_indices_test = selector.get_support(indices=True)
selected_features_test = X_train.columns[selected_feature_indices_test]  # Assuming X_train is a DataFrame

# Print selected features
print("Selected Features Train:")
print(selected_features_test)

logistic_model_forward = linear_model.LogisticRegression()
logistic_model_backward = linear_model.LogisticRegression()

logistic_model_forward.fit(X_selected, y_train)
y_forward_train_predicted = logistic_model_forward.predict(X_selected)

y_predict_forward_test = logistic_model_forward.predict(X_selected_test)

classifier = SVC()

classifier.fit(X_selected, y_train)
y_train_predicted = classifier.predict(X_selected)
y_predict_test = classifier.predict(X_selected_test)
# logistic_model_backward.fit(x_after_feature_selection_backward_train, y_train)
# y_backward_train_predicted = logistic_model_backward.predict(x_after_feature_selection_backward_train)
#
# y_predict_backward_test = logistic_model_backward.predict(x_after_feature_selection_backward_test)

print("First Model - Logistic Regression: Using Forward Selection for Feature Selection")

print('Logistic Regression Accuracy Forward Train', metrics.accuracy_score(np.asarray(y_train), y_forward_train_predicted))
print('Logistic Regression Accuracy Forward Test', metrics.accuracy_score(np.asarray(y_test), y_predict_forward_test))

print("First Model - SVC: Using Forward Selection for Feature Selection")

print('SVC Accuracy Forward Train', metrics.accuracy_score(np.asarray(y_train), y_train_predicted))
print('SVC Accuracy Forward Test', metrics.accuracy_score(np.asarray(y_test), y_predict_test))

featureSelectionClassifier = SVC()
scoringFunction = 'accuracy'
selector_forward_train = SequentialFeatureSelector(featureSelectionClassifier, forward=True, k_features='best',
                                                   scoring=scoringFunction, cv=5)
selector_forward_train.fit(X_train, y_train)

selector_backward_train = SequentialFeatureSelector(featureSelectionClassifier, forward=False, k_features='best',
                                                    scoring=scoringFunction, cv=5)
selector_backward_train.fit(X_train, y_train)

selected_features_forward_train = selector_forward_train.k_feature_idx_
selected_features_backward_train = selector_backward_train.k_feature_idx_

x_after_feature_selection_forward_train = pd.DataFrame()
x_after_feature_selection_backward_train = pd.DataFrame()
x_after_feature_selection_forward_test = pd.DataFrame()
x_after_feature_selection_backward_test = pd.DataFrame()

# Train
for column in selected_features_forward_train:
    column_names_forward = x_scaled.columns[column]
    column_name = X_train.columns[column]
    column_values_train = X_train.iloc[:, column]
    column_values_test = X_test.iloc[:, column]
    x_after_feature_selection_forward_train[column_name] = column_values_train
    x_after_feature_selection_forward_test[column_name] = column_values_test

# x_after_feature_selection_forward_test.to_csv('forward.csv', index=False)

for column in selected_features_backward_train:
    column_names_backward = x_scaled.columns[column]
    column_name = X_train.columns[column]
    column_values_train = X_train.iloc[:, column]
    column_values_test = X_test.iloc[:, column]
    x_after_feature_selection_backward_train[column_name] = column_values_train
    x_after_feature_selection_backward_test[column_name] = column_values_test

# SVC - Logistic - Decision Tree - Random Forest - K-Nearest Neighbors - Naive Bayes

logistic_model_forward = linear_model.LogisticRegression()
logistic_model_backward = linear_model.LogisticRegression()

logistic_model_forward.fit(x_after_feature_selection_forward_train, y_train)
y_forward_train_predicted = logistic_model_forward.predict(x_after_feature_selection_forward_train)

y_predict_forward_test = logistic_model_forward.predict(x_after_feature_selection_forward_test)

logistic_model_backward.fit(x_after_feature_selection_backward_train, y_train)
y_backward_train_predicted = logistic_model_backward.predict(x_after_feature_selection_backward_train)

y_predict_backward_test = logistic_model_backward.predict(x_after_feature_selection_backward_test)

print("First Model - Logistic Regression: Using Forward Selection for Feature Selection")

print('Logistic Regression Accuracy Forward Train', metrics.accuracy_score(np.asarray(y_train), y_forward_train_predicted))
print('Logistic Regression Accuracy Forward Test', metrics.accuracy_score(np.asarray(y_test), y_predict_forward_test))


print("\nSecond Model - Logistic Regression: Using Backward Elimination for Feature Selection")

print('Logistic Regression Accuracy Backward Train', metrics.accuracy_score(np.asarray(y_train), y_backward_train_predicted))
print('Logistic Regression Accuracy Backward Test', metrics.accuracy_score(np.asarray(y_test), y_predict_backward_test))

# logistic_model_forward = linear_model.LogisticRegression()
# logistic_model_backward = linear_model.LogisticRegression()
#
# logistic_model_forward.fit(x_after_feature_selection_forward_train, y_train)
# y_forward_train_predicted = logistic_model_forward.predict(x_after_feature_selection_forward_train)
#
# y_predict_forward_test = logistic_model_forward.predict(x_after_feature_selection_forward_test)
#
# logistic_model_backward.fit(x_after_feature_selection_backward_train, y_train)
# y_backward_train_predicted = logistic_model_backward.predict(x_after_feature_selection_backward_train)
#
# y_predict_backward_test = logistic_model_backward.predict(x_after_feature_selection_backward_test)
#
# print("First Model - Logistic Regression: Using Forward Selection for Feature Selection")
#
# print('Logistic Regression Accuracy Forward Train', metrics.accuracy_score(np.asarray(y_train), y_forward_train_predicted))
# print('Logistic Regression Accuracy Forward Test', metrics.accuracy_score(np.asarray(y_test), y_predict_forward_test))
#
#
# print("\nSecond Model - Logistic Regression: Using Backward Elimination for Feature Selection")
#
# print('Logistic Regression Accuracy Backward Train', metrics.accuracy_score(np.asarray(y_train), y_backward_train_predicted))
# print('Logistic Regression Accuracy Backward Test', metrics.accuracy_score(np.asarray(y_test), y_predict_backward_test))
