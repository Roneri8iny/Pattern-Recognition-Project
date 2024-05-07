import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import warnings
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

warnings.filterwarnings("ignore", category=FutureWarning)
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

# categorical_data_df = pd.concat([categorical_data_df, new_data['rating']], axis=1)
# categorical_data_df.to_csv('categorical.csv', index=False)

# Train Test Split --> Numerical
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.20,
                                                    shuffle=True,
                                                    random_state=10)

# Train Test Split --> Categorical X_train_categorical, X_test_categorical, y_train_categorical, y_test_categorical =
# train_test_split(categorical_data_df, y, test_size=0.20, shuffle=True, random_state=10)

# Train Numerical
selected_numerical_columns = ['Price', 'Number of Ratings', 'Number of Reviews']
numerical_data_df_train = X_train[selected_numerical_columns].copy()

# Test Numerical
numerical_data_df_test = X_test[selected_numerical_columns].copy()

# numerical_data_df = pd.concat([numerical_data_df, new_data['rating']], axis=1)
# numerical_data_df.to_csv('numerical.csv', index=False)

selected_categorical_columns = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type',
                                'ssd', 'hdd', 'os', 'graphic_card_gb', 'warranty', 'Touchscreen_Yes',
                                'msoffice_Yes', 'weight_Casual', 'weight_Gaming', 'weight_ThinNlight']
# Train Categorical
categorical_data_df_train = X_train[selected_categorical_columns].copy()

# Test Categorical
categorical_data_df_test = X_test[selected_categorical_columns].copy()

# Categorical vs Categorical --> Use Chi-Squared or Mutual Info
np.random.seed(41)
# # Information Gain
# Train
k = 5
selector = SelectKBest(mutual_info_classif, k=k)
X_selected_train = selector.fit_transform(categorical_data_df_train, y_train)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_features = categorical_data_df_train.columns[selected_feature_indices]  # Assuming X_train is a DataFrame

# Print selected features
print("Selected Features Train:")
print(selected_features)

# Test
k = 5
selector_test = SelectKBest(mutual_info_classif, k=k)
X_selected_test = selector_test.fit_transform(categorical_data_df_test, y_test)

# Get the indices of the selected features
selected_feature_indices_test = selector.get_support(indices=True)
selected_features_test = categorical_data_df_test.columns[selected_feature_indices_test]  # Assuming X_train is a
# DataFrame

# Print selected features
print("Selected Features Train:")
print(selected_features_test)

# logistic_model_forward = linear_model.LogisticRegression()
# logistic_model_backward = linear_model.LogisticRegression()
#
# logistic_model_forward.fit(X_selected_train, y_train)
# y_forward_train_predicted = logistic_model_forward.predict(X_selected_train)
#
# y_predict_forward_test = logistic_model_forward.predict(X_selected_test)
#
# # print("First Model - Logistic Regression: Using Information Gain for Feature Selection")
#
# print('Logistic Regression Accuracy Train',
#       metrics.accuracy_score(np.asarray(y_train), y_forward_train_predicted))
# print('Logistic Regression Accuracy Test',
#       metrics.accuracy_score(np.asarray(y_test), y_predict_forward_test))

# Numerical vs Categorical ---> Use ANOVA or KENDALL's
# ANOVA
# Train
# X_train_numerical, X_test_numerical, y_train_numerical, y_test_numerical
f_values = f_classif(numerical_data_df_train, y_train)[0]

k = 2
selector = SelectKBest(f_classif, k=k)
X_selected_Anova_train = selector.fit_transform(numerical_data_df_train, y_train)

selected_feature_indices_anova_train = selector.get_support(indices=True)
selected_features_anova = numerical_data_df_train.columns[selected_feature_indices_anova_train]

print("Selected Features ANOVA Train:")
print(selected_features_anova)

# Test

f_values = f_classif(numerical_data_df_test, y_test)[0]

k = 2
selector = SelectKBest(f_classif, k=k)
X_selected_Anova_test = selector.fit_transform(numerical_data_df_test, y_test)

selected_feature_indices_anova_test = selector.get_support(indices=True)
selected_features_anova_test = numerical_data_df_test.columns[selected_feature_indices_anova_test]

print("Selected Features ANOVA Test:")
print(selected_features_anova_test)

X_selected_train_df = pd.DataFrame(X_selected_train)
X_selected_train_df.columns = selected_features
X_selected_Anova_train_df = pd.DataFrame(X_selected_Anova_train)
X_selected_Anova_train_df.columns = selected_features_anova
all_selectedFeatures_df_train = pd.concat([X_selected_train_df, X_selected_Anova_train_df], axis=1)

X_selected_test_df = pd.DataFrame(X_selected_test)
X_selected_test_df.columns = selected_features_test
X_selected_Anova_test_df = pd.DataFrame(X_selected_Anova_test)
X_selected_Anova_test_df.columns = selected_features_anova_test
all_selectedFeatures_df_test = pd.concat([X_selected_test_df, X_selected_Anova_test_df], axis=1)

all_selectedFeatures_df_train.to_csv('SelectedTrain.csv', index=False)
all_selectedFeatures_df_test.to_csv('SelectedTest.csv', index=False)

logistic_model = linear_model.LogisticRegression(C=0.1)

logistic_model.fit(all_selectedFeatures_df_train, y_train)
y_forward_train_predicted = logistic_model.predict(all_selectedFeatures_df_train)

y_predict_forward_test = logistic_model.predict(all_selectedFeatures_df_test)

# print("First Model - Logistic Regression: Using Information Gain for Feature Selection")

print('Logistic Regression Accuracy Train',
      metrics.accuracy_score(np.asarray(y_train), y_forward_train_predicted))
print('Logistic Regression Accuracy Test',
      metrics.accuracy_score(np.asarray(y_test), y_predict_forward_test))

classifier = SVC()

classifier.fit(all_selectedFeatures_df_train, y_train)
y_train_predicted = classifier.predict(all_selectedFeatures_df_train)
y_predict_test = classifier.predict(all_selectedFeatures_df_test)
# logistic_model_backward.fit(x_after_feature_selection_backward_train, y_train)
# y_backward_train_predicted = logistic_model_backward.predict(x_after_feature_selection_backward_train)
#
# y_predict_backward_test = logistic_model_backward.predict(x_after_feature_selection_backward_test)

print('SVC Accuracy Train', metrics.accuracy_score(np.asarray(y_train), y_train_predicted))
print('SVC Accuracy Test', metrics.accuracy_score(np.asarray(y_test), y_predict_test))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10)

dt.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_dt = dt.predict(all_selectedFeatures_df_train)
y_test_predict_dt = dt.predict(all_selectedFeatures_df_test)

train_accuracy_dt = accuracy_score(y_train, y_train_predict_dt)
test_accuracy_dt = accuracy_score(y_test, y_test_predict_dt)

print("Train Accuracy DT:", train_accuracy_dt)
print("Test Accuracy DT:", test_accuracy_dt)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_rf = rf.predict(all_selectedFeatures_df_train)
y_test_predict_rf = rf.predict(all_selectedFeatures_df_test)

train_accuracy_rf = accuracy_score(y_train, y_train_predict_rf)
test_accuracy_rf = accuracy_score(y_test, y_test_predict_rf)

print("Train Accuracy Random Forest:", train_accuracy_rf)
print("Test Accuracy Random Forest:", test_accuracy_rf)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_knn = knn.predict(all_selectedFeatures_df_train)
y_test_predict_knn = knn.predict(all_selectedFeatures_df_test)

train_accuracy_knn = accuracy_score(y_train, y_train_predict_knn)
test_accuracy_knn = accuracy_score(y_test, y_test_predict_knn)

print("Train Accuracy KNN:", train_accuracy_knn)
print("Test Accuracy KNN:", test_accuracy_knn)

# XGBOOST
xgboost = xgb.XGBClassifier()

xgboost.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_xgb = xgboost.predict(all_selectedFeatures_df_train)
y_test_predict_xgb = xgboost.predict(all_selectedFeatures_df_test)

train_accuracy_xgb = accuracy_score(y_train, y_train_predict_xgb)
test_accuracy_xgb = accuracy_score(y_test, y_test_predict_xgb)

print("Train Accuracy XGBOOST:", train_accuracy_xgb)
print("Test Accuracy XGBOOST:", test_accuracy_xgb)

# AdaBoost

ada = AdaBoostClassifier(n_estimators=250, random_state=42)

ada.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_ada = ada.predict(all_selectedFeatures_df_train)
y_test_predict_ada = ada.predict(all_selectedFeatures_df_test)

train_accuracy_ada = accuracy_score(y_train, y_train_predict_ada)
test_accuracy_ada = accuracy_score(y_test, y_test_predict_ada)

print("Train Accuracy AdaBoost:", train_accuracy_ada)
print("Test Accuracy AdaBoost:", test_accuracy_ada)

# Voting
# Adaboost + XGBoost --> Done , AdaBoost + Decision Tree --> Done, XGBOOST + Decision Tree ,
# Decision Tree + SVC --> Done, Decision Tree + Logistic Regression --> Done , Decision Tree + KNN
# base_learners_voting = [
#     ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
#     ('xgb', xgb.XGBClassifier())
# ]

# base_learners_voting = [
#     ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
#     ('dt', DecisionTreeClassifier())
# ]

base_learners_voting = [
    ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ('xgb', xgb.XGBClassifier()),
    ('dt', DecisionTreeClassifier())
]

votingModel = VotingClassifier(estimators=base_learners_voting, voting='hard')

votingModel.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_voting = votingModel.predict(all_selectedFeatures_df_train)
y_test_predict_voting = votingModel.predict(all_selectedFeatures_df_test)

train_accuracy_voting = accuracy_score(y_train, y_train_predict_voting)
test_accuracy_voting = accuracy_score(y_test, y_test_predict_voting)

print(f"Train Accuracy Voting with :", train_accuracy_voting)
print(f"Test Accuracy Voting :", test_accuracy_voting)

# Stacking

base_learners_stacking = [
    # ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ('xgb', xgb.XGBClassifier()),
    ('dt', DecisionTreeClassifier())
    # ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

meta_learner = AdaBoostClassifier(n_estimators=50, random_state=42)

stackingModel = StackingClassifier(estimators=base_learners_stacking, final_estimator=meta_learner)

stackingModel.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_stacking = stackingModel.predict(all_selectedFeatures_df_train)
y_test_predict_stacking = stackingModel.predict(all_selectedFeatures_df_test)

train_accuracy_stacking = accuracy_score(y_train, y_train_predict_stacking)
test_accuracy_stacking = accuracy_score(y_test, y_test_predict_stacking)

print("Train Accuracy Stacking:", train_accuracy_stacking)
print("Test Accuracy Stacking:", test_accuracy_stacking)
