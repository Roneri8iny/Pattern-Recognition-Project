import pickle
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
data = pd.read_csv("G:\\Pattern Recognition\\Pattern-Recognition-Project\Milestone 2\\ElecDeviceRatingPrediction_Milestone2.csv")
new_data = data

ram_categories = ['4 GB', '8 GB', '16 GB', '32 GB']
ssd_categories = ['0 GB', '128 GB', '256 GB', '512 GB', '1024 GB', '2048 GB', '3072 GB']
hdd_categories = ['0 GB', '512 GB', '1024 GB', '2048 GB']
graphic_card_gb_categories = ['0 GB', '2 GB', '4 GB', '6 GB', '8 GB']
warranty_categories = ['No warranty', '1 year', '3 years', '2 years']
generation_categories = ['4th', '7th', '8th', '9th', '10th', '11th', '12th']
# rating_categories = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

label_encoder = LabelEncoder()  # --> validation

label_encoder_brand = LabelEncoder()
label_encoder_brand.handle_unknown = 'use_encoded_value'
label_encoder_brand.unknown_value = -1

label_encoder_processor_brand = LabelEncoder()
label_encoder_processor_brand.handle_unknown = 'use_encoded_value'
label_encoder_processor_brand.unknown_value = -1

label_encoder_processor_name = LabelEncoder()
label_encoder_processor_name.handle_unknown = 'use_encoded_value'
label_encoder_processor_name.unknown_value = -1

label_encoder_ram_type = LabelEncoder()
label_encoder_ram_type.handle_unknown = 'use_encoded_value'
label_encoder_ram_type.unknown_value = -1

label_encoder_os = LabelEncoder()
label_encoder_os.handle_unknown = 'use_encoded_value'
label_encoder_os.unknown_value = -1

label_encoder_rating = LabelEncoder()
label_encoder_rating.handle_unknown = 'use_encoded_value'
label_encoder_rating.unknown_value = -1

# Column name --> label encoder
# X_train['brand'] = label_encoder.fit_transform(X_train['brand'])
# X_train['processor_brand'] = label_encoder.fit_transform(X_train['processor_brand'])
# X_train['processor_name'] = label_encoder.fit_transform(X_train['processor_name'])
# X_train['ram_type'] = label_encoder.fit_transform(X_train['ram_type'])
# X_train['os'] = label_encoder.fit_transform(X_train['os'])
labelEncodersTrainDict = {'brand': label_encoder_brand, 'processor_brand': label_encoder_processor_brand,
                          'processor_name': label_encoder_processor_name, 'ram_type': label_encoder_ram_type,
                          'os': label_encoder_os}


scaler = StandardScaler()


ordinal_encoder_ram = OrdinalEncoder(categories=[ram_categories], handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_ssd = OrdinalEncoder(categories=[ssd_categories], handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_hdd = OrdinalEncoder(categories=[hdd_categories], handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_graphic_card_gb = OrdinalEncoder(categories=[graphic_card_gb_categories],
                                                 handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_warranty = OrdinalEncoder(categories=[warranty_categories], handle_unknown='use_encoded_value',
                                          unknown_value=-1)
ordinal_encoder_generation = OrdinalEncoder(categories=[generation_categories], handle_unknown='use_encoded_value',
                                            unknown_value=-1)


y = pd.DataFrame()
y['rating'] = new_data['rating']
X = new_data.drop(columns=['rating'])
# Train Test Split --> Numerical
X_train_initial, X_test_forbidden, y_train_initial, y_test_forbidden = train_test_split(X, y, test_size=0.20,
                                                                                        shuffle=True,
                                                                                        random_state=10)

X_train, X_validation, y_train, y_validation = train_test_split(X_train_initial, y_train_initial, test_size=0.20,
                                                                shuffle=True,
                                                                random_state=10)

X_test_forbidden['rating'] = y_test_forbidden['rating']
X_test_forbidden.to_csv('testData.csv', index=False)


# Price,  Number of Ratings , Number of Reviews ---> Numerical columns ----> mean
Price_Mean = X_train['Price'].mean()
Number_of_Ratings_mean = X_train['Number of Ratings'].mean()
Number_of_Reviews_mean = X_train['Number of Reviews'].mean()

brand_mode = X_train['brand'].mode()
processor_brand_mode = X_train['processor_brand'].mode()
processor_name_mode = X_train['processor_name'].mode()
processor_gnrtn_mode = X_train['processor_gnrtn'].mode()
ram_gb_mode = X_train['ram_gb'].mode()
ram_type_mode = X_train['ram_type'].mode()
ssd_mode = X_train['ssd'].mode()
hdd_mode = X_train['hdd'].mode()
os_mode = X_train['os'].mode()
graphic_card_gb_mode = X_train['graphic_card_gb'].mode()
weight_mode = X_train['weight'].mode()
warranty_mode = X_train['warranty'].mode()
Touchscreen_mode = X_train['Touchscreen'].mode()
msoffice_mode = X_train['msoffice'].mode()

with open('CMean.pkl', 'wb') as f:
    pickle.dump(Price_Mean, f)
    pickle.dump(Number_of_Ratings_mean, f)
    pickle.dump(Number_of_Reviews_mean, f)

with open('CMode.pkl', 'wb') as f:
    pickle.dump(brand_mode, f)
    pickle.dump(processor_brand_mode, f)
    pickle.dump(processor_name_mode, f)
    pickle.dump(processor_gnrtn_mode, f)
    pickle.dump(ram_gb_mode, f)
    pickle.dump(ram_type_mode, f)
    pickle.dump(ssd_mode, f)
    pickle.dump(hdd_mode, f)
    pickle.dump(os_mode, f)
    pickle.dump(graphic_card_gb_mode, f)
    pickle.dump(weight_mode, f)
    pickle.dump(warranty_mode, f)
    pickle.dump(Touchscreen_mode, f)
    pickle.dump(msoffice_mode, f)


# ordinal_encoder_rating = OrdinalEncoder(categories=[rating_categories])
# for column in new_data.columns:
#     unique_values = new_data[column].unique()
#     print(f"Unique values for column '{column}':")
#     print(unique_values)
#     print()

# Train
# Replacing Not Available in generation column
X_train['processor_gnrtn'] = X_train['processor_gnrtn'].replace("Not Available", pd.NA)
mode_value = X_train['processor_gnrtn'].mode()[0]
X_train['processor_gnrtn'] = X_train['processor_gnrtn'].fillna(mode_value)

# ram-gb ssd hdd graphic-card warranty generation --> Ordinal Encoding

X_train['ram_gb'] = ordinal_encoder_ram.fit_transform(X_train[['ram_gb']])
X_train['ssd'] = ordinal_encoder_ssd.fit_transform(X_train[['ssd']])
X_train['hdd'] = ordinal_encoder_hdd.fit_transform(X_train[['hdd']])
X_train['graphic_card_gb'] = ordinal_encoder_graphic_card_gb.fit_transform(X_train[['graphic_card_gb']])
X_train['warranty'] = ordinal_encoder_warranty.fit_transform(X_train[['warranty']])
X_train['processor_gnrtn'] = ordinal_encoder_generation.fit_transform(X_train[['processor_gnrtn']])
# new_data['rating'] = ordinal_encoder_rating.fit_transform(new_data[['rating']])


with open('Cordinal_encoding.pkl', 'wb') as f:
    pickle.dump(ordinal_encoder_ram, f)
    pickle.dump(ordinal_encoder_ssd, f)
    pickle.dump(ordinal_encoder_hdd, f)
    pickle.dump(ordinal_encoder_graphic_card_gb, f)
    pickle.dump(ordinal_encoder_warranty, f)
    pickle.dump(ordinal_encoder_generation, f)


# From float to int64
X_train['ram_gb'] = X_train['ram_gb'].astype('int64')
X_train['ssd'] = X_train['ssd'].astype('int64')
X_train['hdd'] = X_train['hdd'].astype('int64')
X_train['graphic_card_gb'] = X_train['graphic_card_gb'].astype('int64')
X_train['warranty'] = X_train['warranty'].astype('int64')
X_train['processor_gnrtn'] = X_train['processor_gnrtn'].astype('int64')
# new_data['rating'] = new_data['rating'].astype('int64')

# Touch msoffice weight --> One Hot Encoding
X_train = pd.get_dummies(X_train, columns=['Touchscreen'])
X_train = pd.get_dummies(X_train, columns=['msoffice'])
X_train = pd.get_dummies(X_train, columns=['weight'])
# new_data = pd.get_dummies(new_data, columns=['rating'])

X_train['Touchscreen_No'] = X_train['Touchscreen_No'].map({True: 1, False: 0})
X_train['Touchscreen_Yes'] = X_train['Touchscreen_Yes'].map({True: 1, False: 0})

X_train['msoffice_No'] = X_train['msoffice_No'].map({True: 1, False: 0})
X_train['msoffice_Yes'] = X_train['msoffice_Yes'].map({True: 1, False: 0})

X_train['weight_Casual'] = X_train['weight_Casual'].map({True: 1, False: 0})
X_train['weight_Gaming'] = X_train['weight_Gaming'].map({True: 1, False: 0})
X_train['weight_ThinNlight'] = X_train['weight_ThinNlight'].map({True: 1, False: 0})

# Validation
X_validation['processor_gnrtn'] = X_validation['processor_gnrtn'].replace("Not Available", pd.NA)
mode_value = X_validation['processor_gnrtn'].mode()[0]
X_validation['processor_gnrtn'] = X_validation['processor_gnrtn'].fillna(mode_value)

# ram-gb ssd hdd graphic-card warranty generation --> Ordinal Encoding

X_validation['ram_gb'] = ordinal_encoder_ram.transform(X_validation[['ram_gb']])
X_validation['ssd'] = ordinal_encoder_ssd.transform(X_validation[['ssd']])
X_validation['hdd'] = ordinal_encoder_hdd.transform(X_validation[['hdd']])
X_validation['graphic_card_gb'] = ordinal_encoder_graphic_card_gb.transform(X_validation[['graphic_card_gb']])
X_validation['warranty'] = ordinal_encoder_warranty.transform(X_validation[['warranty']])
X_validation['processor_gnrtn'] = ordinal_encoder_generation.transform(X_validation[['processor_gnrtn']])
# new_data['rating'] = ordinal_encoder_rating.fit_transform(new_data[['rating']])



# From float to int64
X_validation['ram_gb'] = X_validation['ram_gb'].astype('int64')
X_validation['ssd'] = X_validation['ssd'].astype('int64')
X_validation['hdd'] = X_validation['hdd'].astype('int64')
X_validation['graphic_card_gb'] = X_validation['graphic_card_gb'].astype('int64')
X_validation['warranty'] = X_validation['warranty'].astype('int64')
X_validation['processor_gnrtn'] = X_validation['processor_gnrtn'].astype('int64')
# new_data['rating'] = new_data['rating'].astype('int64')

# Touch msoffice weight --> One Hot Encoding
X_validation = pd.get_dummies(X_validation, columns=['Touchscreen'])
X_validation = pd.get_dummies(X_validation, columns=['msoffice'])
X_validation = pd.get_dummies(X_validation, columns=['weight'])
# new_data = pd.get_dummies(new_data, columns=['rating'])

X_validation['Touchscreen_No'] = X_validation['Touchscreen_No'].map({True: 1, False: 0})
X_validation['Touchscreen_Yes'] = X_validation['Touchscreen_Yes'].map({True: 1, False: 0})

X_validation['msoffice_No'] = X_validation['msoffice_No'].map({True: 1, False: 0})
X_validation['msoffice_Yes'] = X_validation['msoffice_Yes'].map({True: 1, False: 0})

X_validation['weight_Casual'] = X_validation['weight_Casual'].map({True: 1, False: 0})
X_validation['weight_Gaming'] = X_validation['weight_Gaming'].map({True: 1, False: 0})
X_validation['weight_ThinNlight'] = X_validation['weight_ThinNlight'].map({True: 1, False: 0})

# new_data['rating_Bad Rating'] = new_data['rating_Bad Rating'].map({True: 1, False: 0})
# new_data['rating_Good Rating'] = new_data['rating_Good Rating'].map({True: 1, False: 0})
# rating_Bad Rating  rating_Good Rating
# new_data.drop(columns=['msoffice_No'], inplace=True)
# new_data.drop(columns=['Touchscreen_No'], inplace=True)
# new_data.drop(columns=['rating_Bad Rating'], inplace=True)
# 0X_train.to_csv('New Encoding.csv', index=False)
# Brand , Processor brand, processor name, ram type, os --> Label Encoding
# X_train['brand'] = label_encoder.fit_transform(X_train['brand'])
# X_train['processor_brand'] = label_encoder.fit_transform(X_train['processor_brand'])
# X_train['processor_name'] = label_encoder.fit_transform(X_train['processor_name'])
# X_train['ram_type'] = label_encoder.fit_transform(X_train['ram_type'])
# X_train['os'] = label_encoder.fit_transform(X_train['os'])
# X_train['rating'] = label_encoder.fit_transform(X_train['rating'])
# new_data['rating'] = new_data['rating'].astype(bool)
for kvp in labelEncodersTrainDict.keys():
    X_train[kvp] = labelEncodersTrainDict[kvp].fit_transform(X_train[kvp])
    X_validation[kvp] = labelEncodersTrainDict[kvp].transform(X_validation[kvp])

y_train = label_encoder_rating.fit_transform(y_train)
y_validation = label_encoder_rating.transform(y_validation)

with open('CLabel_encoding.pkl', 'wb') as f:
    pickle.dump(label_encoder_brand, f)
    pickle.dump(label_encoder_processor_brand, f)
    pickle.dump(label_encoder_processor_name, f)
    pickle.dump(label_encoder_ram_type, f)
    pickle.dump(label_encoder_os, f)
    pickle.dump(label_encoder_rating, f)


# Validation
# X_train.to_csv('New Encoding.csv', index=False)

# Scaling
# x_before_scaling = new_data.drop(columns=['rating'])

X_scaled_train = scaler.fit_transform(X_train)
X_scaled_validation = scaler.transform(X_validation)

with open('CScaling.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X_scaled_train_df = pd.DataFrame(X_scaled_train, columns=X_train.columns)
X_scaled_validation_df = pd.DataFrame(X_scaled_validation, columns=X_validation.columns)
# X_scaled_train_df = pd.DataFrame(X_scaled_train)
# X_scaled_validation_df = pd.DataFrame(X_scaled_validation)

# y = new_data['rating']
# x_scaled = scaled_df

# categorical_data_df = pd.concat([categorical_data_df, new_data['rating']], axis=1)
# categorical_data_df.to_csv('categorical.csv', index=False)


# Train Numerical
selected_numerical_columns = ['Price', 'Number of Ratings', 'Number of Reviews']

numerical_train_df = X_scaled_train_df[selected_numerical_columns].copy()

# Test Numerical
numerical_validation_df = X_scaled_validation_df[selected_numerical_columns].copy()

# numerical_data_df = pd.concat([numerical_data_df, new_data['rating']], axis=1)
# numerical_data_df.to_csv('numerical.csv', index=False)

selected_categorical_columns = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type',
                                'ssd', 'hdd', 'os', 'graphic_card_gb', 'warranty', 'Touchscreen_Yes',
                                'msoffice_Yes', 'weight_Casual', 'weight_Gaming', 'weight_ThinNlight']
# Train Categorical
categorical_train_df = X_scaled_train_df[selected_categorical_columns].copy()

# Test Categorical
categorical_validation_df = X_scaled_validation_df[selected_categorical_columns].copy()

# Categorical vs Categorical --> Use Chi-Squared or Mutual Info
np.random.seed(41)
# # Information Gain
# Train
k = 5
selector = SelectKBest(mutual_info_classif, k=k)
X_selected_train = selector.fit_transform(categorical_train_df, y_train)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_features = categorical_train_df.columns[selected_feature_indices]  # Assuming X_train is a DataFrame

# Print selected features
print("Selected Features Train:")
print(selected_features)



k = 2
selector = SelectKBest(f_classif, k=k)
X_selected_Anova_train = selector.fit_transform(numerical_train_df, y_train)

selected_feature_indices_anova_train = selector.get_support(indices=True)
selected_features_anova = numerical_train_df.columns[selected_feature_indices_anova_train]

print("Selected Features ANOVA Train:")
print(selected_features_anova)



X_selected_train_df = pd.DataFrame(X_selected_train)
X_selected_train_df.columns = selected_features
X_selected_Anova_train_df = pd.DataFrame(X_selected_Anova_train)
X_selected_Anova_train_df.columns = selected_features_anova
all_selectedFeatures_df_train = pd.concat([X_selected_train_df, X_selected_Anova_train_df], axis=1)

#validation IG
X_selected_test_df = pd.DataFrame()
for column in selected_features:
    X_selected_test_df[column] = categorical_validation_df[column]

#validation Anova
X_selected_Anova_test_df = pd.DataFrame()
for column in selected_features_anova:
    X_selected_Anova_test_df[column] = numerical_validation_df[column]

all_selectedFeatures_df_test = pd.concat([X_selected_test_df, X_selected_Anova_test_df], axis=1)

with open('Cfeatures.pkl', 'wb') as f:
    pickle.dump(all_selectedFeatures_df_train.columns, f)

logistic_model = linear_model.LogisticRegression(C=0.1)

logistic_model.fit(all_selectedFeatures_df_train, y_train)
y_forward_train_predicted = logistic_model.predict(all_selectedFeatures_df_train)

y_predict_forward_test = logistic_model.predict(all_selectedFeatures_df_test)

print('Logistic Regression Accuracy Train',
      metrics.accuracy_score(np.asarray(y_train), y_forward_train_predicted))
print('Logistic Regression Accuracy Test',
      metrics.accuracy_score(np.asarray(y_validation), y_predict_forward_test))

classifier = SVC()

classifier.fit(all_selectedFeatures_df_train, y_train)
y_train_predicted = classifier.predict(all_selectedFeatures_df_train)
y_predict_test = classifier.predict(all_selectedFeatures_df_test)
# logistic_model_backward.fit(x_after_feature_selection_backward_train, y_train)
# y_backward_train_predicted = logistic_model_backward.predict(x_after_feature_selection_backward_train)
#
# y_predict_backward_test = logistic_model_backward.predict(x_after_feature_selection_backward_test)

print('SVC Accuracy Train', metrics.accuracy_score(np.asarray(y_train), y_train_predicted))
print('SVC Accuracy Test', metrics.accuracy_score(np.asarray(y_validation), y_predict_test))

# Decision Tree
max_depth = 10
dt = DecisionTreeClassifier(max_depth=max_depth)

dt.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_dt = dt.predict(all_selectedFeatures_df_train)
y_test_predict_dt = dt.predict(all_selectedFeatures_df_test)

train_accuracy_dt = accuracy_score(y_train, y_train_predict_dt)
test_accuracy_dt = accuracy_score(y_validation, y_test_predict_dt)

print("Train Accuracy DT:", train_accuracy_dt)
print("Test Accuracy DT:", test_accuracy_dt)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_rf = rf.predict(all_selectedFeatures_df_train)
y_test_predict_rf = rf.predict(all_selectedFeatures_df_test)

train_accuracy_rf = accuracy_score(y_train, y_train_predict_rf)
test_accuracy_rf = accuracy_score(y_validation, y_test_predict_rf)

print("Train Accuracy Random Forest:", train_accuracy_rf)
print("Test Accuracy Random Forest:", test_accuracy_rf)

# KNN
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_knn = knn.predict(all_selectedFeatures_df_train)
y_test_predict_knn = knn.predict(all_selectedFeatures_df_test)

train_accuracy_knn = accuracy_score(y_train, y_train_predict_knn)
test_accuracy_knn = accuracy_score(y_validation, y_test_predict_knn)

print("Train Accuracy KNN:", train_accuracy_knn)
print("Test Accuracy KNN:", test_accuracy_knn)

# XGBOOST
xgboost = xgb.XGBClassifier()

xgboost.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_xgb = xgboost.predict(all_selectedFeatures_df_train)
y_test_predict_xgb = xgboost.predict(all_selectedFeatures_df_test)

train_accuracy_xgb = accuracy_score(y_train, y_train_predict_xgb)
test_accuracy_xgb = accuracy_score(y_validation, y_test_predict_xgb)

print("Train Accuracy XGBOOST:", train_accuracy_xgb)
print("Test Accuracy XGBOOST:", test_accuracy_xgb)

# AdaBoost
n_estimators_ada = 250
ada = AdaBoostClassifier(n_estimators=n_estimators_ada, random_state=42)

ada.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_ada = ada.predict(all_selectedFeatures_df_train)
y_test_predict_ada = ada.predict(all_selectedFeatures_df_test)

train_accuracy_ada = accuracy_score(y_train, y_train_predict_ada)
test_accuracy_ada = accuracy_score(y_validation, y_test_predict_ada)

print(f"Train Accuracy AdaBoost with n_estimators_ada = {n_estimators_ada}:", train_accuracy_ada)
print(f"Train Accuracy AdaBoost with n_estimators_ada = {n_estimators_ada}:", test_accuracy_ada)

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
    #  ('xgb', xgb.XGBClassifier()),
    ('dt', DecisionTreeClassifier()),
    # ('LR', LogisticRegression())
    # ('knn', KNeighborsClassifier())
    #  ('svc', SVC())
]

votingModel = VotingClassifier(estimators=base_learners_voting, voting='hard')

votingModel.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_voting = votingModel.predict(all_selectedFeatures_df_train)
y_test_predict_voting = votingModel.predict(all_selectedFeatures_df_test)

train_accuracy_voting = accuracy_score(y_train, y_train_predict_voting)
test_accuracy_voting = accuracy_score(y_validation, y_test_predict_voting)

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
# meta_learner = DecisionTreeClassifier()
# meta_learner = xgb.XGBClassifier()
stackingModel = StackingClassifier(estimators=base_learners_stacking, final_estimator=meta_learner)

stackingModel.fit(all_selectedFeatures_df_train, y_train)

y_train_predict_stacking = stackingModel.predict(all_selectedFeatures_df_train)
y_test_predict_stacking = stackingModel.predict(all_selectedFeatures_df_test)

train_accuracy_stacking = accuracy_score(y_train, y_train_predict_stacking)
test_accuracy_stacking = accuracy_score(y_validation, y_test_predict_stacking)

print("Train Accuracy Stacking:", train_accuracy_stacking)
print("Test Accuracy Stacking:", test_accuracy_stacking)


with open('CMyTrainModel.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)
    pickle.dump(classifier, f)
    pickle.dump(dt, f)
    pickle.dump(rf, f)
    pickle.dump(knn, f)
    pickle.dump(xgboost, f)
    pickle.dump(ada, f)
    pickle.dump(votingModel, f)
    pickle.dump(stackingModel, f)