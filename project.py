import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

ram_categories = ['4 GB', '8 GB', '16 GB', '32 GB']
ssd_categories = ['0 GB', '128 GB', '256 GB', '512 GB', '1024 GB', '2048 GB', '3072 GB']
hdd_categories = ['0 GB', '512 GB', '1024 GB', '2048 GB']
graphic_card_gb_categories = ['0 GB', '2 GB', '4 GB', '6 GB', '8 GB']
warranty_categories = ['No warranty', '1 year', '3 years', '2 years']
generation_categories = ['4th', '7th', '8th', '9th', '10th', '11th', '12th']
rating_categories = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

label_encoder = LabelEncoder()
scaler = StandardScaler()

ordinal_encoder_ram = OrdinalEncoder(categories=[ram_categories])
ordinal_encoder_ssd = OrdinalEncoder(categories=[ssd_categories])
ordinal_encoder_hdd = OrdinalEncoder(categories=[hdd_categories])
ordinal_encoder_graphic_card_gb = OrdinalEncoder(categories=[graphic_card_gb_categories])
ordinal_encoder_warranty = OrdinalEncoder(categories=[warranty_categories])
ordinal_encoder_generation = OrdinalEncoder(categories=[generation_categories])
ordinal_encoder_rating = OrdinalEncoder(categories=[rating_categories])

data = pd.read_csv("ElecDeviceRatingPrediction.csv")
new_data = data

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
new_data['rating'] = ordinal_encoder_rating.fit_transform(new_data[['rating']])

# From float to int64
new_data['ram_gb'] = new_data['ram_gb'].astype('int64')
new_data['ssd'] = new_data['ssd'].astype('int64')
new_data['hdd'] = new_data['hdd'].astype('int64')
new_data['graphic_card_gb'] = new_data['graphic_card_gb'].astype('int64')
new_data['warranty'] = new_data['warranty'].astype('int64')
new_data['processor_gnrtn'] = new_data['processor_gnrtn'].astype('int64')
new_data['rating'] = new_data['rating'].astype('int64')

# Touch msoffice weight --> One Hot Encoding
new_data = pd.get_dummies(new_data, columns=['Touchscreen'])
new_data = pd.get_dummies(new_data, columns=['msoffice'])
new_data = pd.get_dummies(new_data, columns=['weight'])


new_data['Touchscreen_No'] = new_data['Touchscreen_No'].map({True: 1, False: 0})
new_data['Touchscreen_Yes'] = new_data['Touchscreen_Yes'].map({True: 1, False: 0})

new_data['msoffice_No'] = new_data['msoffice_No'].map({True: 1, False: 0})
new_data['msoffice_Yes'] = new_data['msoffice_Yes'].map({True: 1, False: 0})

new_data['weight_Casual'] = new_data['weight_Casual'].map({True: 1, False: 0})
new_data['weight_Gaming'] = new_data['weight_Gaming'].map({True: 1, False: 0})
new_data['weight_ThinNlight'] = new_data['weight_ThinNlight'].map({True: 1, False: 0})

new_data.drop(columns=['msoffice_No'], inplace=True)
new_data.drop(columns=['Touchscreen_No'], inplace=True)

# Brand , Processor brand, processor name, ram type, os --> Label Encoding
new_data['brand'] = label_encoder.fit_transform(new_data['brand'])
new_data['processor_brand'] = label_encoder.fit_transform(new_data['processor_brand'])
new_data['processor_name'] = label_encoder.fit_transform(new_data['processor_name'])
new_data['ram_type'] = label_encoder.fit_transform(new_data['ram_type'])
new_data['os'] = label_encoder.fit_transform(new_data['os'])


# Scaling
scaled_data = scaler.fit_transform(new_data)
scaled_df = pd.DataFrame(scaled_data, columns=new_data.columns)

# scaled_df.to_csv('output.csv', index=False)

# Visualization
# target_column = 'rating'
# for column in scaled_df:
#     if column != target_column:
#         plt.figure(figsize=(8, 6))
#         sns.scatterplot(x=scaled_df[column], y=scaled_df[target_column])
#         plt.xlabel(column)
#         plt.ylabel(target_column)
#         plt.title(f'Scatter Plot: {column} vs {target_column}')
#         plt.show()
#
# median_line_color = 'red'
# for column in new_data:
#     if column != target_column:
#         plt.figure(figsize=(8, 6))
#         sns.boxplot(x=new_data[column], y=new_data[target_column], medianprops={'color': median_line_color})
#         plt.xlabel(column)
#         plt.ylabel(target_column)
#         plt.title(f'Box Plot: {target_column} vs {column}')
#
#         plt.show()

# Feature Selection
corr_data = scaled_df.iloc[:, :]
corr = corr_data.corr()
top_feature = corr.index[abs((corr['rating']) > 0.1)]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = corr_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

x_top_features = top_feature.drop('rating')
x_corr = scaled_df[x_top_features]

# Correlation Linear Regression
y_scaled_corr = scaled_df['rating']
X_train_corr, X_test_corr, y_train_corr, y_test_corr = train_test_split(x_corr, y_scaled_corr, test_size=0.20, shuffle=True, random_state=10)

linear_model_corr = linear_model.LinearRegression()

linear_model_corr.fit(X_train_corr, y_train_corr)
y_corr_train_predicted = linear_model_corr.predict(X_train_corr)

y_predict_corr_test = linear_model_corr.predict(X_test_corr)

print('Mean Square Error Correlation Train', metrics.mean_squared_error(np.asarray(y_train_corr), y_corr_train_predicted))
print('Mean Square Error Correlation Test', metrics.mean_squared_error(np.asarray(y_test_corr), y_predict_corr_test))

# Correlation Polynomial Regression
poly_features_corr = PolynomialFeatures(degree=3)

X_train_poly_corr = poly_features_corr.fit_transform(X_train_corr)

poly_model_corr = linear_model.LinearRegression()

poly_model_corr.fit(X_train_poly_corr, y_train_corr)

y_train_predicted_corr = poly_model_corr.predict(X_train_poly_corr)

y_predict_test_corr = poly_model_corr.predict(
    poly_features_corr.transform(X_test_corr))

print('Mean Square Error Corr Train Poly',
      metrics.mean_squared_error(np.asarray(y_train_corr), y_train_predicted_corr))

print('Mean Square Error Corr Test Poly', metrics.mean_squared_error(np.asarray(y_test_corr), y_predict_test_corr))

y_scaled = scaled_df['rating']
x_scaled = scaled_df.drop(columns=['rating'])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.20, shuffle=True, random_state=10)

selector_forward_train = SequentialFeatureSelector(LinearRegression(), forward=True, k_features='best',
                                                   scoring='neg_mean_squared_error', cv=5)
selector_forward_train.fit(X_train, y_train)

selector_backward_train = SequentialFeatureSelector(LinearRegression(), forward=False, k_features='best',
                                                    scoring='neg_mean_squared_error', cv=5)
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

# x_after_feature_selection_backward_test.to_csv('backward.csv', index=False)


linear_model_forward = linear_model.LinearRegression()
linear_model_backward = linear_model.LinearRegression()

linear_model_forward.fit(x_after_feature_selection_forward_train, y_train)
y_forward_train_predicted = linear_model_forward.predict(x_after_feature_selection_forward_train)

y_predict_forward_test = linear_model_forward.predict(x_after_feature_selection_forward_test)

linear_model_backward.fit(x_after_feature_selection_backward_train, y_train)
y_backward_train_predicted = linear_model_backward.predict(x_after_feature_selection_backward_train)

y_predict_backward_test = linear_model_backward.predict(x_after_feature_selection_backward_test)

print('Linear Regression Mean Square Error Forward Train', metrics.mean_squared_error(np.asarray(y_train), y_forward_train_predicted))
print('Linear Regression Mean Square Error Forward Test', metrics.mean_squared_error(np.asarray(y_test), y_predict_forward_test))

print('Linear Regression Mean Square Error Backward Train', metrics.mean_squared_error(np.asarray(y_train), y_backward_train_predicted))
print('Linear Regression Mean Square Error Backward Test', metrics.mean_squared_error(np.asarray(y_test), y_predict_backward_test))

poly_features_forward = PolynomialFeatures(degree=2)
poly_features_backward = PolynomialFeatures(degree=2)

X_train_poly_forward = poly_features_forward.fit_transform(x_after_feature_selection_forward_train)
X_train_poly_backward = poly_features_backward.fit_transform(x_after_feature_selection_backward_train)

poly_model_forward = linear_model.LinearRegression()
poly_model_backward = linear_model.LinearRegression()

poly_model_forward.fit(X_train_poly_forward, y_train)
poly_model_backward.fit(X_train_poly_backward, y_train)

y_train_predicted_forward = poly_model_forward.predict(X_train_poly_forward)
y_train_predicted_backward = poly_model_backward.predict(X_train_poly_backward)

y_predict_test_forward = poly_model_forward.predict(
    poly_features_forward.transform(x_after_feature_selection_forward_test))
y_predict_test_backward = poly_model_backward.predict(
    poly_features_backward.transform(x_after_feature_selection_backward_test))

print('Polynomial Regression Mean Square Error Forward Train Poly',
      metrics.mean_squared_error(np.asarray(y_train), y_train_predicted_forward))
print('Polynomial Regression Mean Square Error Forward Test Poly', metrics.mean_squared_error(np.asarray(y_test), y_predict_test_forward))

print('Polynomial Regression Mean Square Error Backward Train Poly',
      metrics.mean_squared_error(np.asarray(y_train), y_train_predicted_backward))
print('Polynomial Regression Mean Square Error Backward Test Poly', metrics.mean_squared_error(np.asarray(y_test), y_predict_test_backward))


# SVR
svr = SVR(kernel='rbf')

# Forward
svr.fit(x_after_feature_selection_forward_train, y_train)
y_train_predict_forward = svr.predict(x_after_feature_selection_forward_train)
y_test_predict_forward = svr.predict(x_after_feature_selection_forward_test)

train_mse_forward = mean_squared_error(y_train, y_train_predict_forward)
test_mse_forward = mean_squared_error(y_test, y_test_predict_forward)

print("Train MSE Forward SVR:", train_mse_forward)
print("Test MSE Forward SVR:", test_mse_forward)

# Backward
svr.fit(x_after_feature_selection_backward_train, y_train)
y_train_predict_backward = svr.predict(x_after_feature_selection_backward_train)
y_test_predict_backward = svr.predict(x_after_feature_selection_backward_test)

train_mse_backward = mean_squared_error(y_train, y_train_predict_backward)
test_mse_backward = mean_squared_error(y_test, y_test_predict_backward)

print("Train MSE Backward SVR:", train_mse_backward)
print("Test MSE Backward SVR:", test_mse_backward)

# Decision Tree Regression
dt_regressor = DecisionTreeRegressor()

# Forward
dt_regressor.fit(x_after_feature_selection_forward_train, y_train)
y_train_predict_forward_dt = dt_regressor.predict(x_after_feature_selection_forward_train)
y_test_predict_forward_dt = dt_regressor.predict(x_after_feature_selection_forward_test)

train_mse_forward_dt = mean_squared_error(y_train, y_train_predict_forward_dt)
test_mse_forward_dt = mean_squared_error(y_test, y_test_predict_forward_dt)

print("Train MSE Forward DT:", train_mse_forward_dt)
print("Test MSE Forward DT:", test_mse_forward_dt)

# Backward
dt_regressor.fit(x_after_feature_selection_backward_train, y_train)
y_train_predict_backward_dt = dt_regressor.predict(x_after_feature_selection_backward_train)
y_test_predict_backward_dt = dt_regressor.predict(x_after_feature_selection_backward_test)

train_mse_backward_dt = mean_squared_error(y_train, y_train_predict_backward_dt)
test_mse_backward_dt = mean_squared_error(y_test, y_test_predict_backward_dt)

print("Train MSE Backward DT:", train_mse_backward_dt)
print("Test MSE Backward DT:", test_mse_backward_dt)


