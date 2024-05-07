import pickle
import warnings
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

warnings.filterwarnings("ignore", category=FutureWarning)


ram_categories = ['4 GB', '8 GB', '16 GB', '32 GB']
ssd_categories = ['0 GB', '128 GB', '256 GB', '512 GB', '1024 GB', '2048 GB', '3072 GB']
hdd_categories = ['0 GB', '512 GB', '1024 GB', '2048 GB']
graphic_card_gb_categories = ['0 GB', '2 GB', '4 GB', '6 GB', '8 GB']
warranty_categories = ['No warranty', '1 year', '3 years', '2 years']
generation_categories = ['4th', '7th', '8th', '9th', '10th', '11th', '12th']
rating_categories = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']


#label_encoder = LabelEncoder()
scaler = StandardScaler()

label_encoder_brand = LabelEncoder()
label_encoder_processor_brand = LabelEncoder()
label_encoder_processor_name = LabelEncoder()
label_encoder_ram_type = LabelEncoder()
label_encoder_os = LabelEncoder()


ordinal_encoder_ram = OrdinalEncoder(categories=[ram_categories], handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_ssd = OrdinalEncoder(categories=[ssd_categories],handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_hdd = OrdinalEncoder(categories=[hdd_categories],handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_graphic_card_gb = OrdinalEncoder(categories=[graphic_card_gb_categories],handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_warranty = OrdinalEncoder(categories=[warranty_categories], handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_generation = OrdinalEncoder(categories=[generation_categories], handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_rating = OrdinalEncoder(categories=[rating_categories], handle_unknown='use_encoded_value', unknown_value=-1)

data = pd.read_csv("ElecDeviceRatingPrediction.csv")

new_data = data
# Price,  Number of Ratings , Number of Reviews ---> Numerical columns ----> mean 
Price_Mean=new_data['Price'].mean()
Number_of_Ratings_mean=new_data['Number of Ratings'].mean()
Number_of_Reviews_mean=new_data['Number of Reviews'].mean()

brand_mode = new_data['brand'].mode()
processor_brand_mode = new_data['processor_brand'].mode()
processor_name_mode = new_data['processor_name'].mode()
processor_gnrtn_mode = new_data['processor_gnrtn'].mode()
ram_gb_mode = new_data['ram_gb'].mode()
ram_type_mode=new_data['ram_type'].mode()
ssd_mode = new_data['ssd'].mode()
hdd_mode = new_data['hdd'].mode()
os_mode = new_data['os'].mode()
graphic_card_gb_mode = new_data['graphic_card_gb'].mode()
weight_mode = new_data['weight'].mode()
warranty_mode = new_data['warranty'].mode()
Touchscreen_mode=new_data['Touchscreen'].mode()
msoffice_mode = new_data['msoffice'].mode()


with open('Mean.pkl', 'wb') as f:
    pickle.dump(Price_Mean, f)
    pickle.dump(Number_of_Ratings_mean, f)
    pickle.dump(Number_of_Reviews_mean, f)

with open('Mode.pkl', 'wb') as f:
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

with open('ordinal_encoding.pkl', 'wb') as f:
    pickle.dump(ordinal_encoder_ram, f)
    pickle.dump(ordinal_encoder_ssd, f)
    pickle.dump(ordinal_encoder_hdd, f)
    pickle.dump(ordinal_encoder_graphic_card_gb, f)
    pickle.dump(ordinal_encoder_warranty, f)
    pickle.dump(ordinal_encoder_generation, f)
    pickle.dump(ordinal_encoder_rating, f)


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
new_data['brand'] = label_encoder_brand.fit_transform(new_data['brand'])
new_data['processor_brand'] = label_encoder_processor_brand.fit_transform(new_data['processor_brand'])
new_data['processor_name'] = label_encoder_processor_name.fit_transform(new_data['processor_name'])
new_data['ram_type'] = label_encoder_ram_type.fit_transform(new_data['ram_type'])
new_data['os'] = label_encoder_os.fit_transform(new_data['os'])


with open('Label_encoding.pkl', 'wb') as f:
    pickle.dump(label_encoder_brand,f)
    pickle.dump(label_encoder_processor_brand,f)
    pickle.dump(label_encoder_processor_name,f)
    pickle.dump(label_encoder_ram_type,f)
    pickle.dump(label_encoder_os,f)



# Scaling
X = new_data.drop('rating', axis=1)
scaled_data = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_data,columns =X.columns)
scaled_df['rating'] = new_data['rating']

with open('Scaling.pkl', 'wb') as f:
    pickle.dump(scaler,f)

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
#plt.show()

x_top_features = top_feature.drop('rating')
x_corr = scaled_df[x_top_features]

feature_column_corr = x_corr.columns
with open('FeatureSelectionCorr.pkl', 'wb') as f:
    pickle.dump(feature_column_corr,f)

# Correlation Linear Regression
y_scaled_corr = scaled_df['rating']
X_train_corr, X_test_corr, y_train_corr, y_test_corr = train_test_split(x_corr, y_scaled_corr, test_size=0.20, shuffle=True, random_state=10)

linear_model_corr = linear_model.LinearRegression()

linear_model_corr.fit(X_train_corr, y_train_corr)
y_corr_train_predicted = linear_model_corr.predict(X_train_corr)

y_predict_corr_test = linear_model_corr.predict(X_test_corr)

print("First Model - Linear Regression: Using Correlation for Feature Selection")

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

print("\nSecond Model - Polynomial Regression: Using Correlation for Feature Selection")

print('Mean Square Error Correlation Train ',
      metrics.mean_squared_error(np.asarray(y_train_corr), y_train_predicted_corr))

print('Mean Square Error Correlation Test ', metrics.mean_squared_error(np.asarray(y_test_corr), y_predict_test_corr))





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

print("\nThird Model - Linear Regression: Using Forward Selection for Feature Selection")

print('Linear Regression Mean Square Error Forward Train', metrics.mean_squared_error(np.asarray(y_train), y_forward_train_predicted))
print('Linear Regression Mean Square Error Forward Test', metrics.mean_squared_error(np.asarray(y_test), y_predict_forward_test))


print("\nFourth Model - Linear Regression: Using Backward Elimination for Feature Selection")

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

print("\n----- Best Model -----")
print("Fifth Model - Polynomial Regression: Using Forward Selection for Feature Selection")

print('Polynomial Regression Mean Square Error Forward Train ',
      metrics.mean_squared_error(np.asarray(y_train), y_train_predicted_forward))
print('Polynomial Regression Mean Square Error Forward Test ', metrics.mean_squared_error(np.asarray(y_test), y_predict_test_forward))

print("\nSixth Model - Polynomial Regression: Using Backward Elimination for Feature Selection")

print('Polynomial Regression Mean Square Error Backward Train ',
      metrics.mean_squared_error(np.asarray(y_train), y_train_predicted_backward))
print('Polynomial Regression Mean Square Error Backward Test ', metrics.mean_squared_error(np.asarray(y_test), y_predict_test_backward))


# SVR
svr_forward = SVR(kernel='rbf')
svr_backward = SVR(kernel='rbf')
# Forward
svr_forward.fit(x_after_feature_selection_forward_train, y_train)
y_train_predict_forward = svr_forward.predict(x_after_feature_selection_forward_train)
y_test_predict_forward = svr_forward.predict(x_after_feature_selection_forward_test)

train_mse_forward = mean_squared_error(y_train, y_train_predict_forward)
test_mse_forward = mean_squared_error(y_test, y_test_predict_forward)

print("\nSeventh Model - SVR: Using Forward Selection for Feature Selection")

print("Train MSE Forward SVR:", train_mse_forward)
print("Test MSE Forward SVR:", test_mse_forward)

# Backward
svr_backward.fit(x_after_feature_selection_backward_train, y_train)
y_train_predict_backward = svr_backward.predict(x_after_feature_selection_backward_train)
y_test_predict_backward = svr_backward.predict(x_after_feature_selection_backward_test)

train_mse_backward = mean_squared_error(y_train, y_train_predict_backward)
test_mse_backward = mean_squared_error(y_test, y_test_predict_backward)

print("\nEighth Model - SVR: Using Backward Elimination for Feature Selection")

print("Train MSE Backward SVR:", train_mse_backward)
print("Test MSE Backward SVR:", test_mse_backward)

# Decision Tree Regression
dt_regressor_forward = DecisionTreeRegressor()
dt_regressor_backward = DecisionTreeRegressor()

# Forward
dt_regressor_forward.fit(x_after_feature_selection_forward_train, y_train)
y_train_predict_forward_dt = dt_regressor_forward.predict(x_after_feature_selection_forward_train)
y_test_predict_forward_dt = dt_regressor_forward.predict(x_after_feature_selection_forward_test)

train_mse_forward_dt = mean_squared_error(y_train, y_train_predict_forward_dt)
test_mse_forward_dt = mean_squared_error(y_test, y_test_predict_forward_dt)

print("\nNinth Model - Decision Tree Regressor: Using Forward Selection for Feature Selection")

print("Train MSE Forward DT:", train_mse_forward_dt)
print("Test MSE Forward DT:", test_mse_forward_dt)

# Backward
dt_regressor_backward.fit(x_after_feature_selection_backward_train, y_train)
y_train_predict_backward_dt = dt_regressor_backward.predict(x_after_feature_selection_backward_train)
y_test_predict_backward_dt = dt_regressor_backward.predict(x_after_feature_selection_backward_test)

train_mse_backward_dt = mean_squared_error(y_train, y_train_predict_backward_dt)
test_mse_backward_dt = mean_squared_error(y_test, y_test_predict_backward_dt)

print("\nTenth Model - Decision Tree Regressor: Using Backward Elimination for Feature Selection")

print("Train MSE Backward DT:", train_mse_backward_dt)
print("Test MSE Backward DT:", test_mse_backward_dt)


with open('FeatureSelectionPolyCorr.pkl', 'wb') as f:
    pickle.dump(poly_features_corr,f)
    
with open('FeatureSelectionForward.pkl', 'wb') as f:
    pickle.dump(selected_features_forward_train,f)


with open('FeatureSelectionBackward.pkl', 'wb') as f:
    pickle.dump(selected_features_backward_train,f)


with open('poly_features_forward.pkl', 'wb') as f:
    pickle.dump(poly_features_forward,f)


with open('poly_features_backward.pkl', 'wb') as f:
    pickle.dump(poly_features_backward,f)

poly_features_forward
with open('MyTrainModel.pkl', 'wb') as f:

    pickle.dump(linear_model_corr, f)
    pickle.dump(poly_model_corr,f)

    pickle.dump(linear_model_forward, f)
    pickle.dump(linear_model_backward,f)

    pickle.dump(poly_model_forward, f)
    pickle.dump(poly_model_backward,f)

    pickle.dump(svr_forward, f)
    pickle.dump(svr_backward, f)

    pickle.dump(dt_regressor_forward,f)
    pickle.dump(dt_regressor_backward,f)
    
