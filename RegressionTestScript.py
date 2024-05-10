import pickle
import pandas as pd
from sklearn import metrics
import numpy as np

# mean and mode for every column to handle nulls 
# Encoding (ordinal , label , one hot)
# Drop No Columns in One Hot Encoding
# Replacing Not Available in generation column
# From float to int64
# Save Scaling
# Save Feature Selection 
# Save Models

Newdata = pd.read_csv("testData.csv")
test_data = Newdata
with open('Mean.pkl', 'rb') as f:
    Price_Mean = pickle.load(f)
    Number_of_Ratings_mean = pickle.load(f)
    Number_of_Reviews_mean = pickle.load(f)

with open('Mode.pkl', 'rb') as f:
    brand_mode = pickle.load(f)
    processor_brand_mode = pickle.load(f)
    processor_name_mode = pickle.load(f)
    processor_gnrtn_mode = pickle.load(f)
    ram_gb_mode = pickle.load(f)
    ram_type_mode = pickle.load(f)
    ssd_mode = pickle.load(f)
    hdd_mode = pickle.load(f)
    os_mode = pickle.load(f)
    graphic_card_gb_mode = pickle.load(f)
    weight_mode = pickle.load(f)
    warranty_mode = pickle.load(f)
    Touchscreen_mode = pickle.load(f)
    msoffice_mode = pickle.load(f)

with open('ordinal_encoding.pkl', 'rb') as f:
    ordinal_encoder_ram = pickle.load(f)
    ordinal_encoder_ssd = pickle.load(f)
    ordinal_encoder_hdd = pickle.load(f)
    ordinal_encoder_graphic_card_gb = pickle.load(f)
    ordinal_encoder_warranty = pickle.load(f)
    ordinal_encoder_generation = pickle.load(f)
    ordinal_encoder_rating = pickle.load(f)

with open('Label_encoding.pkl', 'rb') as f:
    label_encoder_brand = pickle.load(f)
    label_encoder_processor_brand = pickle.load(f)
    label_encoder_processor_name = pickle.load(f)
    label_encoder_ram_type = pickle.load(f)
    label_encoder_os = pickle.load(f)

with open('Scaling.pkl', 'rb') as f:
    scaler = pickle.load(f)


with open('FeatureSelectionForward.pkl', 'rb') as f:
    selected_features_forward_train = pickle.load(f)


with open('FeatureSelectionBackward.pkl', 'rb') as f:
    selected_features_backward_train = pickle.load(f)


with open('poly_features_forward.pkl', 'rb') as f:
    poly_features_forward = pickle.load(f)


with open('poly_features_backward.pkl', 'rb') as f:
    poly_features_backward = pickle.load(f)


with open('MyTrainModel.pkl', 'rb') as f:
    linear_model_forward = pickle.load(f)
    linear_model_backward = pickle.load(f)

    poly_model_forward = pickle.load(f)
    poly_model_backward = pickle.load(f)

    svr_forward = pickle.load(f)
    svr_backward = pickle.load(f)

    dt_regressor_forward = pickle.load(f)
    dt_regressor_backward = pickle.load(f)


test_data['Price'] = test_data['Price'].apply(lambda x: x if x >= 0 else pd.NA)
test_data['Number of Ratings'] = test_data['Number of Ratings'].apply(lambda x: x if x >= 0 else pd.NA)
test_data['Number of Reviews'] = test_data['Number of Reviews'].apply(lambda x: x if x >= 0 else pd.NA)


test_data['Price'] = test_data['Price'].fillna(Price_Mean)
test_data['Number of Ratings'] = test_data['Number of Ratings'].fillna(Number_of_Ratings_mean)
test_data['Number of Reviews'] = test_data['Number of Reviews'].fillna(Number_of_Reviews_mean)

test_data['brand'] = test_data['brand'].fillna(brand_mode)
test_data['processor_brand'] = test_data['processor_brand'].fillna(processor_brand_mode)
test_data['processor_name'] = test_data['processor_name'].fillna(processor_name_mode)
test_data['ram_gb'] = test_data['ram_gb'].fillna(ram_gb_mode)
test_data['ssd'] = test_data['ssd'].fillna(ssd_mode)
test_data['hdd'] = test_data['hdd'].fillna(hdd_mode)
test_data['os'] = test_data['os'].fillna(os_mode)
test_data['graphic_card_gb'] = test_data['graphic_card_gb'].fillna(graphic_card_gb_mode)
test_data['weight'] = test_data['weight'].fillna(weight_mode)
test_data['warranty'] = test_data['warranty'].fillna(warranty_mode)
test_data['Touchscreen'] = test_data['Touchscreen'].fillna(Touchscreen_mode)
test_data['msoffice'] = test_data['msoffice'].fillna(msoffice_mode)


test_data['brand'] = test_data['brand'].replace("Not Available", pd.NA)
test_data['brand'] = test_data['brand'].fillna(brand_mode)

test_data['processor_brand'] = test_data['processor_brand'].replace("Not Available", pd.NA)
test_data['processor_brand'] = test_data['processor_brand'].fillna(processor_brand_mode)


test_data['processor_name'] = test_data['processor_name'].replace("Not Available", pd.NA)
test_data['processor_name'] = test_data['processor_name'].fillna(processor_name_mode)


test_data['ram_gb'] = test_data['ram_gb'].replace("Not Available", pd.NA)
test_data['ram_gb'] = test_data['ram_gb'].fillna(ram_gb_mode)


test_data['ssd'] = test_data['ssd'].replace("Not Available", pd.NA)
test_data['ssd'] = test_data['ssd'].fillna(ssd_mode)

test_data['hdd'] = test_data['hdd'].replace("Not Available", pd.NA)
test_data['hdd'] = test_data['hdd'].fillna(hdd_mode)

test_data['os'] = test_data['os'].replace("Not Available", pd.NA)
test_data['os'] = test_data['os'].fillna(os_mode)

test_data['graphic_card_gb'] = test_data['graphic_card_gb'].replace("Not Available", pd.NA)
test_data['graphic_card_gb'] = test_data['graphic_card_gb'].fillna(graphic_card_gb_mode)

test_data['processor_gnrtn'] = test_data['processor_gnrtn'].replace("Not Available", pd.NA)
test_data['processor_gnrtn'] = test_data['processor_gnrtn'].fillna(processor_gnrtn_mode)

test_data['weight'] = test_data['weight'].replace("Not Available", pd.NA)
test_data['weight'] = test_data['weight'].fillna(weight_mode)

test_data['warranty'] = test_data['warranty'].replace("Not Available", pd.NA)
test_data['warranty'] = test_data['warranty'].fillna(warranty_mode)

test_data['Touchscreen'] = test_data['Touchscreen'].replace("Not Available", pd.NA)
test_data['Touchscreen'] = test_data['Touchscreen'].fillna(Touchscreen_mode)

test_data['msoffice'] = test_data['msoffice'].replace("Not Available", pd.NA)
test_data['msoffice'] = test_data['msoffice'].fillna(msoffice_mode)

test_data['ram_gb'] = ordinal_encoder_ram.transform(test_data[['ram_gb']])
test_data['ssd'] = ordinal_encoder_ssd.transform(test_data[['ssd']])
test_data['hdd'] = ordinal_encoder_hdd.transform(test_data[['hdd']])
test_data['graphic_card_gb'] = ordinal_encoder_graphic_card_gb.transform(test_data[['graphic_card_gb']])
test_data['warranty'] = ordinal_encoder_warranty.transform(test_data[['warranty']])
test_data['processor_gnrtn'] = ordinal_encoder_generation.transform(test_data[['processor_gnrtn']])
test_data['rating'] = ordinal_encoder_rating.transform(test_data[['rating']])

# From float to int64
test_data['ram_gb'] = test_data['ram_gb'].astype('int64')
test_data['ssd'] = test_data['ssd'].astype('int64')
test_data['hdd'] = test_data['hdd'].astype('int64')
test_data['graphic_card_gb'] = test_data['graphic_card_gb'].astype('int64')
test_data['warranty'] = test_data['warranty'].astype('int64')
test_data['processor_gnrtn'] = test_data['processor_gnrtn'].astype('int64')
test_data['rating'] = test_data['rating'].astype('int64')


# Touch msoffice weight --> One Hot Encoding
test_data = pd.get_dummies(test_data, columns=['Touchscreen'])
test_data = pd.get_dummies(test_data, columns=['msoffice'])
test_data = pd.get_dummies(test_data, columns=['weight'])

test_data['Touchscreen_No'] = test_data['Touchscreen_No'].map({True: 1, False: 0})
test_data['Touchscreen_Yes'] = test_data['Touchscreen_Yes'].map({True: 1, False: 0})

test_data['msoffice_No'] = test_data['msoffice_No'].map({True: 1, False: 0})
test_data['msoffice_Yes'] = test_data['msoffice_Yes'].map({True: 1, False: 0})

test_data['weight_Casual'] = test_data['weight_Casual'].map({True: 1, False: 0})
test_data['weight_Gaming'] = test_data['weight_Gaming'].map({True: 1, False: 0})
test_data['weight_ThinNlight'] = test_data['weight_ThinNlight'].map({True: 1, False: 0})

# Brand , Processor brand, processor name, ram type, os --> Label Encoding
test_data['brand'] = label_encoder_brand.transform(test_data['brand'])
test_data['processor_brand'] = label_encoder_processor_brand.transform(test_data['processor_brand'])
test_data['processor_name'] = label_encoder_processor_name.transform(test_data['processor_name'])
test_data['ram_type'] = label_encoder_ram_type.transform(test_data['ram_type'])
test_data['os'] = label_encoder_os.transform(test_data['os'])

Y = test_data['rating']
X = test_data.drop('rating', axis=1)
scaled_data = scaler.transform(X)
scaled_df = pd.DataFrame(scaled_data, columns=X.columns)

X_Forward_features = pd.DataFrame()
X_Backward_features = pd.DataFrame()

# Forward
for column in selected_features_forward_train:
    column_name = X.columns[column]
    column_values_test = X.iloc[:, column]
    X_Forward_features[column_name] = column_values_test
# Backward
for column in selected_features_backward_train:
    column_name = X.columns[column]
    column_values_test = X.iloc[:, column]
    X_Backward_features[column_name] = column_values_test

y_predict_forward_test = linear_model_forward.predict(X_Forward_features)
y_predict_backward_test = linear_model_backward.predict(X_Backward_features)

print("\nFirst Model - Linear Regression: Using Forward Selection for Feature Selection")
print('Linear Regression Mean Square Error Forward Test', metrics.mean_squared_error(np.asarray(Y),
                                                                                     y_predict_forward_test))
print('Linear Regression R2 Score Forward Test', metrics.r2_score(np.asarray(Y),
                                                                                     y_predict_forward_test))

print("\nSecond Model - Linear Regression: Using Backward Elimination for Feature Selection")
print('Linear Regression Mean Square Error Backward Test', metrics.mean_squared_error(np.asarray(Y),
                                                                                      y_predict_backward_test))

print('Linear Regression R2 Score Backward Test', metrics.r2_score(np.asarray(Y),
                                                                                      y_predict_backward_test))                                                                                                                                                                    

y_predict_test_forward = poly_model_forward.predict(poly_features_forward.transform(X_Forward_features))
y_predict_test_backward = poly_model_backward.predict(poly_features_backward.transform(X_Backward_features))

print("\nThird Model - Polynomial Regression: Using Forward Selection for Feature Selection")
print('Polynomial Regression Mean Square Error Forward Test ', metrics.mean_squared_error(np.asarray(Y),
                                                                                          y_predict_test_forward))
print('Polynomial Regression R2 Score Forward Test ', metrics.r2_score(np.asarray(Y),
                                                                                          y_predict_test_forward))                                                                                          

print("\nFourth Model - Polynomial Regression: Using Backward Elimination for Feature Selection")
print('Polynomial Regression Mean Square Error Backward Test ', metrics.mean_squared_error(np.asarray(Y),
                                                                                           y_predict_test_backward))
print('Polynomial Regression R2 score Backward Test ', metrics.r2_score(np.asarray(Y),
                                                                                           y_predict_test_backward))                                                                                           

y_test_predict_forward = svr_forward.predict(X_Forward_features)
test_mse_forward = metrics.mean_squared_error(Y, y_test_predict_forward)
test_r2_score = metrics.r2_score(Y, y_test_predict_forward)
print("\nFifth Model - SVR: Using Forward Selection for Feature Selection")
print("Test MSE Forward SVR:", test_mse_forward)
print("Test R2 Score Forward SVR:", test_r2_score)

y_test_predict_backward = svr_backward.predict(X_Backward_features)
test_mse_backward = metrics.mean_squared_error(Y, y_test_predict_backward)
test_r2_score = metrics.r2_score(Y, y_test_predict_backward)
print("\nSixth Model - SVR: Using Backward Elimination for Feature Selection")
print("Test MSE Backward SVR:", test_mse_backward)
print("Test R2 Score Backward SVR:", test_r2_score)

y_test_predict_forward_dt = dt_regressor_forward.predict(X_Forward_features)
test_mse_forward_dt = metrics.mean_squared_error(Y, y_test_predict_forward_dt)
test_r2_score = metrics.r2_score(Y, y_test_predict_forward_dt)
print("\nSeventh Model - Decision Tree Regressor: Using Forward Selection for Feature Selection")
print("Test MSE Forward DT:", test_mse_forward_dt)
print("Test R2 Score Forward DT:", test_r2_score)

y_test_predict_backward_dt = dt_regressor_backward.predict(X_Backward_features)
test_mse_backward_dt = metrics.mean_squared_error(Y, y_test_predict_backward_dt)
test_r2_score = metrics.r2_score(Y, y_test_predict_backward_dt)
print("\nEighth Model - Decision Tree Regressor: Using Backward Elimination for Feature Selection")
print("Test MSE Backward DT:", test_mse_backward_dt)
print("Test R2 Score Backward DT:", test_r2_score)
