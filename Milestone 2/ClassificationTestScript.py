import pickle
import pandas as pd
from sklearn import metrics
import numpy as np

Newdata = pd.read_csv("G:\\Pattern Recognition\\Pattern-Recognition-Project\\Milestone 2\\testData.csv")
test_data = Newdata

with open('CMean.pkl', 'rb') as f:
    Price_Mean = pickle.load(f)
    Number_of_Ratings_mean = pickle.load(f)
    Number_of_Reviews_mean = pickle.load(f)

with open('CMode.pkl', 'rb') as f:
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

with open('Cordinal_encoding.pkl', 'rb') as f:
    ordinal_encoder_ram = pickle.load(f)
    ordinal_encoder_ssd = pickle.load(f)
    ordinal_encoder_hdd = pickle.load(f)
    ordinal_encoder_graphic_card_gb = pickle.load(f)
    ordinal_encoder_warranty = pickle.load(f)
    ordinal_encoder_generation = pickle.load(f)

with open('CLabel_encoding.pkl', 'rb') as f:
    label_encoder_brand = pickle.load(f)
    label_encoder_processor_brand = pickle.load(f)
    label_encoder_processor_name = pickle.load(f)
    label_encoder_ram_type = pickle.load(f)
    label_encoder_os = pickle.load(f)
    label_encoder_rating = pickle.load(f)

with open('CScaling.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Cfeatures.pkl', 'rb') as f:
    all_selectedFeatures_df_columns = pickle.load(f)

with open('CMyTrainModel.pkl', 'rb') as f:
    logistic_model = pickle.load(f)
    classifier = pickle.load(f)
    dt = pickle.load(f)
    rf = pickle.load(f)
    knn = pickle.load(f)
    xgboost = pickle.load(f)
    ada = pickle.load(f)
    votingModel = pickle.load(f)
    stackingModel = pickle.load(f)
   

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

# From float to int64
test_data['ram_gb'] = test_data['ram_gb'].astype('int64')
test_data['ssd'] = test_data['ssd'].astype('int64')
test_data['hdd'] = test_data['hdd'].astype('int64')
test_data['graphic_card_gb'] = test_data['graphic_card_gb'].astype('int64')
test_data['warranty'] = test_data['warranty'].astype('int64')
test_data['processor_gnrtn'] = test_data['processor_gnrtn'].astype('int64')


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
test_data['rating'] = label_encoder_rating.transform(test_data['rating'])


Y = test_data['rating']
X = test_data.drop('rating', axis=1)
scaled_data = scaler.transform(X)
scaled_df = pd.DataFrame(scaled_data, columns=X.columns)


all_selectedFeatures_df_test = pd.DataFrame()

for column in all_selectedFeatures_df_columns:
    all_selectedFeatures_df_test[column] = X[column]



#logistic
y_predict_test = logistic_model.predict(all_selectedFeatures_df_test)
print('Logistic Regression Accuracy Test = ',
      metrics.accuracy_score(np.asarray(Y), y_predict_test))

#SVC
y_predict_test = classifier.predict(all_selectedFeatures_df_test)
print('SVC Accuracy Test = ', metrics.accuracy_score(np.asarray(Y), y_predict_test))

#DT
y_predict_test = dt.predict(all_selectedFeatures_df_test)
test_accuracy_dt = metrics.accuracy_score(Y, y_predict_test)
print("DT Accuracy Test = ", test_accuracy_dt)

#RF
y_predict_test = logistic_model.predict(all_selectedFeatures_df_test)
test_accuracy_rf = metrics.accuracy_score(Y, y_predict_test)
print("DT Accuracy Test = ", test_accuracy_rf)

#KNN
y_predict_test = knn.predict(all_selectedFeatures_df_test)
test_accuracy_knn = metrics.accuracy_score(Y, y_predict_test)
print("KNN Accuracy Test = ", test_accuracy_knn)

#xgboost
y_predict_test = xgboost.predict(all_selectedFeatures_df_test)
test_accuracy_xgb = metrics.accuracy_score(Y, y_predict_test)
print("XGBOOST Accuracy Test = ", test_accuracy_xgb)

#adaboost
y_predict_test = ada.predict(all_selectedFeatures_df_test)
test_accuracy_ada = metrics.accuracy_score(Y, y_predict_test)
print("AdaBoost Accuracy Test = ", test_accuracy_ada)

#voting
y_predict_test = votingModel.predict(all_selectedFeatures_df_test)
test_accuracy_voting = metrics.accuracy_score(Y, y_predict_test)
print("Voting Accuracy Test = ", test_accuracy_voting)


#stacking
y_predict_test = stackingModel.predict(all_selectedFeatures_df_test)
test_accuracy_stacking = metrics.accuracy_score(Y, y_predict_test)
print("Stacking Accuracy Test = ", test_accuracy_stacking)