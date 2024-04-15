import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

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
print(data.isna().sum())

# for column in data.columns:
#     unique_values = data[column].unique()
#     print("Unique values in column '{}': {}".format(column, unique_values))

# Replacing Not Available in generation column
new_data['processor_gnrtn'] = new_data['processor_gnrtn'].replace("Not Available", pd.NA)
mode_value = new_data['processor_gnrtn'].mode()[0]
new_data['processor_gnrtn'] = new_data['processor_gnrtn'].fillna(mode_value)

for column in new_data.columns:
    unique_values = new_data[column].unique()
    print("Unique values in column '{}': {}".format(column, unique_values))

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
# print(data.describe())

new_data['Touchscreen_No'] = new_data['Touchscreen_No'].map({True: 1, False: 0})
new_data['Touchscreen_Yes'] = new_data['Touchscreen_Yes'].map({True: 1, False: 0})

new_data['msoffice_No'] = new_data['msoffice_No'].map({True: 1, False: 0})
new_data['msoffice_Yes'] = new_data['msoffice_Yes'].map({True: 1, False: 0})

new_data['weight_Casual'] = new_data['weight_Casual'].map({True: 1, False: 0})
new_data['weight_Gaming'] = new_data['weight_Gaming'].map({True: 1, False: 0})
new_data['weight_ThinNlight'] = new_data['weight_ThinNlight'].map({True: 1, False: 0})

# print(data.info())
new_data.drop(columns=['msoffice_No'], inplace=True)
new_data.drop(columns=['Touchscreen_No'], inplace=True)

# Brand , Processor brand, processor name, ram type, os --> Label Encoding
new_data['brand'] = label_encoder.fit_transform(new_data['brand'])
new_data['processor_brand'] = label_encoder.fit_transform(new_data['processor_brand'])
new_data['processor_name'] = label_encoder.fit_transform(new_data['processor_name'])
new_data['ram_type'] = label_encoder.fit_transform(new_data['ram_type'])
new_data['os'] = label_encoder.fit_transform(new_data['os'])

# Rating  --> Remove eccess string and convert to int


for column in new_data.columns:
    unique_values = new_data[column].unique()
    print("Unique values in column '{}': {}".format(column, unique_values))

new_data.to_csv('output.csv', index=False)

# Scaling
scaled_data = scaler.fit_transform(new_data)
scaled_df = pd.DataFrame(scaled_data, columns=new_data.columns)

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
# print(new_data.info())
# print(new_data.describe())
# Outliers

# Feature Selection
corr_data = new_data.iloc[:, :]
corr = corr_data.corr()
top_feature = corr.index[abs((corr['rating']) > 0.1)]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = corr_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
# top_feature = top_feature.drop('Performance Index')
