'''
Regression Model

Analyze data
then convert the data to use in numerical values through one-hot encoding
define dependent and independent varibles based on your  problem statement or model you want to build
then train test split on independent and dependent variable -- evaluation approach
then train your about 80 percent of your data
then test or validate your data using, test_accuracy, mean_absolute_error, mean_squared_error
keep tweaking it till you get a good validation to make better prediction even on unknown data 
without overfitting and underfitting validation

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#analysis
data = pd.read_csv('insurance.csv')
data.head()
data.shape
data.isna().sum()
data.dropna(inplace=True)
data['col'].str.replace('replaced value', 'what to replace with', regex=False)
data.describe
data.duplicated().sum()
data.drop_duplicates
data.dropna or np.mean(data['col'])

# Define a function to check if all elements have the same type
def check_same_type(column):
    return all(type(elem) == type(column.iloc[0]) for elem in column)


data.value_counts('sex')
sns.countplot(data=data, x='sex')

#converting to column dtype that's date to datetime if not so
data['col'] = pd.to_datetime(data['col'])
data.col.dt.year


#regression
from sklearn import preprocessing
labelenc = preprocessing.LabelEncoder()
data['sex'] = labelenc.fit_transform(data['sex'])

data.corr()['charges'].sort_values(ascending=False)

x = data[['sex', 'age', 'gender']]
y = data['charges']
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaled_x_train = scaler.fit_transform(train_x)
# scaled_x_test = scaler.fit_transform(test_x)

# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(scaled_x_train, train_y)

# from sklearn.metrics import mean_absolute_error
# data_model = data.predict()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
data_model = RandomForestRegressor(random_state=1)
data_model.fit(train_x, train_y)
data_preds = data_model.predict(train_x)
print(mean_absolute_error(data_preds, test_y))

#or -- to capture multiple mae and take the lowest
def get_mae(max_leaf_nodes, train_x, test_x, train_y, test_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random=1)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    mae = mean_absolute_error(pred, test_y)
    return (mae)

for max_lead_nodes in [5, 50, 500, 5000]:
    the_mae = get_mae(max_lead_nodes, train_x, test_x, train_y, test_y)
    print("max leaf nodes %d \t\t mean absolute error: %d "%(max_lead_nodes, the_mae))


def get_mae(max_leaf_nodes, train_x, test_x, train_y, test_y):
    model = RandomForestRegressor(max_leaf_nodes, random_state=1)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    mae = mean_absolute_error(pred, train_y)
    return (mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    the_mae = get_mae(max_leaf_nodes, train_x, test_x, train_y, test_y)
    print("the max lead nodes of %d \t\t has mean absolute error of %d" %(max_lead_nodes, the_mae))

#hadnling missing values
# - 1. drop null
drop_missing_values = [col for col in train_x.columns if train_x[col].isna().any()]

reduce_train_x = train_x.drop(drop_missing_values, axis='columns')
reduce_test_x = test_x.drop(drop_missing_values, axis='columns')
print(score_dataset(reduced_train_x, reduced_test_x, train_y, test_y))

#2. impute
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_x))
imputed_x_val =  pd.DataFrame(my_imputer.transform(test_x))

#the drop columns, bring them back by doing the following
imputed_x_train.columns = train_x.columns
imputed_x_test = test_x.columns

#categorical vars - drop categorical | ordinal encoding | one-hot encoding
s = (train_x.dtypes == 'object')
obj_cols = (list(s[s].index)) # to find out how many cols are cat

#drop categorical
drop_x_train = train_x.select_dtypes(exclude=['object'])
drop_y_train = train_y.select_dtypes(exclude=['object'])

print(score_dataset(drop_x_train, drop_y_train, train_x, train_y))

#one_hot encoding
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 
OH_colsTrain = pd.DataFrame(OH_encoder.fit_transform(train_x[obj_cols]))
OH_colsValid = pd.DataFrame(OH_encoder.transform(test_x[obj_cols]))

# pipeline

"""
1. import pipeline from pipeline and columntransformer from compose; standardscaler and onehotencoder from preprocessing
2. define num and cats stating the list of cols for each
3. define preprocessing steps with pipeline for nums(imputer and standardscaler) and cats (imputer, onehot using
 most frequent as strategy)
4. bundle them together using columntransformer
5. define final pipeline stating the preprocessor and the regressor/classifier you're using
6. pipeline.fit(...) -- fit pipeline
7. make predictions

short:
numerical
categorical
preprocessor
pipeline
predict
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

#define cat and num cols
num_cols = ['col1', 'col2']
cat_cols = ['col3', 'col4']

#define preprocessing steps for cat and num cols
numerical_transformer = Pipeline(steps=[
    'imputer', SimpleImputer(strategy='mean'),
    'scaler', StandardScaler()
])

categorical_transformer = Pipeline(steps=[
    'imputer', SimpleImputer(strategy='most_frequent'),
    'onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)
])

preprocessing = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

pipeline = Pipeline(steps=[
    'preprocessor', preprocessing,
    'regressor', RandomForestRegressor(n_estimators=100, random_state=1)
])

pipeline.fit(train_x, train_y)
pipeline.predict(train_x)

#cross validation

from sklearn.model_selection import cross_val_score
scores = -1 * cross_val_score(pipeline, x, y, cv=5, scoring='neg_mean_absolute_error')
print('Avg MAE ', scores.mean())

scores = -1 * cross_val_score(pipeline, x, y, cv=4, scoring='neg_mean_absolute_error')

#XGBoost

from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, )
model.fit(train_x, train_y, early_stopping_round=5, eval_set=[(train_x, train_y)], verbose=False)


#to handle train-test contamination
data = pd.read_csv('ddddd.csv')

x = data.drop('your_target', axis=1)
y = data['your_target']

