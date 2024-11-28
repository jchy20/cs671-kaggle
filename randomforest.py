import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from category_encoders.hashing import HashingEncoder
from transforms import PropertyTypeTransformer, BathroomTextTransformer, DropColumnsTransformer, DateExtractorTransformer, DateToDaysSinceTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

columns_drop = [
    'name', 'description', 'neighbourhood_cleansed', 'host_verifications',
    'amenities', 'has_availability', 'first_review', 'last_review', 'reviews'
]

num_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler())
])

one_hot_pipeline = Pipeline(steps=[
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessing = ColumnTransformer(transformers=[
    ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
    ('onehot', one_hot_pipeline, make_column_selector(dtype_include=object)),
])

airbnb_train = pd.read_csv('data/train.csv')
airbnb_test = pd.read_csv('data/test.csv')
train = airbnb_train.drop('price', axis=1)
y_label_train = airbnb_train['price']
test = airbnb_test

data_processing_pipeline = Pipeline(steps=[
    ('drop_columns', DropColumnsTransformer(columns_to_drop=columns_drop)),
    ('extract_dates', DateToDaysSinceTransformer(date_columns=['host_since'])),
    ('transform_bathrooms_text', BathroomTextTransformer(column='bathrooms_text')),
    ('transform_property_type', PropertyTypeTransformer(column='property_type')),
    ('preprocessing', preprocessing)
])

id = test['id']
Xtrain_processed = data_processing_pipeline.fit_transform(train)
Xtest_processed = data_processing_pipeline.transform(test)

# Define the custom scorer
def rounded_clipped_rmse(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)
    y_pred_clipped = np.clip(y_pred_rounded, 0, 5)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    return rmse

scorer = make_scorer(rounded_clipped_rmse, greater_is_better=False)

# Define X and y
X = Xtrain_processed
y = y_label_train

# Define the Random Forest regressor
rf_model = RandomForestRegressor(random_state=42, bootstrap=False, max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=300)
rf_model.fit(X, y)
y_pred = rf_model.predict(Xtest_processed)
y_pred_rounded = np.round(y_pred)
y_pred_final = np.clip(y_pred_rounded, 0, 5)

# Save the predictions to a CSV file
submission = pd.DataFrame({
    'id': id,   # Use the 'id' column from the test set
    'price': y_pred_final    # Predicted prices
})

# Save the submission file
submission.to_csv('rf.csv', index=False)

print("Submission file 'rf.csv' created successfully!")