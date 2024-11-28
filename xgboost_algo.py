import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from category_encoders.hashing import HashingEncoder
from transforms import PropertyTypeTransformer, BathroomTextTransformer, DropColumnsTransformer, DateExtractorTransformer, DateToDaysSinceTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np  # Ensure numpy is imported

columns_drop = [
    'name', 'description', 'neighbourhood_cleansed', 'host_verifications',
    'amenities', 'has_availability', 'first_review', 'last_review', 'reviews'
]

num_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler())
])

one_hot_pipepine = Pipeline(steps=[
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessing = ColumnTransformer(transformers=[
    ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
    ('onehot',one_hot_pipepine, make_column_selector(dtype_include=object)),
    # ('target', target_encode_pipeline, target_encode_features)

])


airbnb_train = pd.read_csv('data/train.csv')
airbnb_test = pd.read_csv('data/test.csv')
train = airbnb_train.drop('price', axis=1)
y_label_train = airbnb_train['price']
test = airbnb_test


data_processing_pipeline = Pipeline(steps=[
    ('drop_columns', DropColumnsTransformer(columns_to_drop=columns_drop)),  # Drop specified columns
    ('extract_dates', DateToDaysSinceTransformer(date_columns=['host_since'])),  # Extract date
    ('transform_bathrooms_text', BathroomTextTransformer(column='bathrooms_text')),  # Transform 'bathrooms_text'
    ('transform_property_type', PropertyTypeTransformer(column='property_type')),
    ('preprocessing', preprocessing)  # Preprocessing pipeline
])

id = test['id']
Xtrain_processed = data_processing_pipeline.fit_transform(train)
Xtest_processed = data_processing_pipeline.transform(test)



def rounded_clipped_rmse(y_true, y_pred):
    # Round and clip predictions
    y_pred_rounded = np.round(y_pred)
    y_pred_clipped = np.clip(y_pred_rounded, 0, 5)
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    return rmse

# Define the custom scorer
scorer = make_scorer(rounded_clipped_rmse, greater_is_better=False)


# Define X and y
X = Xtrain_processed
y = y_label_train

# Define the base XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, colsample_bytree=0.8, learning_rate=0.01, max_depth=9, n_estimators=1050, reg_alpha=0, reg_lambda=1.0, subsample=0.8)
xgb_model.fit(X, y)
y_pred = xgb_model.predict(Xtest_processed)
y_pred_rounded = np.round(y_pred)
y_pred_final = np.clip(y_pred_rounded, 0, 5)

# Save the predictions to a CSV file
submission = pd.DataFrame({
    'id': id,   # Use the 'id' column from the test set
    'price': y_pred_final    # Predicted prices
})

# Save the submission file
submission.to_csv('xgb_algo.csv', index=False)

print("Submission file 'xgb_algo.csv' created successfully!")
