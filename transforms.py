import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from category_encoders.hashing import HashingEncoder


from sklearn.base import BaseEstimator, TransformerMixin

#transform property type
class PropertyTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='property_type', top_n=3):
        self.column = column
        self.top_n = top_n
        self.top_categories_ = None

    def fit(self, X, y=None):
        # Identify the top N categories
        self.top_categories_ = (
            X[self.column]
            .value_counts()
            .nlargest(self.top_n)
            .index
            .tolist()
        )
        return self

    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = X[self.column].apply(self._group_categories)
        return X

    def _group_categories(self, value):
        if value in self.top_categories_:
            return value
        else:
            return 'Other'


#transform bathroom text
class BathroomTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='bathrooms_text'):
        self.column = column

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = X[self.column].apply(self._transform_bathroom_text)
        return X

    def _transform_bathroom_text(self, value):
        if pd.isnull(value):
            return 'unspecified'
        value = str(value).lower()
        if 'private' in value:
            return 'private'
        elif 'shared' in value:
            return 'shared'
        else:
            return 'unspecified'

#tranform bathroom text
class BathroomTextTFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='bathrooms_text'):
        self.column = column

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = X[self.column].apply(self._transform_bathroom_text)
        return X

    def _transform_bathroom_text(self, value):
        if pd.isnull(value):
            return 'unspecified'
        value = str(value).lower()
        if 'private' in value:
            return 'private'
        elif 'shared' in value:
            return 'shared'
        else:
            return 'unspecified'
        
#drop columns
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns_to_drop:
            return X.drop(columns=self.columns_to_drop, errors='ignore')
        return X
    
#extract year from host_since
class DateExtractorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns=None):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_columns:
            for col in self.date_columns:
                if col in X.columns:
                    X[col] = pd.to_datetime(X[col]).dt.year.astype(str)
        return X

#extract date to days since
class DateToDaysSinceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns=None):
        self.date_columns = date_columns
        self.earliest_dates = {}

    def fit(self, X, y=None):
        if self.date_columns:
            for col in self.date_columns:
                if col in X.columns:
                    X_col = pd.to_datetime(X[col])
                    self.earliest_dates[col] = X_col.min()
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_columns:
            for col in self.date_columns:
                if col in X.columns and col in self.earliest_dates:
                    X_col = pd.to_datetime(X[col])
                    X[col] = (X_col - self.earliest_dates[col]).dt.days
        return X
    

#convert boolean to integer
class BoolToIntTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # Accept a list of column names to transform

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns:
            for col in self.columns:
                if col in X.columns:
                    # Convert column to boolean if it's not already
                    if X[col].dtype != 'bool':
                        X[col] = bool(X[col])
                    # Convert boolean values to integers (True->1, False->0)
                    X[col] = X[col].astype(int)
        return X
