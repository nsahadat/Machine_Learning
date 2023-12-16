import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from category_encoders import *
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
sys.path.append('./Data')

df = pd.read_csv('./Data/train.csv')
print(df.shape)

cont_col = []
cat_col = []
for c in df.columns:
#     print(f'{c}:{df[c].dtypes}')
    if c not in ['SalePrice', 'Id']:
        if df[c].dtypes== 'object':
            cat_col.append(c)
        else:
            cont_col.append(c)

print(f'cat_col: {cat_col}')
print(f'con_col: {cont_col}')

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder(handle_missing=np.nan, handle_unknown="value"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, cont_col),
        ("cat", categorical_transformer, cat_col),
    ]
)

# clf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0)
clf = xgb.XGBRegressor(n_estimators=150, max_depth=15, eta=0.1, subsample=0.7, colsample_bytree=0.8)

params = {'n_estimators':np.arange(200, 500, 10),
        'max_depth': np.arange(5, 20, 1)
         }

skf = StratifiedKFold(n_splits=2, shuffle= True, random_state= 17)

best_clf = GridSearchCV(estimator= clf, param_grid= params, scoring = 'neg_root_mean_squared_error',
                         cv= skf, verbose= True, n_jobs= -1)

pipeline_ = Pipeline([('preprocessor', preprocessor),
#                      ('regressor', clf)]
                     ('regressor', best_clf)
                    ])

# pipeline.fit(df[cont_col+cat_col], df['SalePrice'])
pipeline_.fit(df[cont_col+cat_col], df['SalePrice'])

print(pipeline_.named_steps['regressor'].best_params_)

print(pipeline_.named_steps['regressor'].best_score_)

print(pipeline_.named_steps['regressor'].best_estimator_)

pipeline = Pipeline([('preprocessor', preprocessor),
                     ('regressor', pipeline_.named_steps['regressor'].best_estimator_)])

pipeline.fit(df[cont_col+cat_col], df['SalePrice'])

df['y_est'] = pipeline.predict(df[cont_col+cat_col])

df['error'] = abs(df['y_est']-df['SalePrice'])

print(df[['SalePrice', 'y_est', 'error']])

mean_squared_error(df['SalePrice'], df['y_est'])


# Test Score generation
df_test = pd.read_csv('./Data/test.csv')
df_test['SalePrice'] = pipeline.predict(df_test[cont_col+cat_col])
df_test[['Id', 'SalePrice']].to_csv('submission.csv',index=False)