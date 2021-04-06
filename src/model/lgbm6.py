import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

train_df = pd.read_pickle("../feature/creatinine_train_df.pkl")

Y_train = train_df["target"]
X_train = train_df.drop("target",axis=1)

print(Y_train.head())
print(X_train.head())

train_x, val_x, train_y, val_y = train_test_split(X_train, Y_train, test_size=0.2)

train_data = lgb.Dataset(train_x, label=train_y)
val_data = lgb.Dataset(val_x, label=val_y, reference=train_data)

params = {
    'boosting_type':'gbdt',
    'objective':'binary'
}

gbm = lgb.train(
    params,
    train_data,
    valid_sets=val_data,
    num_boost_round=1000,
    verbose_eval=10,
    early_stopping_rounds=100
)

gbm.save_model('lgbm6.txt')

y_pred = gbm.predict(val_x, num_iteration=gbm.best_iteration)

print(y_pred)

print("accuracy : ", accuracy_score(val_y.values.tolist(), y_pred.round(0).tolist()))