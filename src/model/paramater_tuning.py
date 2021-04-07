import pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

train_df = pd.read_pickle("../feature/CPK_train_df.pkl")

Y_train = train_df["target"]
X_train = train_df.drop("target",axis=1)

model = lgb.LGBMClassifier(silent=False)

param_grid = {
    "n_estimators":[100,200,500],
    "learning_rate":[0.001,0.01,0.05,0.1],
    "num_leaves":[7,15,31],
    "max_depth":[5,7,9],
    "min_data_in_leaf":[20,30,50],
    #"bagging_fraction":[0.8,0.9],
    #"bagging_freq":[1,3],
    #"feature_fraction":[0.9,1.0],
}

skf = StratifiedKFold(n_splits=10,
                      shuffle=True,
                      random_state=0)

grid_result = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'accuracy',
    cv=skf,
    verbose=3,
    return_train_score=False,
    n_jobs=-1
)

grid_result.fit(X_train,Y_train)

print("[finish] : this is best parameter.")
print(grid_result.best_params_)