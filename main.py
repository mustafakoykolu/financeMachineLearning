from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.model_selection import cross_validate, train_test_split
from xgboost import XGBClassifier
import yfinance as yf


TICKER, START, END = "THYAO.IS", "2000-01-01", "2025-05-10"

df = yf.download(TICKER, start=START, end=END, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)  

df["PreviousClose"] = df["Close"].shift(1)
delta = df["Close"] - df["PreviousClose"]       

threshold = 0.005
df["Target"] = np.where(
    delta >  threshold,  1,         
    np.where(delta < -threshold, 2, 0)   
)

df = df.dropna(subset=["Target"])            
print(df.tail())

X = df.drop(columns=["Close", "Target"])  
y = df["Target"]          # 2 decrease , 0 no change, 1  rise


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=45,
                                                    stratify=y)

eval_set = [(X_test, y_test)]

xgb_cls = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=3,
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    tree_method="hist",
    early_stopping_rounds=50
)

xgb_cls.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],  # 2 set
    verbose=0
)

results    = xgb_cls.evals_result()
train_loss = results["validation_0"]["mlogloss"]   # eğitim
val_loss   = results["validation_1"]["mlogloss"]   # test


plt.figure(figsize=(10, 7))
plt.plot(train_loss, label="Training loss")
plt.plot(val_loss,   label="Validation loss")
plt.xlabel("Number of trees")
plt.ylabel("Log loss")
plt.legend()
plt.show()

cv_model = clone(xgb_cls)                
cv_model.set_params(early_stopping_rounds=None)  

cv_results = cross_validate(
    xgb_cls,        
    X, y,
    cv=10,
    scoring="f1_macro",
    return_train_score=True
)
print("CV f1:", cv_results["test_score"].mean())
