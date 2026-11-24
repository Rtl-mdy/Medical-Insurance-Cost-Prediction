from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

URL="https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv"
for d in ["data","models","outputs"]: Path(d).mkdir(exist_ok=True)
p=Path("data/medical_insurance_raw.csv")
df=pd.read_csv(p if p.exists() else URL)
if not p.exists(): df.to_csv(p,index=False)
df.columns=df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+","_",regex=True).str.strip("_")
required=["age","sex","bmi","children","smoker","region","charges"]
if not set(required).issubset(df.columns): raise ValueError(f"Expected columns: {required}")
for c in ["sex","smoker","region"]: df[c]=df[c].astype("string").str.strip().str.lower()
for c in ["age","bmi","children","charges"]: df[c]=pd.to_numeric(df[c],errors="coerce")
before=len(df); duplicates=int(df.duplicated().sum()); df=df.drop_duplicates().dropna(subset=["charges"]).copy()
df.to_csv("data/medical_insurance_cleaned.csv",index=False)
X=df.drop(columns="charges"); y=df["charges"]
num=["age","bmi","children"]; cat=["sex","smoker","region"]
prep=ColumnTransformer([
 ("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
candidates={"Linear Regression":LinearRegression(),
 "Decision Tree":DecisionTreeRegressor(max_depth=5,random_state=42),
 "Random Forest":RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=-1),
 "Gradient Boosting":GradientBoostingRegressor(random_state=42)}
pipes=[]; metrics=[]
for name,model in candidates.items():
 pipe=Pipeline([("preprocessing",prep),("regressor",model)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
 metrics.append({"Model":name,"MAE":mean_absolute_error(yte,pred),"MSE":mean_squared_error(yte,pred),
 "RMSE":np.sqrt(mean_squared_error(yte,pred)),"R2":r2_score(yte,pred)})
 pipes.append(pipe)
result=pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)
result.to_csv("outputs/model_comparison.csv",index=False)
best_name=result.loc[0,"Model"]; best=pipes[[x[0] for x in candidates.items()].index(best_name)]
joblib.dump(best,"models/best_insurance_model.joblib")
Path("models/model_metadata.json").write_text(json.dumps({"best_model":best_name,"target":"charges","rows":len(df)},indent=2))
print("Original rows:",before,"| duplicates removed:",duplicates,"| cleaned rows:",len(df))
print(result.to_string(index=False)); print("Saved cleaned CSV, metrics, and best model.")
