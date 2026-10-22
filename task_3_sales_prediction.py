from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parent; out=ROOT/"outputs"; source=out/"retail_sales_clean.csv"
if not source.exists(): raise FileNotFoundError("Run task_1_prepare_data.py first.")
data=pd.read_csv(source)
features=["product","region","quantity","unit_price","discount_rate"]; X=data[features]; y=data["sales"]
prep=ColumnTransformer([("category",OneHotEncoder(handle_unknown="ignore"),["product","region"]),("number","passthrough",["quantity","unit_price","discount_rate"])])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
models={"Mean baseline":Pipeline([("prep",prep),("model",DummyRegressor(strategy="mean"))]),"Random Forest":Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42))])}
results={}; predictions={}
for name,model in models.items():
    model.fit(X_train,y_train); pred=model.predict(X_test); predictions[name]=pred
    results[name]={"MAE":round(mean_absolute_error(y_test,pred),2),"RMSE":round(mean_squared_error(y_test,pred)**.5,2),"R2":round(r2_score(y_test,pred),3)}
(out/"sales_model_metrics.json").write_text(json.dumps(results,indent=2))
best=min(results,key=lambda k:results[k]["MAE"]); fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(y_test,predictions[best],alpha=.75)
low=min(float(y_test.min()),float(predictions[best].min())); high=max(float(y_test.max()),float(predictions[best].max())); ax.plot([low,high],[low,high],"--")
ax.set(title=f"Actual vs Predicted Sales — {best}",xlabel="Actual sales",ylabel="Predicted sales"); fig.tight_layout(); fig.savefig(out/"sales_actual_vs_predicted.png",dpi=150); plt.close(fig)
print(json.dumps(results,indent=2)); print("Best by MAE on this test split:",best)
