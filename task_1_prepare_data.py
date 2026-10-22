from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; out=ROOT/"outputs"; out.mkdir(exist_ok=True)
data=pd.read_csv(ROOT/"data/retail_sales.csv",parse_dates=["order_date"])
print("Original shape:",data.shape,"\nMissing values:\n",data.isna().sum())
data["region"]=data["region"].fillna(data["region"].mode().iloc[0])
data=data.drop_duplicates().copy()
data["quantity"]=data["quantity"].clip(lower=0)
data["sales"]=(data["unit_price"]*data["quantity"]*(1-data["discount_rate"])).round(2)
data.to_csv(out/"retail_sales_clean.csv",index=False,date_format="%Y-%m-%d")
print("Remaining missing values:\n",data.isna().sum()); print("Saved cleaned data to",out/"retail_sales_clean.csv")
