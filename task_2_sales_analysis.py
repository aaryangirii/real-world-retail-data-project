from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
ROOT=Path(__file__).resolve().parent; out=ROOT/"outputs"; source=out/"retail_sales_clean.csv"
if not source.exists(): raise FileNotFoundError("Run task_1_prepare_data.py first.")
data=pd.read_csv(source,parse_dates=["order_date"]); sns.set_theme(style="whitegrid")
products=data.groupby("product",as_index=False).agg(total_sales=("sales","sum"),total_profit=("profit","sum"),units_sold=("quantity","sum"),orders=("product","size")).sort_values("total_sales",ascending=False)
regions=data.groupby("region",as_index=False).agg(total_sales=("sales","sum"),total_profit=("profit","sum")).sort_values("total_sales",ascending=False)
products.to_csv(out/"product_summary.csv",index=False); regions.to_csv(out/"region_summary.csv",index=False)
for name,summary,ycol,title in [("sales_by_product.png",products,"total_sales","Total Sales by Product"),("profit_by_region.png",regions,"total_profit","Estimated Profit by Region")]:
    fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(data=summary,x=summary.columns[0],y=ycol,ax=ax); ax.set_title(title); fig.tight_layout(); fig.savefig(out/name,dpi=150); plt.close(fig)
daily=data.groupby("order_date",as_index=False)["sales"].sum()
fig,ax=plt.subplots(figsize=(10,5)); ax.plot(daily.order_date,daily.sales); ax.set(title="Daily Sales Trend",xlabel="Date",ylabel="Sales"); fig.tight_layout(); fig.savefig(out/"daily_sales_trend.png",dpi=150); plt.close(fig)
report=f"RETAIL SALES SUMMARY\nOrders: {len(data)}\nTotal sales: {data.sales.sum():,.2f}\nEstimated profit: {data.profit.sum():,.2f}\nTop product: {products.iloc[0]['product']}\nTop region by sales: {regions.iloc[0]['region']}\n\nSynthetic practice data; validate findings against real data."
(out/"sales_summary_report.txt").write_text(report,encoding="utf-8"); print(report)
