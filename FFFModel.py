import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import numpy as np

TICKER = input("Enter ticker (e.g. AAPL): ")
START = "2000-01-01"
END = None

#monthly returns 
raw = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=False)

# If yfinance returned MultiIndex columns (e.g., ('Adj Close','AAPL')), flatten them

raw.columns = raw.columns.get_level_values(0)

# Keep Adjusted Close as a single column named 'price'
px = raw[["Adj Close"]].rename(columns={"Adj Close": "price"})

# Month-end % returns, force single, flat column name
px_m  = px.resample("ME").last()
ret_m = px_m.pct_change().dropna()
ret_m.columns = ["asset_ret"]          


#Fama–French factors
ff = pd.read_csv("F-F_Research_Data_Factors.csv")
ff["date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
ff = ff[["date", "Mkt-RF", "SMB", "HML", "RF"]]
for c in ["Mkt-RF", "SMB", "HML", "RF"]:
  ff[c] = pd.to_numeric(ff[c], errors="coerce") / 100.0
ff = ff.set_index("date").sort_index()

#Join by month-end date 
data = ret_m.join(ff, how="inner")
data["asset_excess"] = data["asset_ret"] - data["RF"]

model_df = data[["asset_excess", "Mkt-RF", "SMB", "HML"]].dropna()


#OLS regression
Y = model_df["asset_excess"]
X = model_df[["Mkt-RF", "SMB", "HML"]]
X = sm.add_constant(X) # alpha

ols = sm.OLS(Y, X).fit()

avg_factors = model_df[["Mkt-RF", "SMB", "HML"]].mean()
print(avg_factors)

# performance attribution
avg_factors = model_df[["Mkt-RF", "SMB", "HML"]].mean()

contrib = {
  "Alpha (monthly)": ols.params["const"],
  "Mkt-RF": ols.params["Mkt-RF"] * avg_factors["Mkt-RF"],
  "SMB":    ols.params["SMB"]    * avg_factors["SMB"],
  "HML":    ols.params["HML"]    * avg_factors["HML"],
}

contrib_annual = {k: v*12 for k, v in contrib.items()}

print("\nMonthly Contributions (%):")
for k, v in contrib.items():
  print(f"{k:12s}: {v:.4%}")

print("\nAnnual Contributions (%):")
for k, v in contrib_annual.items():
  print(f"{k:12s}: {v:.2%}")

# graph
import matplotlib.pyplot as plt

plt.bar(contrib_annual.keys(), contrib_annual.values())
plt.title(TICKER+" Return Attribution (Annualised)")
plt.ylabel("Contribution to Return")
plt.axhline(0, color="black", linewidth=0.8)
plt.show()

# analyst report
# === Analyst Report (printed to console) ===
alpha_ann = (1 + ols.params["const"])**12 - 1
beta_mkt  = ols.params["Mkt-RF"]
beta_smb  = ols.params["SMB"]
beta_hml  = ols.params["HML"]
r2        = ols.rsquared
tvals     = ols.tvalues

mkt_y = contrib_annual["Mkt-RF"]
smb_y = contrib_annual["SMB"]
hml_y = contrib_annual["HML"]
alp_y = contrib_annual["Alpha (monthly)"]

report = f"""
===========================================
Fama–French 3-Factor Analysis for {TICKER}
===========================================

1) Data
- Asset: {TICKER} monthly returns (Yahoo Finance)
- Sample: {START} to latest
- Factors: US Fama–French 3 Factors (Kenneth French Data Library)

2) Method
OLS regression of excess returns on:
  R_i - R_f = α + β_MKT(Mkt-RF) + β_SMB·SMB + β_HML·HML + ε

3) Results
- Alpha (monthly): {ols.params['const']:.2%} 
- Alpha (annual, comp): {alpha_ann:.2%} 
- Beta (Mkt-RF): {beta_mkt:.2f}   (t={tvals['Mkt-RF']:.2f})
- Beta (SMB):    {beta_smb:.2f}   (t={tvals['SMB']:.2f})
- Beta (HML):    {beta_hml:.2f}   (t={tvals['HML']:.2f})
- R²: {r2:.2f}

4) Performance Attribution (annualised)
- Market (Mkt-RF): {mkt_y:.2%}
- SMB:             {smb_y:.2%}
- HML:             {hml_y:.2%}
- Alpha:           {alp_y:.2%}

5) Interpretation
- Style: {'Growth tilt (HML < 0)' if beta_hml < 0 else 'Value tilt (HML > 0)'}
- Size: {'Large-cap tilt (SMB < 0)' if beta_smb < 0 else 'Small-cap tilt (SMB > 0)'}
- Market sensitivity: β ≈ {beta_mkt:.2f}
- Alpha: {'Positive and significant' if abs(tvals['const'])>=2 else 'Not clearly significant'}

============================================
"""

print(report)

