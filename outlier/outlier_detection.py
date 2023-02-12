
#%% importing the library and reading the data

import pandas as pd
df=pd.read_excel(r"///////////////2.xlsx")
print(df)

#%% setting the context

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)
df.columns

data=df

#%%outlier detection
def outlier_remover(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    outliers=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR)))]
    return outliers

#%% exporting the data
outlier_remover(data).to_excel(r"/////////////outliers.xlsx")
#outlier_remover(data)
# %%
