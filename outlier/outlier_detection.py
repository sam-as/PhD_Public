
#%% importing the library and reading the data

import pandas as pd
df=pd.read_excel(r"C:\Users\aviks\OneDrive\Desktop\MSc Projects\Stage 3\NMR\Final\water_methanol_final_combined_07Feb22.xlsx")
print(df)

#%% setting the context

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)
df.columns

#%% indexing the required parameters
#data=df[['H-C', 'H-C-C=', 'H-C-O', 'H-Ar', 'H-C-C=O', 'S', 'H-C(t)', 'H-C-C=(t)',
       #'H-C-O(t)', 'H-Ar(t)', 'H-C-C=O(t)', 'S(t)']]

data=df

#%%outlier detection
def outlier_remover(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    outliers=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR)))]
    return outliers

#%% exporting the data
outlier_remover(data).to_excel(r"C:\Users\aviks\OneDrive\Desktop\MSc Projects\Stage 3\NMR\Final\outliers.xlsx")
#outlier_remover(data)
# %%
