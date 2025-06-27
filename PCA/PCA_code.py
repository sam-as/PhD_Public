'''
Created Date: Tuesday June 17th 2025
Author: Avik Sam
Webpage: https://sites.google.com/view/aviksam
'''

#%%
# importing the libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# reading the data
# define the filepath
fpath = ''
df = pd.read_excel(fpath)

#%%
def outlier_remover(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_noout = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return data_noout

df1 = df.apply(outlier_remover, 'col')

#%%
# setting up the data
drop_cols = []

X = df1.drop(drop_cols, axis = 1)
X = X.fillna(X.median())
n_sample=len(X) 
n_features= len(X.columns)

#getting the number of components
x= range(1,(min(n_sample, n_features)))

#normalising the data
X_normalised=StandardScaler().fit_transform(X) 

add=0

variance_list=[] 
for i in x:
    pca=PCA(n_components=i)
    pca_con=pca.fit_transform(X_normalised)
    var=pca.explained_variance_
    variance_list.append(var)

z= var

data={'x':x, 'z':z} 

df_=pd.DataFrame(data, columns=['x' ,'z'])

#initialising matplotlib
sns.set_theme(style = 'ticks', font_scale= 1.1)
fig, ax=plt.subplots(figsize = (6, 4))

sns.pointplot(x='x', y='z', data=df_, color='teal', ax=ax)

plt.xlabel('Number of Principal Components', weight = 'demi')
plt.ylabel('Explained Variance', weight = 'demi')
ax.set_title('Skree Plot', weight = 'demi')
ax.legend(labels=['Eigen values'], labelcolor='black', fontsize='small')
ax.set_xlim(0, 12)

plt.axhline(y=1, color='red', 
                linewidth = 1.5,
                linestyle = '--')

plt.grid(alpha = 0.2)
plt.tight_layout()
spath2 = r"" #define the savepath
plt.savefig(spath2, dpi = 300)
plt.show()

#%% Plot 2
##### Number of components: n ####
pca=PCA(n_components=4) # Select the number of components
    
pca_con=pca.fit_transform(X_normalised)
pca_con_df=pd.DataFrame(data=pca_con , columns=['PC1', "PC2", "PC3", 'PC4'])

factor_loadings=pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X.columns)

# Visualise the Loadings from each component
fig, ax=plt.subplots(figsize=(6,9))

map= pd.DataFrame(pca.components_, columns=X.columns)
mdf = map.T
mdf.columns = [f'PCA {i}' for i in range(1,5)]

# define the colormap
cpal  = 'PRGn_r' # change it if you want

sns.heatmap(mdf, cmap=cpal,  
        linewidths=1, 
        vmin = -0.5, 
        vmax = 0.5,
        cbar_kws={"shrink":.5},
        ax=  ax)

# ax.set_yticklabels(f'PCA {i}' for i in range(1,4))
ax.set_xlabel("Principal Component", weight = 'demi')
ax.set_ylabel('Fractions', weight = 'demi')

plt.tight_layout()
spath2 = r"" #this file should end in .png
plt.savefig(spath2, dpi = 300)
plt.show()

#%% exporting the variances
spath3 = r"" # This should have an .xlsx
pca_con_df.to_excel(spath3)

