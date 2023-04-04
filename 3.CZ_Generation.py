#Sam Roy developed this code to identify the ideal number of clusters that ensure optimal intercluster variance under numerous different random seeds
#November 2020
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_samples, silhouette_score
import sys

#user defined sample size, a good bet is 70*number of variables (39) = round up 3000
samp = int(sys.argv[1])
#DATA IMPORT 1#################################################
df = pd.read_csv(r"D:\inspires\data\ClimateNA\out\1961-1990.csv")
id = pd.read_csv(r"D:\inspires\data\ClimateNA\ClimateNA_Reference\ClimateNA_ID.csv")

#CLEAN NaNs########################################################
imp_mean = SimpleImputer(missing_values=-9999.0, strategy='mean')
imp_mean.fit(df)
dfi = pd.DataFrame(imp_mean.transform(df))
df = dfi.rename(columns=dict(zip(dfi.columns,df.columns)))

#SEASONAL AVG MONTHLY DATA#################################################
#brings# variablesdown from 75 to 39
df['Tmin_sp']=df.filter(['Tmin03', 'Tmin04', 'Tmin05'],axis=1).mean(axis=1)
df['Tmin_su']=df.filter(['Tmin06', 'Tmin07', 'Tmin08'],axis=1).mean(axis=1)
df['Tmin_fa']=df.filter(['Tmin09', 'Tmin10', 'Tmin11'],axis=1).mean(axis=1)
df['Tmin_wi']=df.filter(['Tmin12', 'Tmin01', 'Tmin02'],axis=1).mean(axis=1)

df['Tmax_sp']=df.filter(['Tmax03', 'Tmax04', 'Tmax05'],axis=1).mean(axis=1)
df['Tmax_su']=df.filter(['Tmax06', 'Tmax07', 'Tmax08'],axis=1).mean(axis=1)
df['Tmax_fa']=df.filter(['Tmax09', 'Tmax10', 'Tmax11'],axis=1).mean(axis=1)
df['Tmax_wi']=df.filter(['Tmax12', 'Tmax01', 'Tmax02'],axis=1).mean(axis=1)

df['Tave_sp']=df.filter(['Tave03', 'Tave04', 'Tave05'],axis=1).mean(axis=1)
df['Tave_su']=df.filter(['Tave06', 'Tave07', 'Tave08'],axis=1).mean(axis=1)
df['Tave_fa']=df.filter(['Tave09', 'Tave10', 'Tave11'],axis=1).mean(axis=1)
df['Tave_wi']=df.filter(['Tave12', 'Tave01', 'Tave02'],axis=1).mean(axis=1)

df['PPT_sp']=df.filter(['PPT03', 'PPT04', 'PPT05'],axis=1).mean(axis=1)
df['PPT_su']=df.filter(['PPT06', 'PPT07', 'PPT08'],axis=1).mean(axis=1)
df['PPT_fa']=df.filter(['PPT09', 'PPT10', 'PPT11'],axis=1).mean(axis=1)
df['PPT_wi']=df.filter(['PPT12', 'PPT01', 'PPT02'],axis=1).mean(axis=1)

df=df.drop(['Tmin01','Tmin02','Tmin03','Tmin04','Tmin05','Tmin06','Tmin07',
'Tmin08','Tmin09','Tmin10','Tmin11','Tmin12'], axis=1)

df=df.drop(['Tmax01','Tmax02','Tmax03','Tmax04','Tmax05','Tmax06','Tmax07',
'Tmax08','Tmax09','Tmax10','Tmax11','Tmax12'], axis=1)

df=df.drop(['Tave01','Tave02','Tave03','Tave04','Tave05','Tave06','Tave07',
'Tave08','Tave09','Tave10','Tave11','Tave12','Tave_sm','Tave_wt'], axis=1)

df=df.drop(['PPT01','PPT02','PPT03','PPT04','PPT05','PPT06','PPT07',
'PPT08','PPT09','PPT10','PPT11','PPT12','PPT_sm','PPT_wt'], axis=1)

#df = df.replace(-9999.0,np.NaN) #MAR has some NaNs defined with a numeric value

#STANDARDIZE AND FIT THE PCA############################################
df=df.set_index('ID')
id=id.set_index('ID')
svals=StandardScaler().fit_transform(df)

pca=PCA(0.95) #FYI, 5 PCs explain 96% of all variance
pcs=pca.fit_transform(svals)

pcdf = pd.DataFrame(data = pcs, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
pcdf=pcdf.set_index(df.index)

n_clust = 24
n_seed = 50
sil_matrix = np.zeros([n_seed,n_clust])
for i in range(2,n_clust):
    for j in range(n_seed):
        try:
            print('cluster %s, seed %s' %(i,j))
            pcsample = pcdf.sample(samp, random_state = j) #Dolnicar et al. 2014 recommend 70* the number of variables, mst conservative sampling approach suggested
            #ward agglomerative clustering is used by Briggs and Lemin
            model = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward').fit(pcsample)
            sil_matrix[j,i] = silhouette_score(pcsample, model.labels_)
        except:
            print('error on cluster %s, seed %s' %(i,j))
            import pdb; pdb.set_trace()

sil_matrix = pd.DataFrame(sil_matrix)
sil_matrix.rename(columns=dict(zip(sil_matrix.columns,list(range(2,n_clust)))))
sil_matrix.to_csv(r"D:\inspires\data\ClimateNA\out\sil_matrix %s.csv" %samp)
