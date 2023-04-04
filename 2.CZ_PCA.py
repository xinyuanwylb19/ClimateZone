# PCA analysis

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score


#DATA IMPORT 1#################################################
df = pd.read_csv(r"D:\inspires\data\ClimateNA\out\1961-1990.csv")
id = pd.read_csv(r"D:\inspires\data\ClimateNA\ClimateNA_Reference\ClimateNA_ID.csv")

#CLEAN NaNs########################################################
from sklearn.impute import SimpleImputer
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

df['PPT_sp']=df.filter(['PPT03', 'PPT04', 'PPT05'],axis=1).sum(axis=1)
df['PPT_su']=df.filter(['PPT06', 'PPT07', 'PPT08'],axis=1).sum(axis=1)
df['PPT_fa']=df.filter(['PPT09', 'PPT10', 'PPT11'],axis=1).sum(axis=1)
df['PPT_wi']=df.filter(['PPT12', 'PPT01', 'PPT02'],axis=1).sum(axis=1)

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

scaler = StandardScaler()
svals=pd.DataFrame(scaler.fit_transform(df))
svals=svals.rename(columns=dict(zip(svals.columns,df.columns)))
svals=svals.set_index(df.index.astype(int))

#pca=PCA(n_components=2)
pca=PCA(0.95) #FYI, 4 PCs explain 96% of all variance
pcs=pca.fit_transform(svals)

#pcdf = pd.DataFrame(data = pcs, columns = ['pc1', 'pc2'])
pcdf = pd.DataFrame(data = pcs, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
pcdf=pcdf.set_index(df.index)

expVar = pca.explained_variance_ratio_
loadings = pd.DataFrame(pca.components_, index=['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
#loadings = loadings.transpose()
vars = dict(zip(loadings.columns,df.columns))
loadings = loadings.rename(columns=vars)
#loadings.to_csv(r"D:\inspires\data\ClimateNA\1961-1990_loadings_full.csv")
print('total explained variance is %s with %s PCs' %(sum(expVar), len(expVar)))

#replicate pca scores from standardized data and pca loadings
#replicate_pcdf = pd.DataFrame((scaler.transform(df)*loadings.loc['PC1'].to_numpy()).sum(axis=1))

#RUN CLUSTER ALGORITHM ON PCs################################################
from sklearn.cluster import AgglomerativeClustering
#n_clusters = 9
#subsample pcdf because the full number of samples with 4 PCs require ~70GB ram.
#It can handle 10k, and probably 50k. test with fewer PCs too.
#If I really need to do this with all samples, would need to move it to ACG resources.
#Empirically, #PCs doesn't affect time to solution too much (but I feel it must, somehow).
#50k takes about 5 mins and a lot of thinking with 2PCs.
#5,000 with seed 18 or 37 works great. Use seed 19, cluster 9 for B&L Comparison
#pcdf=pcdf.join(df['ID'])
pcsample=pcdf.sample(5000,random_state=37)
#pcsample_idx = pcsample['ID']
#pcsample = pcsample.drop('ID',axis=1)
#pcdf = pcdf.drop('ID',axis=1)

#define cluster model and fit
#ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
n_clust=3#9#15#3 #B&L use 9 #with seed 18, optimals are 3, 7, 15, 23
model = AgglomerativeClustering(distance_threshold=None, n_clusters=n_clust, affinity='euclidean', linkage='ward').fit(pcsample)
#from sklearn.cluster import DBSCAN
#model = DBSCAN(eps=0.1, min_samples=1000).fit(pcsample)

#labels = pd.DataFrame(model.labels_)
#labels=labels.rename(columns={0: 'label'})
#c = model.children_
#d = model.distances_

#CLASSIFY NON-TRAINING DATA######################################################
# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(pcsample, model.labels_)
res = knn.predict(pcdf)
#sil_samples = silhouette_samples(pcdf, res)

#INTRACLUSTER (OR INTRA-CLASS) VARIANCE#######################################
#from sklearn.metrics import silhouette_samples, silhouette_score
silhouette_avg = silhouette_score(pcsample, model.labels_)

#individual samples
sil_samples = silhouette_samples(pcsample, model.labels_)
#sil=pcsample
pcsample['cluster']=model.labels_
pcsample['sil']=sil_samples
sil_thresholds = pcsample.groupby(['cluster'])['sil'].min()

#BRIGGS & LEMIN###############################################################
#check score of Briggs and Lemin zones based on pcsamples:
briggs_sample = id.loc[pcsample.index,'Briggs']
briggs_sample = briggs_sample.loc[briggs_sample>0]
silhouette_score(pcsample.loc[briggs_sample.index], briggs_sample)
#plot briggs
briggs = id.loc[id.Briggs>0]
fig, ax= plt.subplots(figsize=(4,4))
plt.scatter(briggs.lon,briggs.lat,s=1,c=briggs.Briggs)
plt.axis('scaled')
plt.show()

labels=pd.DataFrame(res)
labels=labels.set_index(id.index.astype(int))
fig, ax= plt.subplots(figsize=(4,4))
plt.scatter(briggs.lon,briggs.lat,s=1,c=labels.loc[briggs.index,0])
plt.axis('scaled')
plt.show()


#CLUSTER STATS FOR DELINEATED CLIMATE ZONES###########################################
df['cluster'] = res #df['cluster15'] = res
df_means = df.groupby(['cluster']).mean()
df.groupby(['cluster']).plot.kde()
df_stds = df.groupby(['cluster']).std()
df_quantiles = df.groupby(['cluster']).quantile([0, 0.25, 0.5, 0.75, 1])

df_quantiles.to_csv(r"D:\inspires\data\ClimateNA\out\outRasters\CZ9_quants_s37.csv")
df_means.to_csv(r"D:\inspires\data\ClimateNA\out\outRasters\CZ9_means_s37.csv")
df_stds.to_csv(r"D:\inspires\data\ClimateNA\out\outRasters\CZ9_stds_s37.csv")

pcdfr.loc[pcdfr['res'] == 2]['pc1'].plot.kde()
df.groupby(['cluster']).boxplot()
pcdfr.groupby(['res']).boxplot()
svals['cluster']=res
svals.groupby(['cluster']).boxplot()
#SPATIAL/MAP PLOT TEMPLATE###############################################################
#spatial stuff
#import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(16,8))
fig.tight_layout()
#cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
im = ax.scatter(id.lon,id.lat,s=1,c=res, cmap='viridis')
#fig.colorbar(im, orientation = 'vertical')
fig.colorbar(im,
            boundaries=np.append(np.unique(model.labels_)-0.5,np.max(model.labels_)+0.5),
            ticks=np.append(np.unique(model.labels_),np.max(model.labels_)+1),
            spacing='proportional',
            orientation='vertical')
#plt.scatter(id['lon'], id['lat'], c=res)
plt.axis('scaled')
plt.show()

#PLOT DATA AS A GRID/RASTER#################################################################
#get origin point
y_min=np.min(id.y_km)
x_min=np.min(id.x_km)
#clip km coords to origin
y=id.y_km-y_min
x=id.x_km-x_min
#define grid dimensions
yline=range(0,np.max(y)+1,1)
xline=range(0,np.max(x)+1,1)
#create grid
X,Y=np.meshgrid(xline,yline)
Z=-9999+np.empty(np.shape(Y))
#populate grid with cluster flags
Z[y,x]=res
Z=Z.astype('int16')
#plot the grid
#plt.imshow(Z, interpolation='nearest')
#plt.show()

#convert to raster using rasterio and reference map
import rasterio
refmap=rasterio.open(r"D:\inspires\data\ClimateNA\ClimateNA_Reference\ClimateNA_ID_StudyDomainClip.tif",'r')
new_map = rasterio.open(r"D:\inspires\data\ClimateNA\out\outRasters\CZ3_1961-1990_37.tif",
'w',driver=refmap.driver,height=Z.shape[0],width=Z.shape[1],count=1,dtype=Z.dtype,
crs=refmap.crs,transform=refmap.transform)
new_map.write(Z, 1) #write Z toraster band 1 in new_map
new_map.close() #close dat

#2D LOADINGS VECTOR PLOT##################################################################
import matplotlib.pyplot as plt
z = np.zeros(np.shape(loadings)[1])
for i in range(len(z)):
    #plt.arrow(z[i],z[i],loadings.iloc[0][i],loadings.iloc[1][i])
    plt.arrow(z[i],z[i],loadings.iloc[0][i],loadings.iloc[4][i])

for i in range(len(z)):
    plt.text(loadings.iloc[0][i]* 1.15, loadings.iloc[1][i] * 1.15, loadings.columns[i], color = 'g', ha = 'center', va = 'center')
    #plt.text(loadings.iloc[0][i]* 2, loadings.iloc[1][i] * 2, loadings.columns[i], color = 'g', ha = 'center', va = 'center')

plt.xlim([-0.9, 0.9])
plt.ylim([-.9, 0.9])
plt.show()


#PC SCATTER PLOT TEMPLATE############################################################
import matplotlib.pyplot as plt
#all points, including interp'd
plt.scatter(pcdf['pc1'], pcdf['pc2'], s=1, c=res)
plt.show()
#just sampled points
plt.scatter(pcsample['pc1'], pcsample['pc2'], s=1, c=model.labels_)
plt.show()


###############################################################################
#PROCESS FUTURE PROJECTIONS##############################################################
#DATA IMPORT 1#################################################
df_cmip = pd.read_csv(r"D:\inspires\data\ClimateNA\out\2050s_85.csv")
imp_mean_cmip = SimpleImputer(missing_values=-9999.0, strategy='mean')
imp_mean_cmip.fit(df_cmip)
dfi_cmip = pd.DataFrame(imp_mean_cmip.transform(df_cmip))
df_cmip = dfi_cmip.rename(columns=dict(zip(dfi_cmip.columns,df_cmip.columns)))

df_cmip['Tmin_sp']=df_cmip.filter(['Tmin03', 'Tmin04', 'Tmin05'],axis=1).mean(axis=1)
df_cmip['Tmin_su']=df_cmip.filter(['Tmin06', 'Tmin07', 'Tmin08'],axis=1).mean(axis=1)
df_cmip['Tmin_fa']=df_cmip.filter(['Tmin09', 'Tmin10', 'Tmin11'],axis=1).mean(axis=1)
df_cmip['Tmin_wi']=df_cmip.filter(['Tmin12', 'Tmin01', 'Tmin02'],axis=1).mean(axis=1)

df_cmip['Tmax_sp']=df_cmip.filter(['Tmax03', 'Tmax04', 'Tmax05'],axis=1).mean(axis=1)
df_cmip['Tmax_su']=df_cmip.filter(['Tmax06', 'Tmax07', 'Tmax08'],axis=1).mean(axis=1)
df_cmip['Tmax_fa']=df_cmip.filter(['Tmax09', 'Tmax10', 'Tmax11'],axis=1).mean(axis=1)
df_cmip['Tmax_wi']=df_cmip.filter(['Tmax12', 'Tmax01', 'Tmax02'],axis=1).mean(axis=1)

df_cmip['Tave_sp']=df_cmip.filter(['Tave03', 'Tave04', 'Tave05'],axis=1).mean(axis=1)
df_cmip['Tave_su']=df_cmip.filter(['Tave06', 'Tave07', 'Tave08'],axis=1).mean(axis=1)
df_cmip['Tave_fa']=df_cmip.filter(['Tave09', 'Tave10', 'Tave11'],axis=1).mean(axis=1)
df_cmip['Tave_wi']=df_cmip.filter(['Tave12', 'Tave01', 'Tave02'],axis=1).mean(axis=1)

df_cmip['PPT_sp']=df_cmip.filter(['PPT03', 'PPT04', 'PPT05'],axis=1).sum(axis=1)
df_cmip['PPT_su']=df_cmip.filter(['PPT06', 'PPT07', 'PPT08'],axis=1).sum(axis=1)
df_cmip['PPT_fa']=df_cmip.filter(['PPT09', 'PPT10', 'PPT11'],axis=1).sum(axis=1)
df_cmip['PPT_wi']=df_cmip.filter(['PPT12', 'PPT01', 'PPT02'],axis=1).sum(axis=1)

df_cmip=df_cmip.drop(['Tmin01','Tmin02','Tmin03','Tmin04','Tmin05','Tmin06','Tmin07',
'Tmin08','Tmin09','Tmin10','Tmin11','Tmin12'], axis=1)

df_cmip=df_cmip.drop(['Tmax01','Tmax02','Tmax03','Tmax04','Tmax05','Tmax06','Tmax07',
'Tmax08','Tmax09','Tmax10','Tmax11','Tmax12'], axis=1)

df_cmip=df_cmip.drop(['Tave01','Tave02','Tave03','Tave04','Tave05','Tave06','Tave07',
'Tave08','Tave09','Tave10','Tave11','Tave12','Tave_sm','Tave_wt'], axis=1)

df_cmip=df_cmip.drop(['PPT01','PPT02','PPT03','PPT04','PPT05','PPT06','PPT07',
'PPT08','PPT09','PPT10','PPT11','PPT12','PPT_sm','PPT_wt'], axis=1)
#STANDARDIZE AND FIT THE PCA############################################
df_cmip=df_cmip.set_index('ID')

#MEAN AND STANDARD DEV OF CLIMATE VARIABLES FROM 1961-1990 ERA################
svals_cmip = scaler.transform(df_cmip)

#pcdf_cmip = pd.DataFrame(np.zeros(np.shape(pcdf)))
#pcdf_cmip = pcdf_cmip.rename(columns=dict(zip(pcdf_cmip.columns,pcdf.columns)))
#pcdf_cmip = pcdf_cmip.set_index(df.index.astype(int))

#for i in range(np.shape(loadings)[0]):
#    pcdf_cmip.loc[:,'pc%s' %(i + 1)] = (svals_cmip*loadings.loc['pc%s' %(i + 1),:].to_numpy()).sum(axis=1)

#pca=PCA(n_components=2)
#Get PC loadings for cmip using the transform for the historic data: important to keep using the distributions from the historic data to see how trends 'evolve' with climate change
pcdf_cmip = pd.DataFrame(pca.transform(svals_cmip))
pcdf_cmip = pcdf_cmip.rename(columns=dict(zip(pcdf_cmip.columns,pcdf.columns)))
pcdf_cmip = pcdf_cmip.set_index(df.index.astype(int))

#infer cluster numbers based on the cmip PCs
res_cmip = knn.predict(pcdf_cmip)
pcdf_cmip['cluster'] = res_cmip

#sample the cmip as if you were preparing for agglomerative clustering
cmip_sample=pcdf_cmip.sample(5000,random_state=37)

#LocalOutlierTest
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(novelty=True, contamination=0.001)
cmip_novelty = cmip_sample[['cluster']]
cmip_novelty['novelty'] = 0
for i in range(n_clust):
    lof.fit(pcsample.loc[model.labels_==i,:].iloc[:,0:5])
    cmip_novelty_tmp = lof.predict(cmip_sample.loc[cmip_sample['cluster']==i,:].iloc[:,0:5])
    cmip_novelty.loc[cmip_novelty['cluster']==i,'novelty']=cmip_novelty_tmp
lof.fit(pcsample.iloc[:,0:5])
cmip_novelty = lof.predict(cmip_sample.iloc[:,0:5])

knn_nov = KNeighborsClassifier()
knn_nov.fit(cmip_sample, cmip_novelty['novelty'])
res_novelty = knn_nov.predict(pcdf_cmip)

#knn_nov = KNeighborsClassifier()
#knn_nov.fit(cmip_sample, cmip_novelty)
#res_novelty = knn_nov.predict(pcdf_cmip)

#Loop through silhouette scores of cmip samples
#Why a loop? If you run silhouette sample for all cmip at once, you skew the cluster centers
#If you run one at a time appended to the historic data, the cluster centers are virtually unchanged
#A better approach would just use the intra and inter distances, but I can't seem to crack open the code to get that.
#cmip_sample['sil'] = 0.
#for i in range(len(cmip_sample)):
#    print(i)
#    sil_temp = pd.concat([cmip_sample.iloc[i:i+1],pcsample])
#    tmp = silhouette_samples(sil_temp.iloc[:,:5], sil_temp['cluster'].to_numpy())
#    cmip_sample.iloc[i,6]=tmp[0]

#merge_cmip_sample = cmip_sample.merge(sil_thresholds, how = 'inner', on = ['cluster'])
#merge_cmip_sample['threshold']=merge_cmip_sample['sil_x']-merge_cmip_sample['sil_y']
#merge_cmip_sample['threshold']=np.ceil(merge_cmip_sample['threshold'])

#tknn = KNeighborsClassifier()
#tknn.fit(merge_cmip_sample.iloc[:,:6], merge_cmip_sample.loc[:,'threshold'])
#tres = tknn.predict(pcdf_cmip)
#pcdf_cmip['threshold'] = tres

#PLOT DATA AS A GRID/RASTER#################################################################
#get origin point
y_min=np.min(id.y_km)
x_min=np.min(id.x_km)
#clip km coords to origin
y=id.y_km-y_min
x=id.x_km-x_min
#define grid dimensions
yline=range(0,np.max(y)+1,1)
xline=range(0,np.max(x)+1,1)
#create grid
X,Y=np.meshgrid(xline,yline)
Z=-9999+np.empty(np.shape(Y))
#populate grid with cluster flags
Z[y,x]=res_novelty
Z=Z.astype('int16')
#plot the grid
#plt.imshow(Z, interpolation='nearest')
#plt.show()

#convert to raster using rasterio and reference map
import rasterio
refmap=rasterio.open(r"D:\inspires\data\ClimateNA\ClimateNA_Reference\ClimateNA_ID_StudyDomainClip.tif",'r')
new_map = rasterio.open(r"D:\inspires\data\ClimateNA\out\outRasters\CZ3novelty_2050s_RCP85_37_ver2.tif",
'w',driver=refmap.driver,height=Z.shape[0],width=Z.shape[1],count=1,dtype=Z.dtype,
crs=refmap.crs,transform=refmap.transform)
new_map.write(Z, 1) #write Z toraster band 1 in new_map
new_map.close() #close dat

#SVMs, aren't good at outliers....
#from sklearn import svm
#clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
#clf.fit(pcsample)
#cmip_novelty = clf.predict(cmip_sample)


#pcdf = pd.DataFrame(data = pcs, columns = ['pc1', 'pc2'])
#pcdf = pd.DataFrame(data = pcs, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
#pcdf=pcdf.set_index(df.index)

fig, ax= plt.subplots(figsize=(16,8))
fig.tight_layout()
#cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
im = ax.scatter(id.lon,id.lat,s=1,c=res_novelty, cmap='viridis')
#fig.colorbar(im, orientation = 'vertical')
fig.colorbar(im,
            boundaries=np.append(np.unique(model.labels_)-0.5,np.max(model.labels_)+0.5),
            ticks=np.append(np.unique(model.labels_),np.max(model.labels_)+1),
            spacing='proportional',
            orientation='vertical')
#plt.scatter(id['lon'], id['lat'], c=res)
plt.axis('scaled')
plt.show()


expVar = pca.explained_variance_ratio_
loadings = pd.DataFrame(pca.components_, index=['PC1', 'PC2', 'pc3', 'pc4', 'pc5'])


#CREATE DENDROGRAM PLOT######################################
#plot dendrogram (doesn't work without distances, can't get distances if prescribing n_clusters)
from scipy.cluster.hierarchy import dendrogram

counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)
for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count
linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)

# Plot the corresponding dendrogram
dend=dendrogram(linkage_matrix, p=6, truncate_mode='level')
plt.show()


'''
#################################################################################
#AGGREGATING CLUSTERS USING CHILDREN AND DISTANCE ARRAYS#########################
# working with model.children_ data to aggregate/nest clusters
# children_ array is shape(n_samples-1, 2)
#the 2 arrays represent the two children of a parent node
#the values determine between leaf and nonleaf nodes. all nonleaf nodes have value >= n_samples
#you can find children by using a parent value subtracted by n_samples as a row index
#for example, n_samples = 20, i = 21 is a non-leaf node whose children are located at children_[21-20]
#finding children of a parent is easy

#c_idx = c[i-n_samples]
#finding parents is harder and requires np.where to look for the value to find its row index
d_threshold = 200
label_aggregate = np.zeros(np.shape(labels))
for i in range(len(pcsample)):
    c_idx = np.where(c == i)[0]
    p_idx = c_idx
    while d[p_idx] <= d_threshold:
        p_idx = np.where(c == c_idx + n_samples)[0] # I keep only the row as col value doesn't really matter for this
        #you could then use p_idx as the new cluster label for the sample
        #look at distance if the idea is to threshold clusters based on a certain distance
        if d[p_idx] <= d_threshold:
            c_idx = p_idx
        else:
            label_aggregate[i] = c_idx[0]
'''