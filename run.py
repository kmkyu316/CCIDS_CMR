# -*-coding=utf-8-*-

# radiomics
from featureExtractor import feature_extract

# file_path
from glob import glob
import os

# read image
import tifffile
import pydicom

# save as table
import pandas as pd

# mT
import numpy as np
import matplotlib.pyplot as plt

# PCA
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# K-means
from sklearn.cluster import KMeans

# MDS
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

import random


def radiomics2csv():
	res = ""
	First = True
	for file in ["Kangnam","Sinchon"]:
		# file list
		filepath = "../Stomach/"+file
		# print(glob(filepath+"/*"))
		for num,pf in enumerate(glob(filepath+"/*")):
			pf_name = pf.split("/")[-1]
			print(file,num+1,"/",len(glob(filepath+"/*")),":",pf_name)
			if pf_name in ["5821012"]:
				continue
			mask_path = pf + "/" + pf_name +"m.tif"
			image_path = pf + "/" + pf_name +"/"+ pf_name +".dcm"

			mask = tifffile.imread(mask_path)
			image = pydicom.read_file(image_path).pixel_array
			
			# 1band to 3 bands
			# mask = np.stack((mask,)*3,axis=-1)
			# image = np.stack((image,)*3,axis=-1)
			
			# print(mask.shape)
			# print(image.shape)

			feature_values, feature_columns = feature_extract(image, mask, voxel = (1,1,1))

			# Add Results to Pandas DataFrame
			feature_dict = {}
			feature_dict["hospital"] = file
			feature_dict["Pid"] = pf_name
			pindex = file[0]+str(pf_name)
			for val,feat in zip(feature_columns, feature_values):
				feature_dict[val] = feat
			# add hospital type

			if First:
				First = False
				total_df = pd.DataFrame(feature_dict, index=[pindex])

			else:
				temp_df = pd.DataFrame(feature_dict, index=[pindex])
				total_df = total_df.append(temp_df, sort = True)


	total_df.to_csv('GI_cancer_radiomics.csv')

	return total_df

def PCA():

	df = pd.read_csv('Radiomics.csv')

	x = df.drop(['PID'], axis=1)
	x = pd.DataFrame(scale_data, index=[df['PID']])

	scale_data = preprocessing.scale(x)

	scale_data1 = StandardScaler().fit_transform(x)

	pca = PCA()
	pca.fit(scale_data)
	pca_data = pca.transform(scale_data)

	per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)

	labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]

	pca_df = pd.DataFrame(pca_data, index=[df['PID']],columns=labels)
	pca_df.to_csv('GI_cancer_PCA.csv')

	return pca_df
	# plt.scatter(pca_df.PC1,pca_df.PC2)
	# plt.xlabel('PC1 - {0}%'.format(per_var[0]))
	# plt.ylabel('PC2 - {0}%'.format(per_var[1]))

def kmeans(pf_data,z):

	x = pf_data.drop(['PID'], axis=1)

	data_points = x.values
	data_points.shape

	km= KMeans(n_clusters=z,max_iter=10000000,random_state=10)
	km.fit(data_points)
	x['cluster_id'] = km.labels_
	x.to_csv('GI_cancer_'+str(z)+'_kmeans.csv')
	dist = 1 - cosine_similarity(data_points)
	mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
	pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

	xs, ys = pos[:, 0], pos[:, 1]
	# random color
	r = lambda: random.randint(0,255)
	cluster_colors = dict()
	for i in range(0,100):
		cluster_colors[i] = '#%02X%02X%02X' % (r(),r(),r())

	clusters = km.labels_.tolist()

	df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 
	df.to_csv('GI_cancer_'+str(z)+'_mds_kmeans.csv')


def main():
	# get radiomics features
	pf_data = radiomics2csv()
	# run PCA
	pf_data = PCA(pf_data)
	# run kmeans
	k = 5
	kmeans(pf_data,k)

if __name__ == '__main__':
	main()