import numpy as np
import pandas as pd
import glob
import os

pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', 100)

#print os.getcwd()

#os.chdir('\Research\Topic Modeling\WOS')
file_names = glob.glob("*.txt")

pieces = []

for path in file_names:
	eo = pd.read_csv(path, sep='\t')
	pieces.append(eo)

articles = pd.concat(pieces, ignore_index=True)

print articles.shape
print articles['SO'].value_counts()
print articles['SO'].nunique()

print len(articles)
articles = articles.drop_duplicates()
print len(articles)

#articles['AB'] = articles['AB'].fillna('')

#index using unique WOS id
articles = articles.set_index('UT')

#keep author, title, journal name, publication date, year published, abstract, and times cited
columns_to_keep = ['AU', 'TI', 'SO', 'PD', 'PY', 'AB', 'TC']

articles = articles[columns_to_keep]

#drop observations with missing data (i.e., missing abstracts)
articles = articles.dropna()

print articles['AB'].describe()

#articles.to_csv('\\articles_1990_present_w_abstracts.csv')

#print articles['PY'].value_counts()

#print articles['PY'].hist()