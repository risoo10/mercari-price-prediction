import pandas as pd
import numpy as np

data = pd.read_csv('data/train.tsv', sep="\t")
data.set_index('train_id', inplace=True)
# TRAIN_SIZE = 100000
# data = data.iloc[:TRAIN_SIZE]

data.to_csv("train.csv.gz", compression="gzip")

# Separate categories to custom columns
category_path_max = 3
data['categories'] = data['category_name'].apply(lambda x: x.split('/') if isinstance(x, str) else [np.nan,np.nan,np.nan])
for i in range(category_path_max):
  data[str(i) + '_category'] = data['categories'].apply(lambda x: x[i] if len(x) > i else np.nan)

data = data.drop(['categories', 'category_name'], 1)


# Brand name
data["name_and_brand"] = data["name"].map(str) + ' ' + data["brand_name"].map(str)
vc = data["brand_name"].value_counts() < 5
mask = pd.DataFrame(vc[vc])
data.loc[data["brand_name"].isin(mask.index), "brand_name"] = "Other"
data = data.drop("name", axis=1)

print(data.head())
