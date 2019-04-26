import pandas as pd
import numpy as np
import datetime

data = pd.read_csv('data/train.tsv', sep="\t")
data.set_index('train_id', inplace=True)
# TRAIN_SIZE = 100000
# data = data.iloc[:TRAIN_SIZE]

# data.to_csv("train.csv.gz", compression="gzip")

# Separate categories to custom columns
category_path_max = 3
data['categories'] = data['category_name'].apply(lambda x: x.split('/') if isinstance(x, str) else [np.nan,np.nan,np.nan])
for i in range(category_path_max):
  data[str(i) + '_category'] = data['categories'].apply(lambda x: x[i] if len(x) > i else np.nan)

data = data.drop(['categories', 'category_name'], 1)


# Brand name
vc = data["brand_name"].value_counts() < 5
mask = pd.DataFrame(vc[vc])
data.loc[data["brand_name"].isin(mask.index), "brand_name"] = "Other"
data["name_and_brand"] = data["name"].map(str) + ' ' + data["brand_name"].map(str)
data = data.drop("name", axis=1)

before = datetime.datetime.now()
print(f"ohe start - {before}")

# item_condition_id
encoded_item_condition_id = pd.get_dummies(data["item_condition_id"])
data = pd.concat([data, encoded_item_condition_id], axis=1)
data = data.drop("item_condition_id", axis=1)

# brand_name
encoded_brand_name = pd.get_dummies(data["brand_name"])
data = pd.concat([data, encoded_brand_name], axis=1)
data = data.drop("brand_name", axis=1)

# 0_category
encoded_category_0 = pd.get_dummies(data["0_category"])
data = pd.concat([data, encoded_category_0], axis=1)
data = data.drop("0_category", axis=1)

# 1_category
encoded_category_1 = pd.get_dummies(data["1_category"])
data = pd.concat([data, encoded_category_1], axis=1)
data = data.drop("1_category", axis=1)

# 2_category
encoded_category_2 = pd.get_dummies(data["2_category"])
data = pd.concat([data, encoded_category_2], axis=1)
data = data.drop("2_category", axis=1)

after = datetime.datetime.now()
print(f"ohe finish - {after}")
print(f"took - {after - before}")

data.to_csv('data/train_encoded.tsv', sep="\t")

print(data.head())
