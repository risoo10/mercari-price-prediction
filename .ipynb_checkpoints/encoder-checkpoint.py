from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

data = pd.read_csv('data/train_sm.tsv', sep="\t")
data.set_index('train_id', inplace=True)
data_to_encode = data["item_condition_id"][:100]

# example usage
#
# ohe = OurOHE()
# encoded = ohe.fit_transform(data_to_encode, "item_condition_id")
# encoded2 = ohe.transform(data_to_encode, "item_condition_id")
# decoded = ohe.inverse_transform(encoded)

class OurOHE():
    def __init__(self):
        self.onehot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
        self.label_encoder = LabelEncoder()

    def fit_transform(self, input_data, column_name):
        data_to_encode = np.array(input_data)

        # integer encode
        integer_encoded = self.label_encoder.fit_transform(data_to_encode)

        # one hot encode
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.onehot_encoder.fit_transform(integer_encoded)

        columns = [f"{column_name}_{i}" for i in range(onehot_encoded.shape[1])]
        dataframe = pd.DataFrame(data=onehot_encoded, columns = columns)

        return dataframe

    def transform(self, input_data, column_name):
        data_to_encode = np.array(input_data)

        # integer encode
        integer_encoded = self.label_encoder.transform(data_to_encode)

        # one hot encode
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.onehot_encoder.transform(integer_encoded)

        columns = [f"{column_name}_{i}" for i in range(onehot_encoded.shape[1])]
        dataframe = pd.DataFrame(data=onehot_encoded, columns = columns)

        return dataframe

    def inverse_transform(self, input_data):
        onehot_encoded = np.array(input_data.values)
        one_hot_inverted = self.onehot_encoder.inverse_transform(onehot_encoded).astype(int)
        label_inverted = self.label_encoder.inverse_transform(one_hot_inverted)
        return label_inverted