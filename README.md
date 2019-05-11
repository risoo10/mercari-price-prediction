# mercari-price-prediction
Supervised regression of price for mercari kaggle challange.

## Problem
Mercari is Japan’s biggest community-powered shopping application. Our moti-vation lies in solving a problem of offering pricing suggestions to sellers on theMercari’s marketplace. This is tough because their sellers are able to put for salejust about anything and in any bundle of things.

## Knowledge discovery task
The task of knowledge discovery lies in predicting the price for an item put into the market place based on its text and categorical attributes. We are goingto solve this task by using supervised regression.

## Data
Data are not included in the repository and can be obtained from the Kaggle's website for Mercari Price Challange at [https://www.kaggle.com/c/mercari-price-suggestion-challenge].

## Source code organization
Source code is mainly written in Jupyter Notebooks.

### Preprocessing
Code related to the preprocessing of the data is located in the `notebooks/Predspracovanie.ipynb`.

### Training 
After preprocessing we needed to transform data to `pytables` hdf format to read data during training from the disc instead of RAM.
Code related to this transformation is located at `notebooks/Transform-data.ipynb`.

Training of the Neural Network models and their evaluation is located in the file `notebooks/Training_MLP.ipynb`.

### Vizualization
Code related to tSNE vizualization of the documents vector embeddings is located in the file `notebooks/Visualization.ipynb`.

