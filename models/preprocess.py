import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_uploaded_data(df):
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop("income", axis=1)
    y = df["income"]

    y = y.apply(lambda x: 1 if ">50K" in x else 0)

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
