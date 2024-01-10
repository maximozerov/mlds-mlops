import dvc.api
import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def make_train_test(test_size: 0.3):
    with dvc.api.open("./data/all_rooms_combined.csv", encoding="utf8") as file:
        df = pd.read_csv(file, index_col=0, encoding="utf8")
        features = df.drop("price", axis=1)
        target = df["price"]
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=test_size, random_state=1, shuffle=True
        )
        return (features_train, target_train), (features_test, target_test)


def transform_dfs(train_df, test_df):
    log = ["total_meters", "kitchen_meters"]
    categorical = ["admin_okrug", "subway", "is_skyscraper", "class_real", "way_to_subway"]
    ordinal = ["rooms"]

    def log_transform(x):
        return np.log(x + 1)

    log_transformer = FunctionTransformer(log_transform)
    col_transformer = ColumnTransformer(
        [
            ("Log transform", log_transformer, log),
            ("Scale", StandardScaler(), ordinal),
            ("One hot", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical),
        ],
        remainder="passthrough",
    )

    X_train_transformed = col_transformer.fit_transform(train_df[0][log + categorical + ordinal])
    X_test_transformed = col_transformer.transform(test_df[0][log + categorical + ordinal])

    return X_train_transformed, X_test_transformed


def save_model(model, path, features_shape):
    initial_type = [("float_input", FloatTensorType([None, features_shape]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
