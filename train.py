import warnings

import dvc.api
import numpy as np
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


warnings.simplefilter(action="ignore", category=FutureWarning)


def make_train_test(test_size: 0.3):
    with dvc.api.open("./data/all_rooms_combined.csv", encoding="utf8") as file:
        df = pd.read_csv(file, index_col=0, encoding="utf8")
        features = df.drop("price", axis=1)
        target = df["price"]
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=test_size, random_state=1, shuffle=True
        )
        return (features_train, target_train), (features_test, target_test)


def train(cfg: DictConfig):
    train_df, test_df = make_train_test(cfg.train_and_validate.get("test_size", 0.2))

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

    rf = RandomForestRegressor(
        criterion=cfg.rf.get("criterion"), n_estimators=cfg.rf.get("n_estimators", 200)
    )
    rf.fit(X_train_transformed, train_df[1])
    train_preds = rf.predict(X_train_transformed)
    test_preds = rf.predict(X_test_transformed)

    MAE_train = round(mean_absolute_error(train_df[1], train_preds), 3)
    MAE_test = round(mean_absolute_error(test_df[1], test_preds), 3)

    print(MAE_train, MAE_test)

    initial_type = [("float_input", FloatTensorType([None, X_train_transformed.shape[1]]))]
    onnx_model = convert_sklearn(rf, initial_types=initial_type)
    with open(cfg.model.get("save_name", "rf_regressor"), "wb") as f:
        f.write(onnx_model.SerializeToString())


def main():
    initialize(version_base=None, config_path="configs", job_name="cian_prediction")
    cfg = compose(config_name="config")
    train(cfg)


if __name__ == "__main__":
    main()
