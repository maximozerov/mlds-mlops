import dvc.api
import joblib
import numpy as np
import onnx
import onnxruntime
import pandas as pd
from onnxruntime import InferenceSession, SessionOptions
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


def log_transform(x):
    return np.log(x + 1)


def transform_dfs(train_df, test_df, path_for_transformer):
    log = ["total_meters", "kitchen_meters"]
    categorical = ["admin_okrug", "subway", "is_skyscraper", "class_real", "way_to_subway"]
    ordinal = ["rooms"]

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

    save_transformer(col_transformer, path_for_transformer)
    return X_train_transformed, X_test_transformed


def save_model(model, path, features_shape):
    initial_type = [("float_input", FloatTensorType([None, features_shape]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def load_model(path):
    with dvc.api.open(path, encoding="utf8") as file:
        onnx_model = onnx.load(file)
        onnx.checker.check_model(onnx_model)
    return onnx_model


def save_transformer(transformer, path):
    joblib.dump(transformer, path)


def load_transformer(path):
    with dvc.api.open(path, encoding="utf8") as file:
        transformer = joblib.load(file)
    return transformer


def make_inference(model, transformer, df, save_path):
    with dvc.api.open(df, encoding="utf8") as file:
        df = pd.read_csv(file, index_col=0, encoding="utf8")

    log = ["total_meters", "kitchen_meters"]
    categorical = ["admin_okrug", "subway", "is_skyscraper", "class_real", "way_to_subway"]
    ordinal = ["rooms"]

    features_transformed = transformer.transform(df[log + categorical + ordinal]).astype("float32")

    options = SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1

    session = InferenceSession(model, options, providers=["CPUExecutionProvider"])
    session.disable_fallback()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    predictions = session.run([output_name], {input_name: features_transformed})

    result_df = pd.DataFrame(predictions[0])
    result_df.to_csv(save_path, header=False)

    return predictions
