import dvc.api
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def make_train_test(test_size: 0.3):
    with dvc.api.open("./data/all_rooms_combined.csv") as file:
        df = pd.read_csv(file, header=True, index_col=0)
        X = df.drop("price", axis=1)
        y = df["price"]
        features_train, features_test, target_train, target_test = train_test_split(
            X, y, test_size=test_size, random_state=1, shuffle=True
        )
        return (features_train, target_train), (features_test, target_test)


def train(cfg: DictConfig):
    train_dataset, valid_dataset = make_train_test(cfg.train_and_validate.get("test_size", 0.2))


def main():
    initialize(version_base=None, config_path="configs", job_name="cian_prediction")
    cfg = compose(config_name="config")
    train(cfg)


if __name__ == "__main__":
    main()
