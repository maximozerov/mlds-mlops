from hydra import compose, initialize
from mlds_mlops.utils import make_train_test, save_model, transform_dfs
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train(cfg: DictConfig):
    train_df, test_df = make_train_test(cfg.train_and_validate.get("test_size", 0.2))

    train_features, test_features = transform_dfs(train_df, test_df)

    rf = RandomForestRegressor(
        criterion=cfg.rf.get("criterion"), n_estimators=cfg.rf.get("n_estimators", 200)
    )

    rf.fit(train_features, train_df[1])

    train_preds = rf.predict(train_features)
    test_preds = rf.predict(test_features)

    MAE_train = round(mean_absolute_error(train_df[1], train_preds), 3)
    MAE_test = round(mean_absolute_error(test_df[1], test_preds), 3)

    print(MAE_train, MAE_test)

    save_model(rf, cfg.model.get("save_name", "rf_regressor"), train_features.shape[1])


def main():
    initialize(version_base=None, config_path="configs", job_name="cian_prediction")
    cfg = compose(config_name="config")
    train(cfg)


if __name__ == "__main__":
    main()
