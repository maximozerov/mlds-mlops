from hydra import compose, initialize
from mlds_mlops.utils import load_transformer, make_inference
from omegaconf import DictConfig


def infer(cfg: DictConfig):
    model = cfg.model.get("save_name")
    transformer = load_transformer(cfg.transformer.get("save_path"))

    make_inference(model, transformer, "./data/all_rooms_combined.csv", cfg.infer.get("save_path"))


def main():
    initialize(version_base=None, config_path="configs", job_name="cian_prediction")
    cfg = compose(config_name="config")
    infer(cfg)


if __name__ == "__main__":
    main()
