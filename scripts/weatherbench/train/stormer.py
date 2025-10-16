import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import zephyr
from zephyr.config import DataConfig
from zephyr.data import VARIABLES_CATALOG
from zephyr.models.stormer import Config as StormerConfig
from zephyr.training.trainer import TrainerArgs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

cs = ConfigStore.instance()
cs.store(name="model_schema", node=StormerConfig)
cs.store(name="data_schema", node=DataConfig)
cs.store(name="training_schema", node=TrainerArgs)


@hydra.main(version_base=None, config_path=Path(zephyr.__file__) / "configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_cfg = OmegaConf.to_object(cfg.model)
    data_cfg = OmegaConf.to_object(cfg.data)
    training_cfg = OmegaConf.to_object(cfg.training)

    if not data_cfg.zarr_path.exists():
        logger.error(f"Zarr data not found: {data_cfg.zarr_path}")
        return

    train_dataset = zephyr.WeatherBenchDataset(
        zarr_path=data_cfg.zarr_path, years=data_cfg.train_years,
        sequence_length=data_cfg.sequence_length, forecast_horizon=data_cfg.forecast_horizon,
        normalize=data_cfg.normalize,
    )
    val_dataset = zephyr.WeatherBenchDataset(
        zarr_path=data_cfg.zarr_path, years=data_cfg.val_years,
        sequence_length=data_cfg.sequence_length, forecast_horizon=data_cfg.forecast_horizon,
        normalize=data_cfg.normalize,
    )

    model = zephyr.Stormer(
        img_size=tuple(model_cfg.img_size), variables=VARIABLES_CATALOG,
        patch_size=model_cfg.patch_size, hidden_size=model_cfg.hidden_size,
        depth=model_cfg.depth, num_heads=model_cfg.num_heads, mlp_ratio=model_cfg.mlp_ratio,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable)")

    trainer = zephyr.Trainer(
        model=model, args=training_cfg,
        training_dataset=train_dataset, val_dataset=val_dataset,
        collate_fn=zephyr.collate_fn,
    )

    trainer.train()


if __name__ == "__main__":
    main()
