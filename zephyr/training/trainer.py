import itertools
import logging
import multiprocessing as mp
import random
import string
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import zephyr

from ..data.structures import WeatherBatch
from ..data.variables import FORCED_VARIABLES, VARIABLES_CATALOG
from .losses import LatitudeWeightedMSELoss
from .metrics import WeatherMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerArgs:
    model_name: str
    device: str = "cuda"
    training_batch_size: int = 1
    validation_batch_size: int = 1
    number_of_steps: int = 100
    training_size_per_step: int = 500
    validation_size_per_step: int = 1_000
    wandb_track: bool = False

    # optimizer
    lr: float = 5e-5
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.85
    weight_decay: float = 1e-9

    # checkpointing
    intermediate_checkpointing: bool = True
    load_from_checkpoint: bool = False
    checkpoint_frequency: int = 10
    checkpoint_path: Optional[Path] = None

    # monitoring
    track_gradnorm: bool = True

    # training enhancements
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    validation_frequency: int = 1  # validate every N steps

    # early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # performance
    pin_memory: bool = True
    persistent_workers: bool = True
    num_workers: Optional[int] = None  # None means mp.cpu_count() - 1

    # distributed training
    log_with: Optional[str] = None

    def __post_init__(self) -> None:
        if self.device not in ["cpu", "cuda"] and not self.device.startswith("cuda:"):
            raise ValueError(f"Invalid device: {self.device}")
        if self.training_batch_size <= 0 or self.validation_batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        if self.number_of_steps <= 0 or self.training_size_per_step <= 0:
            raise ValueError("Training steps/size must be positive")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.gradient_clip_val < 0:
            raise ValueError("Gradient clip must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        if self.mixed_precision and not torch.cuda.is_available():
            logger.warning("Mixed precision unavailable without CUDA")
        if self.early_stopping and (
            self.early_stopping_patience <= 0 or self.early_stopping_min_delta < 0
        ):
            raise ValueError("Invalid early stopping parameters")
        if self.checkpoint_frequency <= 0:
            raise ValueError("Checkpoint frequency must be positive")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainerArgs,
        training_dataset,
        val_dataset=None,
        optimizer=None,
        scheduler=None,
        loss=None,
        collate_fn=None,
        output_transform=None,
    ) -> None:
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.training_dataset = training_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.output_transform = nn.Identity() if output_transform is None else output_transform

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="fp16" if args.mixed_precision else "no",
            log_with=args.log_with if args.log_with else None,
            project_dir=str(Path(zephyr.__file__).parents[1] / "accelerate_logs"),
        )
        self.device = self.accelerator.device

        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.should_stop = False
        self.accumulated_loss = 0.0
        self.step_metrics = {"train_loss": [], "val_loss": [], "grad_norm": []}

        if self.args.wandb_track and self.accelerator.is_main_process:
            wandb.init(project="zephyr", config=self.args.__dict__)

        if self.accelerator.is_main_process:
            run_name = wandb.run.name if self.args.wandb_track else "local"
            self.folder_name = (
                Path(zephyr.__file__).parents[1]
                / "checkpoints"
                / (run_name + "_".join(random.choices(string.ascii_uppercase + string.digits, k=5)))
            )
            self.folder_name.mkdir(parents=True, exist_ok=True)
        else:
            self.folder_name = None

        num_workers = (
            self.args.num_workers
            if self.args.num_workers is not None
            else max(1, mp.cpu_count() - 1)
        )

        self.train_dataloader = DataLoader(
            dataset=self.training_dataset,
            batch_size=self.args.training_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=self.args.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.args.persistent_workers and num_workers > 0,
        )

        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.args.validation_batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=num_workers,
                pin_memory=self.args.pin_memory and torch.cuda.is_available(),
                persistent_workers=self.args.persistent_workers and num_workers > 0,
            )
        else:
            self.val_dataloader = None

        self._setup()

    def get_current_epoch(self, step_num: int) -> int:
        total_batches = step_num * self.args.training_size_per_step
        return total_batches // self.batches_per_epoch

    def _setup(self) -> None:
        self.model = self._get_model()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.loss = self._get_loss()

        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        if self.scheduler is not None:
            self.scheduler = self.accelerator.prepare(self.scheduler)
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        self.train_dataiter = itertools.cycle(self.train_dataloader)
        self.val_dataiter = itertools.cycle(self.val_dataloader) if self.val_dataloader else None
        self.batches_per_epoch = len(self.train_dataloader)

        self.train_metrics = WeatherMetrics(
            variable_names=VARIABLES_CATALOG,
            forced_variables=FORCED_VARIABLES,
            device=str(self.device),
        )
        self.val_metrics = WeatherMetrics(
            variable_names=VARIABLES_CATALOG,
            forced_variables=FORCED_VARIABLES,
            device=str(self.device),
        )

        if self.args.wandb_track and self.accelerator.is_main_process:
            wandb.watch(self.model, log="all", log_freq=100)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.config.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": total_params * 4 / (1024**2),
                }
            )

    def _get_model(self) -> nn.Module:
        model = deepcopy(self.model)
        if self.args.load_from_checkpoint:
            assert self.args.checkpoint_path is not None
            checkpoint = torch.load(self.args.checkpoint_path, map_location=self.accelerator.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint: {self.args.checkpoint_path}")
        return model

    def _get_loss(self) -> nn.Module:
        if self.loss is not None:
            return self.loss
        try:
            return LatitudeWeightedMSELoss(VARIABLES_CATALOG, FORCED_VARIABLES)
        except Exception as e:
            logger.warning(f"Failed to create LatitudeWeightedMSELoss: {e}, using MSELoss")
            return nn.MSELoss()

    def _get_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        if self.scheduler is not None:
            return self.scheduler
        assert self.optimizer is not None
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="min",
            factor=0.1,
            patience=15,
            threshold=0.01,
            threshold_mode="abs",
            cooldown=3,
        )

    def _get_optimizer(self) -> optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        return optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

    def _scheduler_step(self, validation_loss: float) -> None:
        if self.scheduler is not None:
            self.scheduler.step(validation_loss)

    def _apply_gradient_clipping(self) -> float:
        if self.args.gradient_clip_val > 0:
            return self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.args.gradient_clip_val
            ).item()
        grads = [p.grad.detach().flatten() for p in self.model.parameters() if p.grad is not None]
        return torch.cat(grads).norm().item() if grads else 0.0

    def _check_early_stopping(self, val_loss: float) -> bool:
        if not self.args.early_stopping:
            return False
        if val_loss < self.best_val_loss - self.args.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        if self.early_stopping_counter >= self.args.early_stopping_patience:
            logger.info(f"Early stopping after {self.early_stopping_counter} steps")
            return True
        return False

    def _singlestep_prediction(self, weather_batch: WeatherBatch) -> WeatherBatch:
        self.model.validate_weather_batch(weather_batch)
        return self.model(weather_batch)

    def _prediction(self, weather_batch: WeatherBatch) -> WeatherBatch:
        return self._singlestep_prediction(weather_batch)

    def _train_step(self, weather_batch: WeatherBatch) -> torch.Tensor:
        prediction_batch = self._prediction(weather_batch)
        return self.calculate_loss(
            weather_batch.flatten_targets(), prediction_batch.flatten_targets()
        )

    def calculate_loss(self, target: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        return self.loss(target, predictions)

    def _validate_batch(self, weather_batch: WeatherBatch) -> bool:
        if weather_batch.surface_inputs is not None and weather_batch.surface_inputs.isnan().any():
            return False
        if (
            weather_batch.surface_targets is not None
            and weather_batch.surface_targets.isnan().any()
        ):
            return False
        if (
            weather_batch.atmospheric_inputs is not None
            and weather_batch.atmospheric_inputs.isnan().any()
        ):
            return False
        if (
            weather_batch.atmospheric_targets is not None
            and weather_batch.atmospheric_targets.isnan().any()
        ):
            return False
        try:
            self.model.validate_weather_batch(weather_batch)
        except ValueError:
            return False
        return True

    def _process_single_batch(self, weather_batch: WeatherBatch) -> float:
        with self.accelerator.accumulate(self.model):
            loss = self._train_step(weather_batch)
            self.accelerator.backward(loss)
        return loss.item()

    def _optimizer_step(self) -> float:
        grad_norm = self._apply_gradient_clipping()
        self.accelerator.step(self.optimizer)
        self.accelerator.zero_grad(self.optimizer)
        return grad_norm

    def train_step(self, number_of_batches: int) -> Tuple[float, float]:
        self.model.train()
        loss_history, norms_history = [], []
        self.accumulated_loss, valid_batches = 0.0, 0

        for batch_idx in tqdm(range(number_of_batches), desc="Training"):
            weather_batch = next(self.train_dataiter)
            if not self._validate_batch(weather_batch):
                continue

            batch_loss = self._process_single_batch(weather_batch)
            self.accumulated_loss += batch_loss
            valid_batches += 1

            if self.accelerator.sync_gradients:
                grad_norm = self._apply_gradient_clipping()
                avg_loss = self.accumulated_loss / valid_batches
                loss_history.append(avg_loss)
                norms_history.append(grad_norm)
                self.accumulated_loss, valid_batches = 0.0, 0

        if not loss_history:
            return float("nan"), float("nan")
        return np.mean(loss_history).item(), np.mean(norms_history).item()

    @torch.no_grad()
    def evaluate_step(self, number_of_batches: int = 300):
        self.model.eval()
        if self.val_dataiter is None:
            return {"val_loss": float("nan")}

        self.val_metrics.reset()
        validation_loss_history = []

        for _ in tqdm(range(number_of_batches), desc="Validation"):
            weather_batch = next(self.val_dataiter)
            if not self._validate_batch(weather_batch):
                continue

            with self.accelerator.autocast():
                prediction_batch = self._prediction(weather_batch)
                targets_flat = weather_batch.flatten_targets()
                preds_flat = prediction_batch.flatten_targets()
                loss = self.calculate_loss(targets_flat, preds_flat)

            validation_loss_history.append(loss.item())
            self.val_metrics.update(preds_flat, targets_flat)

        if not validation_loss_history:
            return {"val_loss": float("nan")}

        metrics_dict = self.val_metrics.compute()
        metrics_dict["val_loss"] = np.mean(validation_loss_history).item()
        return metrics_dict

    def train(self) -> None:
        logger.info(f"Starting training for {self.args.number_of_steps} steps")

        for step_num in range(self.args.number_of_steps):
            train_loss, gradnorm = self.train_step(self.args.training_size_per_step)

            if np.isnan(train_loss):
                logger.error(f"NaN loss at step {step_num}")
                break

            logger.info(f"Step {step_num}: train_loss={train_loss:.4f}, grad_norm={gradnorm:.4f}")

            val_metrics, val_loss = {}, np.nan
            if self.val_dataiter is not None and step_num % self.args.validation_frequency == 0:
                val_metrics = self.evaluate_step(self.args.validation_size_per_step)
                val_loss = val_metrics.get("val_loss", np.nan)

                if not np.isnan(val_loss):
                    logger.info(
                        f"Step {step_num}: val_loss={val_loss:.4f}, "
                        f"val_rmse={val_metrics.get('overall/rmse', np.nan):.4f}, "
                        f"val_mae={val_metrics.get('overall/mae', np.nan):.4f}"
                    )
                    self._scheduler_step(val_loss)

                    if self._check_early_stopping(val_loss):
                        break

            number_of_obs = (
                self.args.training_batch_size * self.args.training_size_per_step * (step_num + 1)
            )

            if (
                self.args.intermediate_checkpointing
                and step_num % self.args.checkpoint_frequency == 0
                and self.accelerator.is_main_process
                and self.folder_name is not None
            ):
                checkpoint_data = {
                    "step_num": step_num,
                    "total_number_observations": number_of_obs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss,
                    "early_stopping_counter": self.early_stopping_counter,
                    "args": self.args,
                }
                checkpoint_path = self.folder_name / f"step_num_{step_num}.pth"
                self.accelerator.save_state(str(self.folder_name / f"accelerator_state_{step_num}"))
                torch.save(checkpoint_data, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            if self.args.wandb_track and self.accelerator.is_main_process:
                log_dict = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "num_samples": number_of_obs,
                    "epoch": self.get_current_epoch(step_num),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "step": step_num,
                }

                if self.args.track_gradnorm and not np.isnan(gradnorm):
                    log_dict["grad_norm"] = gradnorm
                if self.args.early_stopping:
                    log_dict.update(
                        {
                            "best_val_loss": self.best_val_loss,
                            "early_stopping_counter": self.early_stopping_counter,
                        }
                    )
                if val_metrics:
                    for metric_name, metric_value in val_metrics.items():
                        if metric_name != "val_loss":
                            log_dict[f"val/{metric_name}"] = metric_value
                wandb.log(log_dict)

        logger.info("Training completed")
