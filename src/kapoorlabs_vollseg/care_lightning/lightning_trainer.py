"""Lightning training scaffolding used by CARE (and ROI).

Trimmed copy of ``kapoorlabs_lightning.lightning_trainer`` keeping only
the classes the CARE / ROI training paths actually instantiate:

* ``LightningModelTrain`` — high-level façade that wraps a Lightning
  ``Trainer.fit`` call with our SLURM-friendly signal handling.
* ``LightningTrainer`` — thin ``lightning.Trainer`` subclass that forces
  ``num_sanity_val_steps=0`` and forwards every other arg.
* ``_HandlersCompose`` / ``_KlabSignalConnector`` — SIGTERM/SIGUSR1
  handlers that copy NPZ logs to a backup dir and requeue the SLURM job.

The ``MitosisInception`` class and the duplicate ``LightningTrainer``
that lived in the upstream file were dropped — they pulled in
``DenseNet`` / ``OneatActionModule`` / ``CellFateModule`` etc., none of
which CARE or ROI needs.
"""

import fcntl
import logging
import os
import shutil
import signal
import threading
from datetime import timedelta
from pathlib import Path
from subprocess import call
from types import FrameType
from typing import Any, Callable, Optional, Union
from collections.abc import Iterable

from lightning import Callback, LightningModule, Trainer
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
)
from lightning.pytorch.utilities.rank_zero import (
    rank_prefixed_message,
    rank_zero_info,
)

from .base_module import BaseModule
from .pytorch_datasets import GenericDataModule


class LightningModelTrain:

    """
    Class for training PyTorch Lightning models.

    Args:
        datamodule (LightningDataModule): LightningDataModule instance for data handling.
        model (LightningModule): The PyTorch Lightning model to be trained.
        callbacks (List[Callback]): List of PyTorch Lightning callbacks.
        logger (Logger): Logger for recording training logs.
        ckpt_path (str): Path to the checkpoint file.
        min_epochs (int): Minimum number of epochs to train the model.
        epochs (int): Total number of epochs to train the model.
        accelerator (str): Accelerator type for distributed training (e.g., 'cpu', 'gpu', 'tpu').
        devices (int): Number of devices to use for training.
        num_nodes (int): Number of nodes to use for distributed training.
        strategy (str): Distributed training strategy.
        enable_checkpointing (bool): Whether to enable checkpointing during training.
        rank_zero_only (bool): Whether to log only for the master process.
        log_every_n_steps (int): Frequency of logging steps.
        default_root_dir (str): Default root directory for logs and checkpoints.
        slurm_auto_requeue (bool): Whether to automatically requeue jobs on SLURM.
        use_slurm (bool): Whether to use SLURM for distributed training.
        precision (Union[int, str]): Precision for training (e.g., '16', '32', '16-true', '16-false').
        deterministic (Union[bool, str]): Whether to use deterministic training (True/False).
        gradient_clip_val (Optional[Union[int, float]]): Value to clip gradients during training.
        gradient_clip_algorithm (Optional[str]): Algorithm to use for gradient clipping.

    Methods:
        train_model(): Train the PyTorch Lightning model.
        callback_metrics(): Get callback metrics from the trainer.

    Attributes:
        datamodule (LightningDataModule): LightningDataModule instance for data handling.
        model (LightningModule): The PyTorch Lightning model to be trained.
        callbacks (List[Callback]): List of PyTorch Lightning callbacks.
        logger (Logger): Logger for recording training logs.
        ckpt_path (str): Path to the checkpoint file.
        min_epochs (int): Minimum number of epochs to train the model.
        epochs (int): Total number of epochs to train the model.
        accelerator (str): Accelerator type for distributed training.
        devices (int): Number of devices to use for training.
        num_nodes (int): Number of nodes to use for distributed training.
        strategy (str): Distributed training strategy.
        enable_checkpointing (bool): Whether to enable checkpointing during training.
        rank_zero_only (bool): Whether to log only for the master process.
        log_every_n_steps (int): Frequency of logging steps.
        default_root_dir (str): Default root directory for logs and checkpoints.
        slurm_auto_requeue (bool): Whether to automatically requeue jobs on SLURM.
        use_slurm (bool): Whether to use SLURM for distributed training.
        precision (str): Precision for training.
        deterministic (Union[bool, str]): Whether to use deterministic training.
        gradient_clip_val (Optional[Union[int, float]]): Value to clip gradients during training.
        gradient_clip_algorithm (Optional[str]): Algorithm to use for gradient clipping.
        accumulate_grad_batches: Accumulate gradient batches

    Raises:
        AssertionError: If ckpt_path is specified but the file does not exist.
    """

    def __init__(
        self,
        datamodule: GenericDataModule = None,
        model: LightningModule = None,
        train_dataloaders=None,
        val_dataloaders=None,
        callbacks: list[Callback] = None,
        logger: Logger = None,
        ckpt_path: str = None,
        min_epochs: int = 1,
        epochs: int = 10,
        accelerator: str = "cuda",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "auto",
        enable_checkpointing: bool = True,
        rank_zero_only: bool = False,
        log_every_n_steps: int = 20,
        default_root_dir: str = None,
        slurm_auto_requeue: bool = True,
        use_slurm: bool = True,
        precision: _PRECISION_INPUT = "32-true",
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        reload_dataloaders_every_n_epochs=0,
        accumulate_grad_batches: int = 1,
    ):
        self._datamodule = datamodule

        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.model = model
        self.callbacks = callbacks
        self.logger = logger
        self.slurm_auto_requeue = slurm_auto_requeue
        self.ckpt_path = ckpt_path
        self.min_epochs = min_epochs
        self.epochs = epochs
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.enable_checkpointing = enable_checkpointing
        self.rank_zero_only = rank_zero_only
        self.log_every_n_steps = log_every_n_steps
        self.default_root_dir = default_root_dir
        self.precision = precision
        self.deterministic = deterministic
        self.use_slurm = use_slurm
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_val = gradient_clip_val
        self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.accumulate_grad_batches = accumulate_grad_batches

        self._setup()

    def get_datamodule(self):
        return self._datamodule

    # Setter for datamodules
    def set_datamodule(self, datamodule: GenericDataModule):
        self._datamodule = datamodule

    def _setup(self):
        if self.use_slurm:
            if self.slurm_auto_requeue:
                plugins = [SLURMEnvironment(requeue_signal=signal.SIGTERM)]
            else:
                plugins = [SLURMEnvironment(auto_requeue=False)]
        else:
            plugins = []

        self.trainer = LightningTrainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            num_nodes=self.num_nodes,
            callbacks=self.callbacks,
            min_epochs=self.min_epochs,
            max_epochs=self.epochs,
            default_root_dir=self.default_root_dir,
            enable_checkpointing=self.enable_checkpointing,
            log_every_n_steps=self.log_every_n_steps,
            num_sanity_val_steps=0,
            deterministic=self.deterministic,
            precision=self.precision,
            plugins=plugins,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            reload_dataloaders_every_n_epochs=self.reload_dataloaders_every_n_epochs,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )

    def train_model(self):
        if self.ckpt_path is not None:
            if not os.path.isfile(self.ckpt_path):
                self.ckpt_path = None

        if self.slurm_auto_requeue:
            self.trainer._signal_connector = _KlabSignalConnector(
                self.trainer, self.model
            )
            self.trainer._signal_connector.register_signal_handlers()
        if self._datamodule is not None:
            self.trainer.fit(
                self.model,
                datamodule=self.get_datamodule(),
                ckpt_path=self.ckpt_path,
                weights_only=False,
            )

        elif (
            self._datamodule is None
            and self.train_dataloaders is not None
            and self.val_dataloaders is not None
        ):
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloaders,
                val_dataloaders=self.val_dataloaders,
                ckpt_path=self.ckpt_path,
                weights_only=False,
            )
        elif (
            self._datamodule is None
            and self.train_dataloaders is not None
            and self.val_dataloaders is None
        ):
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloaders,
                ckpt_path=self.ckpt_path,
                weights_only=False,
            )
        elif (
            self._datamodule is None
            and self.train_dataloaders is None
            and self.val_dataloaders is not None
        ):
            self.trainer.fit(
                self.model,
                val_dataloaders=self.val_dataloaders,
                ckpt_path=self.ckpt_path,
                weights_only=False,
            )
        else:
            raise ValueError(
                "No datamodule or train or validation dataloaders provided"
            )

    def callback_metrics(self):
        return self.trainer.callback_metrics


class LightningTrainer(Trainer):
    """
    A PyTorch Lightning Trainer subclass for training Lightning models.

    Args:
        accelerator (Union[str, Accelerator]): Accelerator type for distributed training.
        strategy (Union[str, Strategy]): Distributed training strategy.
        devices (Union[List[int], str, int]): Device(s) to use for training.
        num_nodes (int): Number of nodes to use for distributed training.
        precision (_PRECISION_INPUT): Precision for training (e.g., '16', '32', '16-true', '16-false').
        logger (Optional[Union[Logger, Iterable[Logger], bool]]): Logger for recording training logs.
        callbacks (Optional[Union[List[Callback], Callback]]): Callbacks for monitoring/tracking training.
        fast_dev_run (Union[int, bool]): Whether to run a fast development mode.
        max_epochs (Optional[int]): Maximum number of epochs to train the model.
        min_epochs (Optional[int]): Minimum number of epochs to train the model.
        max_steps (int): Maximum number of training steps.
        min_steps (Optional[int]): Minimum number of training steps.
        max_time (Optional[Union[str, timedelta, Dict[str, int]]]): Maximum time for training.
        limit_train_batches (Optional[Union[int, float]]): Limiting training batches.
        limit_val_batches (Optional[Union[int, float]]): Limiting validation batches.
        limit_test_batches (Optional[Union[int, float]]): Limiting test batches.
        limit_predict_batches (Optional[Union[int, float]]): Limiting prediction batches.
        overfit_batches (Union[int, float]): Number of batches to use for overfitting.
        val_check_interval (Optional[Union[int, float]]): Validation check interval.
        check_val_every_n_epoch (Optional[int]): Check validation every n epochs.
        num_sanity_val_steps (Optional[int]): Number of sanity validation steps.
        log_every_n_steps (Optional[int]): Log frequency (in steps).
        enable_checkpointing (Optional[bool]): Whether to enable checkpointing.
        enable_progress_bar (Optional[bool]): Whether to enable progress bar.
        enable_model_summary (Optional[bool]): Whether to enable model summary.
        accumulate_grad_batches (int): Accumulate gradient batches.
        gradient_clip_val (Optional[Union[int, float]]): Value to clip gradients.
        gradient_clip_algorithm (Optional[str]): Algorithm for gradient clipping.
        deterministic (Optional[Union[bool, _LITERAL_WARN]]): Whether to use deterministic training.
        benchmark (Optional[bool]): Whether to use benchmark mode.
        inference_mode (bool): Whether to enable inference mode.
        use_distributed_sampler (bool): Whether to use distributed sampler.
        profiler (Optional[Union[Profiler, str]]): Profiler for profiling training.
        detect_anomaly (bool): Whether to detect anomalies during training.
        barebones (bool): Whether to use barebones mode.
        plugins: Additional plugins for trainer.
        sync_batchnorm (bool): Whether to synchronize batch normalization.
        reload_dataloaders_every_n_epochs (int): Reload dataloaders every n epochs.
        default_root_dir (Optional[_PATH]): Default root directory for logs and checkpoints.

    Attributes:
        All arguments passed to the constructor are available as attributes.

    Raises:
        NotImplementedError: If `accelerator` or `strategy` is set to 'auto'.
    """

    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: _PRECISION_INPUT = "32-true",
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        callbacks: Optional[Union[list[Callback], Callback]] = None,
        fast_dev_run: Union[int, bool] = False,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins=None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,
    ):
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
        )


# copied from signal.pyi
_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[Callable[[_SIGNUM, FrameType], Any], int, signal.Handlers, None]

log = logging.getLogger(__name__)


class _HandlersCompose:
    def __init__(self, signal_handlers: Union[list[_HANDLER], _HANDLER]) -> None:
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum: _SIGNUM, frame: FrameType) -> None:
        for signal_handler in self.signal_handlers:
            if isinstance(signal_handler, int):
                signal_handler = signal.getsignal(signal_handler)
            if callable(signal_handler):
                signal_handler(signum, frame)


class _KlabSignalConnector:
    def __init__(self, trainer: LightningTrainer, model: BaseModule) -> None:
        self.received_sigterm = False
        self.trainer = trainer
        self.model = model
        self._original_handlers: dict[_SIGNUM, _HANDLER] = {}

    def register_signal_handlers(self) -> None:
        self.received_sigterm = False
        self._original_handlers = self._get_current_signal_handlers()

        sigusr_handlers: list[_HANDLER] = []
        sigterm_handlers: list[_HANDLER] = [self._sigterm_notifier_fn]

        environment = self.trainer._accelerator_connector.cluster_environment
        if isinstance(environment, SLURMEnvironment) and environment.auto_requeue:
            log.info("SLURM auto-requeueing enabled. Setting signal handlers.")
            sigusr_handlers.append(self._slurm_sigusr_handler_fn)
            sigterm_handlers.append(self._sigterm_handler_fn)

        sigusr = (
            environment.requeue_signal
            if isinstance(environment, SLURMEnvironment)
            else signal.SIGUSR1
        )
        assert sigusr is not None
        if sigusr_handlers and not self._has_already_handler(sigusr):
            self._register_signal(sigusr, _HandlersCompose(sigusr_handlers))

        # we have our own handler, but include existing ones too
        if self._has_already_handler(signal.SIGTERM):
            sigterm_handlers.append(signal.getsignal(signal.SIGTERM))
        self._register_signal(signal.SIGTERM, _HandlersCompose(sigterm_handlers))

    def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        rank_zero_info(f"Handling auto-requeue signal: {signum}")

        log.info("recieved sigusr, Klabs custom pytorch lightning handler")

        # save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")
        # Save the metrics
        self._copy_files_on_sigterm()

    def _copy_files_on_sigterm(self) -> None:
        log.info("Copying files before handling SIGTERM.")
        present_files = os.listdir(self.trainer.default_root_dir)

        for file in present_files:
            if file.endswith(".npz") or file.endswith(".json"):
                backup_dir = os.path.join(self.trainer.default_root_dir, "backup")
                Path(backup_dir).mkdir(parents=True, exist_ok=True)

                with open(
                    os.path.join(self.trainer.default_root_dir, file), "rb"
                ) as src_file:
                    with open(
                        os.path.join(
                            backup_dir,
                            Path(file).stem
                            + f"_epoch_{self.trainer.current_epoch}_step_{self.trainer.global_step}"
                            + ".npz",
                        ),
                        "wb",
                    ) as dst_file:
                        fcntl.lockf(src_file.fileno(), fcntl.LOCK_SH)
                        shutil.copyfileobj(src_file, dst_file)
                        fcntl.lockf(src_file.fileno(), fcntl.LOCK_UN)

        if self.trainer.is_global_zero:
            # find job id
            array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
            if array_job_id is not None:
                array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
                job_id = f"{array_job_id}_{array_task_id}"
            else:
                job_id = os.environ["SLURM_JOB_ID"]

            cmd = ["scontrol", "requeue", job_id]

            # requeue job
            log.info(f"requeing job {job_id}...")
            try:
                result = call(cmd)
            except FileNotFoundError:
                # This can occur if a subprocess call to `scontrol` is run outside a shell context
                # Re-attempt call (now with shell context). If any error is raised, propagate to user.
                # When running a shell command, it should be passed as a single string.
                joint_cmd = [str(x) for x in cmd]
                result = call(" ".join(joint_cmd), shell=True)

            # print result text
            if result == 0:
                log.info(f"requeued exp {job_id}")
            else:
                log.warning("requeue failed...")

        input()

    def _sigterm_notifier_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        log.info(
            rank_prefixed_message(
                f"Received SIGTERM: {signum}", self.trainer.local_rank
            )
        )
        # subprocesses killing the parent process is not supported, only the parent (rank 0) does it
        if not self.received_sigterm:
            # send the same signal to the subprocesses
            launcher = self.trainer.strategy.launcher
            if launcher is not None:
                launcher.kill(signum)
        self.received_sigterm = True

    def _sigterm_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        log.info(f"Bypassing SIGTERM: {signum}")

    def teardown(self) -> None:
        """Restores the signals that were previously configured before :class:`_SignalConnector` replaced them."""
        for signum, handler in self._original_handlers.items():
            if handler is not None:
                self._register_signal(signum, handler)
        self._original_handlers = {}

    @staticmethod
    def _get_current_signal_handlers() -> dict[_SIGNUM, _HANDLER]:
        """Collects the currently assigned signal handlers."""
        valid_signals = _KlabSignalConnector._valid_signals()
        valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
        return {signum: signal.getsignal(signum) for signum in valid_signals}

    @staticmethod
    def _valid_signals() -> set[signal.Signals]:
        """Returns all valid signals supported on the current platform."""
        return signal.valid_signals()

    @staticmethod
    def _has_already_handler(signum: _SIGNUM) -> bool:
        return signal.getsignal(signum) not in (None, signal.SIG_DFL)

    @staticmethod
    def _register_signal(signum: _SIGNUM, handlers: _HANDLER) -> None:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signum, handlers)  # type: ignore[arg-type]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_original_handlers"] = {}
        return state
