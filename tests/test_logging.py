"""Real-time pipeline logging: stage markers, failures, rank files and the training callback."""

import logging

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from onboard_vlm.utils.callbacks import PipelineLoggingCallback
from onboard_vlm.utils.logging import describe_module, fmt_metrics, get_logger, setup_logging, stage


@pytest.fixture
def run_dir(tmp_path):
    log = setup_logging(str(tmp_path))
    yield tmp_path, log
    for h in list(logging.getLogger("onboard_vlm").handlers):  # leave other tests' logging untouched
        h.close()
        logging.getLogger("onboard_vlm").removeHandler(h)


def _text(tmp_path):
    for h in logging.getLogger("onboard_vlm").handlers:
        h.flush()
    return (tmp_path / "run_rank0.log").read_text()


def test_stage_logs_start_end_and_failure(run_dir):
    tmp_path, _ = run_dir
    log = get_logger("onboard_vlm.test")
    with stage("Loading something", log):
        log.info("     inside")
    with pytest.raises(RuntimeError):
        with stage("Broken step", log):
            raise RuntimeError("boom")
    text = _text(tmp_path)
    assert "===> Loading something" in text and "<=== Loading something done in" in text and "inside" in text
    assert "===> Broken step" in text and "Broken step FAILED" in text and "[rank0][onboard_vlm.test]" in text


def test_describe_and_metrics_formatting():
    assert describe_module(torch.nn.Linear(10, 10)).startswith("Linear: 0.00M params (0.00M trainable; float32, cpu)")
    assert fmt_metrics({"train/loss": torch.tensor(0.5), "val/x": 1.0}, ["train"]) == "train/loss=0.5"


class _Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)

    def training_step(self, batch, _):
        loss = self.lin(batch).square().mean()
        self.log("train/total", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_callback_reports_progress(run_dir):
    tmp_path, _ = run_dir
    tr = pl.Trainer(max_steps=3, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                    enable_model_summary=False, callbacks=[PipelineLoggingCallback(every_n_steps=1)])
    tr.fit(_Tiny(), DataLoader(torch.randn(12, 4), batch_size=4))
    text = _text(tmp_path)
    assert "===> Training: _Tiny" in text and "step 1/3" in text and "step 3/3" in text and "ETA" in text
    assert "<=== Training finished" in text
