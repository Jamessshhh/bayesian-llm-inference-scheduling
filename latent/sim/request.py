from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Request:
    id: int
    arrival_time: float
    service_time: float              # TRUE cost (only the oracle may read this)
    input_tokens: int = 0
    output_tokens: int = 0
    predicted_service: Optional[float] = None   # predicted MEDIAN cost (what a real scheduler sees)
    pred_sigma: Optional[float] = None          # predictor's claimed log-scale uncertainty
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def wait(self) -> float:
        return self.start_time - self.arrival_time

    @property
    def latency(self) -> float:
        return self.finish_time - self.arrival_time