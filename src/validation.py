from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_requested_inputs(tickers: list[str], start: date, end: date) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if start >= end:
        errors.append("Baslangic tarihi bitis tarihinden once olmali.")
    if len(tickers) < 2:
        errors.append("Portfoy analizi icin en az iki varlik girilmeli.")
    if (end - start).days < 90:
        warnings.append("Tarih araligi kisa; risk ve getiri metrikleri daha oynak olabilir.")

    return ValidationResult(errors=errors, warnings=warnings)


def validate_clean_dataset(prices: pd.DataFrame, returns: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if prices.shape[1] < 2:
        errors.append("Analiz icin en az iki kullanilabilir varlik gerekiyor.")
    if returns.empty or len(returns) < 20:
        errors.append("Analiz icin yeterli getiri gozlemi bulunamadi.")
    elif len(returns) < 63:
        warnings.append("Veri seti 63 islem gununden kisa; rolling risk grafigi sinirli olabilir.")

    missing_ratio = prices.isna().mean().max() if not prices.empty else 1
    if missing_ratio > 0.1:
        warnings.append("Veri setinde yuksek eksik fiyat orani var; sonuclar etkilenebilir.")

    return ValidationResult(errors=errors, warnings=warnings)


def concentration_warnings(weights: np.ndarray, assets: list[str]) -> list[str]:
    if len(weights) == 0:
        return []
    top_index = int(np.argmax(weights))
    if weights[top_index] >= 0.65:
        return [f"Portfoy {assets[top_index]} uzerinde yogunlasmis; cesitlendirme riski yuksek."]
    return []
