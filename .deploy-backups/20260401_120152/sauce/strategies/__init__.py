"""strategies — pluggable trading strategy implementations."""

from sauce.strategies.crypto_momentum import CryptoMomentumReversion
from sauce.strategies.equity_momentum import EquityMomentum

__all__ = ["CryptoMomentumReversion", "EquityMomentum"]
