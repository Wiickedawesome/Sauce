"""
test_accounting.py — Comprehensive tests for sauce.accounting module.

Covers fee/slippage BPS model, asset class normalization,
round-trip trade accounting, and edge cases.
"""

from __future__ import annotations

import pytest

from sauce.accounting import (
    ExecutionCosts,
    TradeAccounting,
    _impact_multiplier,
    _normalize_asset_class,
    estimate_side_costs,
    estimate_trade_accounting,
)


# ── Asset class normalization ─────────────────────────────────────────────────


class TestNormalizeAssetClass:
    def test_equity_passthrough(self):
        assert _normalize_asset_class("equity") == "equity"

    def test_crypto_passthrough(self):
        assert _normalize_asset_class("crypto") == "crypto"

    def test_option_passthrough(self):
        assert _normalize_asset_class("option") == "option"

    def test_options_plural_normalized(self):
        assert _normalize_asset_class("options") == "option"

    def test_case_insensitive(self):
        assert _normalize_asset_class("CRYPTO") == "crypto"
        assert _normalize_asset_class("Equity") == "equity"
        assert _normalize_asset_class("OPTIONS") == "option"

    def test_whitespace_stripped(self):
        assert _normalize_asset_class("  crypto  ") == "crypto"

    def test_unknown_defaults_to_equity(self):
        assert _normalize_asset_class("forex") == "equity"
        assert _normalize_asset_class("") == "equity"


# ── Side cost estimation ──────────────────────────────────────────────────────


class TestImpactMultiplier:
    """Unit tests for the tiered market-impact model."""

    def test_seed_tier_no_uplift(self):
        assert _impact_multiplier(0.0) == pytest.approx(1.0)
        assert _impact_multiplier(999.99) == pytest.approx(1.0)

    def test_scaling_tier(self):
        assert _impact_multiplier(1_000.0) == pytest.approx(1.5)
        assert _impact_multiplier(5_000.0) == pytest.approx(1.5)
        assert _impact_multiplier(9_999.99) == pytest.approx(1.5)

    def test_growth_tier(self):
        assert _impact_multiplier(10_000.0) == pytest.approx(2.0)
        assert _impact_multiplier(49_999.99) == pytest.approx(2.0)

    def test_institutional_tier(self):
        assert _impact_multiplier(50_000.0) == pytest.approx(3.0)
        assert _impact_multiplier(1_000_000.0) == pytest.approx(3.0)


class TestEstimateSideCosts:
    def test_equity_costs_seed_tier(self):
        # equity: fee=0 bps, slippage=5 bps, notional $100 → 1.0× impact
        costs = estimate_side_costs("equity", 100.0)
        assert costs.notional == 100.0
        assert costs.fees_paid == pytest.approx(0.0)
        assert costs.slippage_paid == pytest.approx(0.05)  # 100 * 5/10000 * 1.0
        assert costs.total_cost == pytest.approx(0.05)

    def test_equity_costs_scaling_tier(self):
        # $5k order → 1.5× slippage
        costs = estimate_side_costs("equity", 5_000.0)
        assert costs.slippage_paid == pytest.approx(3.75)   # 5000 * 5/10000 * 1.5
        assert costs.fees_paid == pytest.approx(0.0)  # fees stay flat

    def test_equity_costs_growth_tier(self):
        # $10k order is exactly in the 2.0× tier
        costs = estimate_side_costs("equity", 10_000.0)
        assert costs.slippage_paid == pytest.approx(10.0)   # 10000 * 5/10000 * 2.0
        assert costs.fees_paid == pytest.approx(0.0)

    def test_crypto_costs_seed_tier(self):
        # crypto: fee=15 bps, slippage=15 bps, notional $100 → 1.0×
        costs = estimate_side_costs("crypto", 100.0)
        assert costs.fees_paid == pytest.approx(0.15)
        assert costs.slippage_paid == pytest.approx(0.15)

    def test_crypto_costs_scaling_tier(self):
        costs = estimate_side_costs("crypto", 5_000.0)
        assert costs.fees_paid == pytest.approx(7.50)       # flat
        assert costs.slippage_paid == pytest.approx(11.25)  # 5000 * 15/10000 * 1.5

    def test_crypto_costs_growth_tier(self):
        # $10k is 2.0× tier
        costs = estimate_side_costs("crypto", 10_000.0)
        assert costs.fees_paid == pytest.approx(15.0)       # flat
        assert costs.slippage_paid == pytest.approx(30.0)   # 10000 * 15/10000 * 2.0

    def test_option_costs_growth_tier(self):
        # $10k is 2.0× tier
        costs = estimate_side_costs("option", 10_000.0)
        assert costs.fees_paid == pytest.approx(50.0)       # flat
        assert costs.slippage_paid == pytest.approx(100.0)  # 10000 * 50/10000 * 2.0

    def test_fees_never_scaled(self):
        """Fees must be identical regardless of multiplier tier."""
        seed = estimate_side_costs("crypto", 500.0)       # 1.0×
        scaling = estimate_side_costs("crypto", 5_000.0)  # 1.5×
        growth = estimate_side_costs("crypto", 20_000.0)  # 2.0×
        # Fee BPS is 15 bps each. Notional ratio is 1:10:40 — fees should scale linearly.
        assert scaling.fees_paid == pytest.approx(seed.fees_paid * 10)
        assert growth.fees_paid == pytest.approx(seed.fees_paid * 40)
        # But slippage-to-fee ratio grows with notional.
        assert scaling.slippage_paid > scaling.fees_paid * 1.0  # 1.5× uplift
        assert growth.slippage_paid > growth.fees_paid * 1.0    # 2.0× uplift

    def test_negative_notional_clamped_to_zero(self):
        costs = estimate_side_costs("equity", -500.0)
        assert costs.notional == 0.0
        assert costs.fees_paid == 0.0
        assert costs.slippage_paid == 0.0

    def test_zero_notional(self):
        costs = estimate_side_costs("crypto", 0.0)
        assert costs.total_cost == 0.0


# ── Round-trip trade accounting ───────────────────────────────────────────────


class TestEstimateTradeAccounting:
    def test_profitable_equity_trade(self):
        # entry: $1000 → 1.5× tier; exit: $1100 → 1.5× tier
        ta = estimate_trade_accounting("equity", qty=10, entry_price=100.0, exit_price=110.0)
        assert ta.asset_class == "equity"
        assert ta.multiplier == 1.0
        assert ta.entry_notional == pytest.approx(1_000.0)
        assert ta.exit_notional == pytest.approx(1_100.0)
        assert ta.gross_pnl == pytest.approx(100.0)
        # fees: equity=0 bps → $0
        # slippage: 5 bps × 1.5× impact each side
        #   entry: 1000 * 5/10000 * 1.5 = 0.75
        #   exit:  1100 * 5/10000 * 1.5 = 0.825
        assert ta.slippage_paid == pytest.approx(1.575)
        assert ta.fees_paid == pytest.approx(0.0)
        assert ta.realized_pnl == pytest.approx(100.0 - 1.575)

    def test_losing_crypto_trade(self):
        ta = estimate_trade_accounting("crypto", qty=0.5, entry_price=60_000.0, exit_price=58_000.0)
        assert ta.gross_pnl == pytest.approx(-1_000.0)
        assert ta.realized_pnl < ta.gross_pnl  # costs make it worse

    def test_option_contract_multiplier(self):
        ta = estimate_trade_accounting("option", qty=1, entry_price=5.0, exit_price=6.0)
        assert ta.multiplier == 100.0
        assert ta.entry_notional == pytest.approx(500.0)
        assert ta.exit_notional == pytest.approx(600.0)
        assert ta.gross_pnl == pytest.approx(100.0)

    def test_custom_contract_multiplier(self):
        ta = estimate_trade_accounting(
            "option", qty=1, entry_price=5.0, exit_price=6.0,
            contract_multiplier=10.0,
        )
        assert ta.multiplier == 10.0
        assert ta.gross_pnl == pytest.approx(10.0)

    def test_zero_qty_ignored(self):
        ta = estimate_trade_accounting("equity", qty=0, entry_price=100.0, exit_price=110.0)
        assert ta.gross_pnl == 0.0
        assert ta.realized_pnl == 0.0

    def test_negative_qty_clamped(self):
        ta = estimate_trade_accounting("equity", qty=-5, entry_price=100.0, exit_price=110.0)
        assert ta.qty == 0.0

    def test_negative_prices_clamped(self):
        ta = estimate_trade_accounting("equity", qty=10, entry_price=-5.0, exit_price=10.0)
        assert ta.entry_notional == 0.0
        # gross_pnl still uses raw prices for direction: (10 - (-5)) * 10 = 150
        assert ta.gross_pnl == pytest.approx(150.0)

    def test_flat_trade(self):
        """Entry == exit → gross P&L is 0, costs still apply."""
        ta = estimate_trade_accounting("crypto", qty=1.0, entry_price=100.0, exit_price=100.0)
        assert ta.gross_pnl == 0.0
        assert ta.realized_pnl < 0.0  # costs make it negative
