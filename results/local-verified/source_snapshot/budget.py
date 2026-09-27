"""Persistent, conservative API spending guard; amounts are integer nanodollars.

Reservations survive crashes and uncertain HTTP failures. The ledger covers calls
made through this project, not other applications, taxes, or credit purchases.
"""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
import sqlite3
import time
import uuid


NANODOLLARS = 1_000_000_000
# Official Jev 1.13 has a 64k *total* input context, not just the 32k state cap.
# Use 65,536 to conservatively cover both interpretations of "64k".
JEV_MAX_INPUT_TOKENS = 65_536
JEV_INPUT_RATE = 0.042


class BudgetError(RuntimeError):
    """The ledger cannot safely account for a request."""


class BudgetExceeded(BudgetError):
    """A request would exceed the cumulative configured budget."""


def _positive_decimal(value: float, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise BudgetError(f"{name} must be a finite positive number") from None
    if not result.is_finite() or result <= 0:
        raise BudgetError(f"{name} must be a finite positive number")
    return result


class BudgetLedger:
    """Track paid attempts across runs sharing the same SQLite file.

    Increasing the limit (up to $30) never clears spending. A changed token rate
    is refused: silently changing it could invalidate earlier reservations.
    Failed requests intentionally have no automatic refund operation.
    """

    def __init__(
        self,
        path: str | Path = ".haltiq/budget.sqlite",
        limit_usd: float = 5.0,
        rate_per_million: float = JEV_INPUT_RATE,
    ) -> None:
        limit = _positive_decimal(limit_usd, "limit_usd")
        if limit > 30:
            raise BudgetError("limit_usd must not exceed the project's $30 ceiling")
        self._rate = _positive_decimal(rate_per_million, "rate_per_million")
        self.rate_per_million = float(self._rate)
        self._limit_nanos = int((limit * NANODOLLARS).to_integral_value(rounding=ROUND_FLOOR))
        if self._limit_nanos < 1:
            raise BudgetError("limit_usd is smaller than the ledger's accounting precision")
        self.limit_usd = self._limit_nanos / NANODOLLARS
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS attempts ("
                "id TEXT PRIMARY KEY, created_at REAL NOT NULL, reserved_tokens INTEGER NOT NULL, "
                "charge_nanos INTEGER NOT NULL CHECK(charge_nanos >= 0), "
                "actual_tokens INTEGER, settled INTEGER NOT NULL DEFAULT 0)"
            )
            recorded = connection.execute(
                "SELECT value FROM settings WHERE key = 'rate_per_million'"
            ).fetchone()
            if recorded is None:
                connection.execute(
                    "INSERT INTO settings VALUES ('rate_per_million', ?)", (str(self._rate),)
                )
            elif Decimal(recorded[0]) != self._rate:
                raise BudgetError("Token rate differs from the existing ledger; spending was not reset")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            with connection:
                yield connection
        finally:
            connection.close()

    def _cost_nanos(self, input_tokens: int) -> int:
        if type(input_tokens) is not int or input_tokens < 0:
            raise BudgetError("input_tokens must be a nonnegative integer")
        amount = Decimal(input_tokens) * self._rate * NANODOLLARS / 1_000_000
        return int(amount.to_integral_value(rounding=ROUND_CEILING))

    def reserve(self, input_tokens: int = JEV_MAX_INPUT_TOKENS) -> str:
        """Commit a worst-case charge *before* starting one billable HTTP attempt."""
        amount = self._cost_nanos(input_tokens)
        if input_tokens < 1:
            raise BudgetError("A reservation requires at least one input token")
        reservation_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            used = connection.execute("SELECT COALESCE(SUM(charge_nanos), 0) FROM attempts").fetchone()[0]
            if used + amount > self._limit_nanos:
                raise BudgetExceeded(
                    f"Cumulative API budget exhausted: ${used / NANODOLLARS:.6f} used/reserved "
                    f"of ${self.limit_usd:.2f}; next attempt needs ${amount / NANODOLLARS:.6f}"
                )
            connection.execute(
                "INSERT INTO attempts (id, created_at, reserved_tokens, charge_nanos) VALUES (?, ?, ?, ?)",
                (reservation_id, time.time(), input_tokens, amount),
            )
        return reservation_id

    def settle(self, reservation_id: str, input_tokens: int) -> float:
        """Reconcile a valid successful response, releasing unused reservation."""
        amount = self._cost_nanos(input_tokens)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT reserved_tokens, actual_tokens, settled FROM attempts WHERE id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise BudgetError("Unknown budget reservation")
            reserved_tokens, actual_tokens, settled = row
            if input_tokens > reserved_tokens:
                raise BudgetError(
                    "Provider usage exceeded its documented context allowance; reservation retained"
                )
            if settled and actual_tokens != input_tokens:
                raise BudgetError("A settled reservation cannot be changed")
            connection.execute(
                "UPDATE attempts SET actual_tokens = ?, charge_nanos = ?, settled = 1 WHERE id = ?",
                (input_tokens, amount, reservation_id),
            )
        return amount / NANODOLLARS

    def snapshot(self) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(settled),0), "
                "COALESCE(SUM(CASE WHEN settled = 1 THEN charge_nanos ELSE 0 END),0), "
                "COALESCE(SUM(CASE WHEN settled = 0 THEN charge_nanos ELSE 0 END),0), "
                "COALESCE(SUM(actual_tokens),0) FROM attempts"
            ).fetchone()
        attempts, successful, spent, reserved, input_tokens = row
        used = spent + reserved
        return {
            "path": str(self.path),
            "limit_usd": self.limit_usd,
            "rate_per_million": self.rate_per_million,
            "spent_usd": spent / NANODOLLARS,
            "reserved_usd": reserved / NANODOLLARS,
            "used_usd": used / NANODOLLARS,
            "remaining_usd": max(0, self._limit_nanos - used) / NANODOLLARS,
            "attempts": attempts,
            "successful_requests": successful,
            "unsettled_requests": attempts - successful,
            "input_tokens": input_tokens,
        }
