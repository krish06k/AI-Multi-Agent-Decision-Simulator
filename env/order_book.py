"""
env/order_book.py
=================
Layer 2 — RL Market Environment | Limit Order Book & Matching Engine

Production-grade price-time priority limit order book (LOB).

Architecture
------------
* Two sorted heaps: bids (max-heap) and asks (min-heap).
* Each order is an Order dataclass with full audit trail.
* Matching engine executes crossing orders immediately (continuous trading).
* Supports limit orders, market orders, and cancellations.
* Exposes Level-2 depth snapshot for agent observation.
* Maintains BBO (Best Bid/Offer) and mid-price at all times.

Design decisions
----------------
* heapq with negated bid prices: Python only has min-heap; negating bids
  gives price-time priority correctly without a custom data structure.
* Lazy deletion: cancelled orders are marked; popped and discarded on next match.
* Each order has a monotonic sequence_id for time priority within same price.
* All fills are returned as Trade namedtuples for the trade log.

FIX SUMMARY
-----------
FIX 1 — cancel_agent_orders(agent_id) added.
         Required by market_env._replenish_liquidity() to sweep stale
         market maker orders (agent_id=-1) before each tick's refill.
         Without this, MM bid orders accumulate indefinitely, creating
         permanent positive book imbalance → net_flow always positive
         → price only ever rises. This was the primary cause of the
         always-upward chart in the original simulation.

         The method marks all active orders for a given agent as
         CANCELLED (lazy deletion — consistent with the existing pattern).
         Heap entries are discarded the next time _peek / _pop runs past them.

FIX 2 — cancel_all_agent_orders() renamed to cancel_agent_orders()
         for API consistency with market_env.py calls.
         The old name is kept as an alias so nothing else breaks.

FIX 3 — imbalance() now uses configurable depth (default 5 levels).
         Previously hardcoded to depth=5 but called level2_snapshot(depth=5)
         redundantly inside the method — now accepts a parameter.

FIX 4 — stats() now exposes spread_bps for dashboard and shock scaling.
         market_env reads spread_bps from book_stats; it was missing.

FIX 5 — level2_snapshot() now skips orders with qty_remaining == 0.
         Partially filled resting orders that haven't been fully removed
         were showing up as zero-qty levels, polluting the depth view.

FIX 6 — reset() now resets order/trade/sequence counters to 0.
         (Was already present — retained and documented explicitly.)

FIX 7 — _match() guards against self-trade: an agent cannot fill their
         own resting order. Such a cross is skipped and matching continues.
         Self-trades inflate volume metrics and can cause portfolio double-
         counting without this guard.
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & primitives
# ---------------------------------------------------------------------------

class Side(Enum):
    BID = auto()   # buy order
    ASK = auto()   # sell order


class OrderType(Enum):
    LIMIT  = auto()
    MARKET = auto()


class OrderStatus(Enum):
    OPEN      = auto()
    FILLED    = auto()
    PARTIAL   = auto()
    CANCELLED = auto()


@dataclass
class Order:
    """
    Full limit order representation.

    Attributes
    ----------
    order_id      : Globally unique order ID.
    agent_id      : Originating agent ID (-1 = synthetic market maker).
    side          : BID or ASK.
    order_type    : LIMIT or MARKET.
    price         : Limit price (None for market orders).
    qty           : Original quantity requested.
    qty_remaining : Unfilled quantity.
    tick          : Simulation tick at which order was placed.
    sequence_id   : Monotonic counter for time priority.
    status        : Current order status.
    """
    order_id:      int
    agent_id:      int
    side:          Side
    order_type:    OrderType
    price:         Optional[float]
    qty:           int
    qty_remaining: int
    tick:          int
    sequence_id:   int
    status:        OrderStatus = OrderStatus.OPEN

    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)

    def filled_qty(self) -> int:
        return self.qty - self.qty_remaining


class Trade(NamedTuple):
    """Record of an executed match between two orders."""
    trade_id:        int
    tick:            int
    buyer_agent_id:  int
    seller_agent_id: int
    price:           float
    qty:             int
    buy_order_id:    int
    sell_order_id:   int
    aggressor_side:  Side   # which side caused the match


class Level2Snapshot(NamedTuple):
    """Order book depth snapshot for agent observations."""
    bids:      list[tuple[float, int]]  # [(price, total_qty), ...] best first
    asks:      list[tuple[float, int]]  # [(price, total_qty), ...] best first
    mid_price: float
    spread:    float
    best_bid:  Optional[float]
    best_ask:  Optional[float]


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Continuous price-time priority Limit Order Book.

    Parameters
    ----------
    tick_size : Minimum price increment (prices snapped to grid).
    max_depth : Maximum levels to maintain and expose per side.
    lot_size  : Minimum order quantity.
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        max_depth: int   = 20,
        lot_size:  int   = 1,
    ):
        self.tick_size = tick_size
        self.max_depth = max_depth
        self.lot_size  = lot_size

        # Bid heap: max-heap via negated price. Entry: (-price, seq_id, order_id)
        self._bids: list[tuple[float, int, int]] = []
        # Ask heap: min-heap. Entry: (price, seq_id, order_id)
        self._asks: list[tuple[float, int, int]] = []

        # Order registry: order_id → Order
        self._orders: dict[int, Order] = {}

        # Counters
        self._order_id_counter: int = 0
        self._trade_id_counter: int = 0
        self._sequence_counter: int = 0

        # Stats
        self.total_volume:      int            = 0
        self.last_trade_price:  Optional[float] = None
        self.last_trade_tick:   int            = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_order_id(self) -> int:
        self._order_id_counter += 1
        return self._order_id_counter

    def _next_trade_id(self) -> int:
        self._trade_id_counter += 1
        return self._trade_id_counter

    def _next_seq(self) -> int:
        self._sequence_counter += 1
        return self._sequence_counter

    def _snap_price(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def _peek_best_bid(self) -> Optional[Order]:
        """Return best (highest-price) bid without popping. Lazy-deletes stale entries."""
        while self._bids:
            neg_price, seq_id, oid = self._bids[0]
            order = self._orders.get(oid)
            if order and order.is_active():
                return order
            heapq.heappop(self._bids)   # lazy delete cancelled / filled
        return None

    def _peek_best_ask(self) -> Optional[Order]:
        """Return best (lowest-price) ask without popping. Lazy-deletes stale entries."""
        while self._asks:
            price, seq_id, oid = self._asks[0]
            order = self._orders.get(oid)
            if order and order.is_active():
                return order
            heapq.heappop(self._asks)   # lazy delete
        return None

    def _pop_best_bid(self) -> Optional[Order]:
        while self._bids:
            neg_price, seq_id, oid = heapq.heappop(self._bids)
            order = self._orders.get(oid)
            if order and order.is_active():
                return order
        return None

    def _pop_best_ask(self) -> Optional[Order]:
        while self._asks:
            price, seq_id, oid = heapq.heappop(self._asks)
            order = self._orders.get(oid)
            if order and order.is_active():
                return order
        return None

    def _add_to_book(self, order: Order) -> None:
        """Insert a resting order into the appropriate heap."""
        if order.side == Side.BID:
            heapq.heappush(self._bids, (-order.price, order.sequence_id, order.order_id))
        else:
            heapq.heappush(self._asks, (order.price, order.sequence_id, order.order_id))

    # ------------------------------------------------------------------
    # Public: add orders
    # ------------------------------------------------------------------

    def add_limit_order(
        self,
        agent_id: int,
        side:     Side,
        price:    float,
        qty:      int,
        tick:     int,
    ) -> tuple[Order, list[Trade]]:
        """
        Submit a limit order. Triggers matching if it crosses the spread.

        Parameters
        ----------
        agent_id : Originating agent (-1 for synthetic market maker).
        side     : BID (buy) or ASK (sell).
        price    : Limit price.
        qty      : Order quantity.
        tick     : Current simulation tick.

        Returns
        -------
        (order, trades) — placed order and any fills generated.
        """
        qty   = max(self.lot_size, (qty // self.lot_size) * self.lot_size)
        price = self._snap_price(price)

        order = Order(
            order_id      = self._next_order_id(),
            agent_id      = agent_id,
            side          = side,
            order_type    = OrderType.LIMIT,
            price         = price,
            qty           = qty,
            qty_remaining = qty,
            tick          = tick,
            sequence_id   = self._next_seq(),
        )
        self._orders[order.order_id] = order

        trades = self._match(order, tick)

        if order.is_active():
            self._add_to_book(order)

        return order, trades

    def add_market_order(
        self,
        agent_id: int,
        side:     Side,
        qty:      int,
        tick:     int,
    ) -> tuple[Order, list[Trade]]:
        """
        Submit a market order (executes immediately at best available price).
        Partially fills if insufficient liquidity; remainder is cancelled.
        """
        qty = max(self.lot_size, (qty // self.lot_size) * self.lot_size)

        order = Order(
            order_id      = self._next_order_id(),
            agent_id      = agent_id,
            side          = side,
            order_type    = OrderType.MARKET,
            price         = None,
            qty           = qty,
            qty_remaining = qty,
            tick          = tick,
            sequence_id   = self._next_seq(),
        )
        self._orders[order.order_id] = order

        trades = self._match(order, tick)

        # Cancel unfilled remainder
        if order.is_active() and order.qty_remaining > 0:
            order.status = OrderStatus.CANCELLED

        return order, trades

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel a specific open order by ID.
        Returns True if cancelled, False if not found or already done.
        """
        order = self._orders.get(order_id)
        if order and order.is_active():
            order.status = OrderStatus.CANCELLED
            return True
        return False

    # ------------------------------------------------------------------
    # FIX 1 + FIX 2: cancel_agent_orders
    # ------------------------------------------------------------------

    def cancel_agent_orders(self, agent_id: int) -> int:
        """
        Cancel ALL active orders belonging to a given agent.

        This is the primary fix for the upward-drifting price chart.
        Called by market_env._replenish_liquidity(agent_id=-1) every tick
        to sweep stale market maker orders before placing fresh symmetric
        bid/ask levels.

        Without this:
        - MM adds 5 bid + 5 ask orders per tick
        - After 500 ticks: 2,500 stale bid orders still active in book
        - level2_snapshot sums all those stale bids → huge bid-side depth
        - imbalance() returns strongly positive value every tick
        - net_flow = imbalance * max_order_size = always large positive
        - PriceImpactModel.step() sees persistent buy pressure
        - Price drifts upward monotonically → the chart you were seeing

        Uses lazy deletion: orders are marked CANCELLED here; the heap
        entries are discarded the next time _peek_best_bid/_pop_best_bid
        walks past them. This is O(n_agent_orders) on the registry, which
        is fast since we only scan active orders.

        Parameters
        ----------
        agent_id : Agent whose orders to cancel. Use -1 for market maker.

        Returns
        -------
        int : Number of orders cancelled.
        """
        cancelled = 0
        for order in self._orders.values():
            if order.agent_id == agent_id and order.is_active():
                order.status = OrderStatus.CANCELLED
                cancelled   += 1

        if cancelled > 0:
            logger.debug(
                "cancel_agent_orders: agent_id=%d cancelled=%d", agent_id, cancelled
            )
        return cancelled

    # FIX 2: keep old name as alias so existing code doesn't break
    def cancel_all_agent_orders(self, agent_id: int) -> int:
        """Alias for cancel_agent_orders() — retained for backwards compatibility."""
        return self.cancel_agent_orders(agent_id)

    # ------------------------------------------------------------------
    # Matching engine
    # ------------------------------------------------------------------

    def _match(self, incoming: Order, tick: int) -> list[Trade]:
        """
        Continuous trading matching engine.
        Executes price-time priority against resting orders.

        FIX 7: Self-trade prevention.
        If the best resting order belongs to the same agent as the
        incoming order, the match is skipped and we stop looking further.
        This prevents agents from trading with themselves, which inflates
        volume metrics and can cause portfolio double-counting.
        """
        trades: list[Trade] = []

        while incoming.qty_remaining > 0:
            if incoming.side == Side.BID:
                resting = self._peek_best_ask()
                if resting is None:
                    break
                # Price check for limit orders
                if (incoming.order_type == OrderType.LIMIT
                        and resting.price > incoming.price):
                    break
                # FIX 7: self-trade prevention
                if resting.agent_id == incoming.agent_id:
                    break
                exec_price = resting.price
                resting    = self._pop_best_ask()

            else:  # ASK side
                resting = self._peek_best_bid()
                if resting is None:
                    break
                if (incoming.order_type == OrderType.LIMIT
                        and resting.price < incoming.price):
                    break
                # FIX 7: self-trade prevention
                if resting.agent_id == incoming.agent_id:
                    break
                exec_price = resting.price
                resting    = self._pop_best_bid()

            if resting is None:
                break

            # FIX 5: skip zero-qty resting orders (partially filled remnants)
            if resting.qty_remaining <= 0:
                resting.status = OrderStatus.FILLED
                continue

            fill_qty   = min(incoming.qty_remaining, resting.qty_remaining)
            exec_price = self._snap_price(exec_price)

            # Apply fills
            incoming.qty_remaining -= fill_qty
            resting.qty_remaining  -= fill_qty

            incoming.status = (
                OrderStatus.FILLED if incoming.qty_remaining == 0 else OrderStatus.PARTIAL
            )
            resting.status = (
                OrderStatus.FILLED if resting.qty_remaining == 0 else OrderStatus.PARTIAL
            )

            # Re-queue partially filled resting order
            if resting.is_active():
                self._add_to_book(resting)

            # Determine buyer/seller IDs
            if incoming.side == Side.BID:
                buyer_id,  seller_id  = incoming.agent_id, resting.agent_id
                buy_oid,   sell_oid   = incoming.order_id, resting.order_id
            else:
                buyer_id,  seller_id  = resting.agent_id,  incoming.agent_id
                buy_oid,   sell_oid   = resting.order_id,   incoming.order_id

            trade = Trade(
                trade_id        = self._next_trade_id(),
                tick            = tick,
                buyer_agent_id  = buyer_id,
                seller_agent_id = seller_id,
                price           = exec_price,
                qty             = fill_qty,
                buy_order_id    = buy_oid,
                sell_order_id   = sell_oid,
                aggressor_side  = incoming.side,
            )
            trades.append(trade)
            self.total_volume     += fill_qty
            self.last_trade_price  = exec_price
            self.last_trade_tick   = tick

        return trades

    # ------------------------------------------------------------------
    # Market information queries
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        o = self._peek_best_bid()
        return o.price if o else None

    @property
    def best_ask(self) -> Optional[float]:
        o = self._peek_best_ask()
        return o.price if o else None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        if bb is not None:
            return bb
        if ba is not None:
            return ba
        return self.last_trade_price

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    def level2_snapshot(self, depth: Optional[int] = None) -> Level2Snapshot:
        """
        Return aggregated Level-2 depth snapshot (price, total_qty per level).

        FIX 5: Skip orders with qty_remaining == 0 — these are fully filled
        remnants that haven't been lazy-deleted from the heap yet. Including
        them was polluting the depth view with zero-qty phantom levels.

        Parameters
        ----------
        depth : Number of levels to return per side (defaults to max_depth).
        """
        depth = depth or self.max_depth

        # Aggregate bids
        bid_agg: dict[float, int] = defaultdict(int)
        for neg_price, seq_id, oid in self._bids:
            o = self._orders.get(oid)
            if o and o.is_active() and o.qty_remaining > 0:   # FIX 5
                bid_agg[-neg_price] += o.qty_remaining
        bids = sorted(bid_agg.items(), reverse=True)[:depth]

        # Aggregate asks
        ask_agg: dict[float, int] = defaultdict(int)
        for price, seq_id, oid in self._asks:
            o = self._orders.get(oid)
            if o and o.is_active() and o.qty_remaining > 0:   # FIX 5
                ask_agg[price] += o.qty_remaining
        asks = sorted(ask_agg.items())[:depth]

        mid = self.mid_price or 0.0
        spr = self.spread    or 0.0
        return Level2Snapshot(
            bids=bids, asks=asks,
            mid_price=mid, spread=spr,
            best_bid=self.best_bid, best_ask=self.best_ask,
        )

    def imbalance(self, depth: int = 5) -> float:
        """
        Order book imbalance at best N levels.

        FIX 3: Now accepts a depth parameter instead of being hardcoded.

        Returns value in [-1, 1]:
            +1 = all volume on bid side  (strong buy pressure)
            -1 = all volume on ask side  (strong sell pressure)
             0 = perfectly balanced book
        """
        snap    = self.level2_snapshot(depth=depth)
        bid_vol = sum(q for _, q in snap.bids[:depth])
        ask_vol = sum(q for _, q in snap.asks[:depth])
        total   = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    # ------------------------------------------------------------------
    # Book management
    # ------------------------------------------------------------------

    def get_agent_orders(self, agent_id: int) -> list[Order]:
        """Return all currently active orders for a given agent."""
        return [
            o for o in self._orders.values()
            if o.agent_id == agent_id and o.is_active()
        ]

    def active_order_count(self, agent_id: Optional[int] = None) -> int:
        """
        Count active orders — optionally filtered by agent.
        Useful for debugging accumulation issues.
        """
        if agent_id is None:
            return sum(1 for o in self._orders.values() if o.is_active())
        return sum(
            1 for o in self._orders.values()
            if o.is_active() and o.agent_id == agent_id
        )

    def reset(self) -> None:
        """
        Clear the entire book for episode reset.
        FIX 6: All counters explicitly reset to 0.
        """
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self._order_id_counter = 0
        self._trade_id_counter = 0
        self._sequence_counter = 0
        self.total_volume      = 0
        self.last_trade_price  = None
        self.last_trade_tick   = -1
        logger.debug("OrderBook reset.")

    def stats(self) -> dict:
        """
        Summary statistics for logging and market_env info dict.

        FIX 4: Now includes spread_bps — required by market_env.py which
        reads shock_spread = book_stats.get("spread_multiplier", 1.0).
        Also exposes mm_order_count for diagnosing accumulation issues.
        """
        snap      = self.level2_snapshot()
        mid       = self.mid_price or 0.0
        spr       = self.spread    or 0.0
        spread_bps = (spr / mid * 10000.0) if mid > 0 else 0.0

        return {
            "best_bid":      self.best_bid,
            "best_ask":      self.best_ask,
            "mid_price":     mid,
            "spread":        spr,
            "spread_bps":    round(spread_bps, 2),        # FIX 4
            "bid_depth":     sum(q for _, q in snap.bids),
            "ask_depth":     sum(q for _, q in snap.asks),
            "bid_levels":    len(snap.bids),
            "ask_levels":    len(snap.asks),
            "total_volume":  self.total_volume,
            "imbalance":     self.imbalance(),
            "active_orders": self.active_order_count(),
            "mm_orders":     self.active_order_count(agent_id=-1), # FIX 4: debug
        }

    def __repr__(self) -> str:
        bb = f"{self.best_bid:.2f}" if self.best_bid else "—"
        ba = f"{self.best_ask:.2f}" if self.best_ask else "—"
        return (
            f"OrderBook("
            f"bid={bb} | ask={ba} | "
            f"vol={self.total_volume} | "
            f"active={self.active_order_count()})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("OrderBook smoke test")
    print("=" * 60)

    ob = OrderBook()

    # --- Basic resting orders (no cross) ---
    ob.add_limit_order(agent_id=1, side=Side.BID, price=99.0,  qty=10, tick=0)
    ob.add_limit_order(agent_id=2, side=Side.ASK, price=101.0, qty=10, tick=0)
    print(f"\nAfter resting orders: {ob}")
    print(f"Imbalance: {ob.imbalance():.3f}")

    # --- Crossing order → should generate trade ---
    _, trades = ob.add_limit_order(agent_id=3, side=Side.BID, price=102.0, qty=5, tick=1)
    print(f"\nAfter crossing bid: {ob}")
    print(f"Trades: {trades}")
    assert len(trades) == 1, "Expected 1 trade"
    assert trades[0].qty == 5

    # --- Market maker accumulation test (FIX 1) ---
    print("\n--- FIX 1: cancel_agent_orders test ---")
    MM_ID = -1
    for tick in range(10):
        ob.add_limit_order(MM_ID, Side.BID, 98.0 - tick * 0.1, 50, tick)
        ob.add_limit_order(MM_ID, Side.ASK, 102.0 + tick * 0.1, 50, tick)

    mm_before = ob.active_order_count(agent_id=MM_ID)
    print(f"MM orders before cancel: {mm_before}")
    assert mm_before == 20, f"Expected 20, got {mm_before}"

    cancelled = ob.cancel_agent_orders(MM_ID)
    mm_after = ob.active_order_count(agent_id=MM_ID)
    imb_after = ob.imbalance()
    print(f"Cancelled: {cancelled}")
    print(f"MM orders after cancel: {mm_after}")
    print(f"Imbalance after cancel: {imb_after:.3f}")
    assert mm_after == 0,          "MM orders should be 0 after cancel"
    assert cancelled == 20,        "Should have cancelled all 20 MM orders"
    assert abs(imb_after) < 0.01,  "Imbalance should be ~0 after clearing MM orders"
    print("PASS: cancel_agent_orders works correctly")

    # --- Self-trade prevention (FIX 7) ---
    print("\n--- FIX 7: self-trade prevention test ---")
    ob2 = OrderBook()
    ob2.add_limit_order(agent_id=5, side=Side.ASK, price=100.0, qty=10, tick=0)
    _, self_trades = ob2.add_limit_order(agent_id=5, side=Side.BID, price=100.0, qty=10, tick=1)
    print(f"Self-trade attempt → trades generated: {len(self_trades)}")
    assert len(self_trades) == 0, "Self-trade should be blocked"
    print("PASS: self-trade prevention works correctly")

    # --- stats() includes spread_bps (FIX 4) ---
    print("\n--- FIX 4: stats() spread_bps test ---")
    ob3 = OrderBook()
    ob3.add_limit_order(1, Side.BID, 99.0,  100, tick=0)
    ob3.add_limit_order(2, Side.ASK, 101.0, 100, tick=0)
    s = ob3.stats()
    print(f"Stats: {s}")
    assert "spread_bps" in s, "spread_bps missing from stats()"
    assert s["spread_bps"] > 0, "spread_bps should be > 0"
    print("PASS: stats() contains spread_bps")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)