"""
genai/narrator.py
=================
100% local, zero-API, self-training LSTM narrator.

How it works:
  1. Runs rule-based commentary for the first 500 ticks (warmup)
  2. Silently collects (features, text) pairs from every tick
  3. Auto-trains an LSTM on that collected data
  4. After training, generates all commentary locally via LSTM
  5. Retrains every 200 new ticks to keep improving
  6. Saves model + vocab to disk so training survives restarts

No internet. No API key. No quota. Runs entirely on your machine.
"""

from __future__ import annotations

import os
import time
import random
import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

_HISTORY_MAX      = 200
_DEFAULT_INTERVAL = 10       # fire narrator every N ticks
_REGIME_COOLDOWN  = 10
_MIN_TRAIN_TICKS  = 500      # collect this many ticks before first train
_RETRAIN_EVERY    = 200      # retrain every N new ticks after first train
_SEQ_LEN          = 20       # LSTM looks back this many ticks
_EMBED_DIM        = 64       # word embedding size
_HIDDEN_SIZE      = 256      # LSTM hidden units
_NUM_LAYERS       = 2        # LSTM layers
_DROPOUT          = 0.3
_EPOCHS           = 25       # training epochs per cycle
_BATCH_SIZE       = 32
_LR               = 0.001
_MAX_TEXT_LEN     = 30       # max words in generated sentence
_TEMPERATURE      = 0.8      # sampling temperature (higher = more creative)

_MODEL_PATH = Path("narrator_lstm.pt")
_VOCAB_PATH = Path("narrator_vocab.pkl")
_DATA_PATH  = Path("narrator_data.pkl")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NarratorEntry:
    tick:      int
    text:      str
    trigger:   str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TickData:
    tick:                int
    price:               float
    price_change:        float
    buy_volume:          int
    sell_volume:         int
    hold_count:          int
    shock_active:        bool
    shock_regime:        str
    dominant_action:     str
    dominant_agent_type: str
    notable_events:      list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule-based text engine (used during warmup AND as training labels)
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "shock": [
        "Price dislocates violently as {shock} shock tears through the order book at ${price:.2f}.",
        "A {shock} shock detonates at tick {tick}, sending price lurching to ${price:.2f}.",
        "Tape fractures on {shock} event — cascading flow drives price to ${price:.2f} ({change:+.2f}).",
        "Market structure breaks under {shock} pressure, printing ${price:.2f} on panic volume.",
        "{shock_cap} shock overwhelms the book — price collapses to ${price:.2f} in a single move.",
    ],
    "buy_heavy": [
        "Aggressive buyers dominate — {buys} bids absorbed as price firms at ${price:.2f}.",
        "Buy-side conviction builds; {buys} orders push tape higher to ${price:.2f} ({change:+.2f}).",
        "{agent} agents accumulate hard — {buys} buys vs {sells} sells lift price to ${price:.2f}.",
        "Bid pressure overwhelms the offer — tape prints ${price:.2f} on {buys} buy orders.",
        "Sustained buying from {agent} agents drives price to ${price:.2f} with {buys} orders filled.",
    ],
    "sell_heavy": [
        "Sellers in control — {sells} offers hit as price retreats to ${price:.2f} ({change:+.2f}).",
        "{agent} agents unload aggressively; {sells} sell orders drag tape to ${price:.2f}.",
        "Distribution in play — {sells} sells overwhelm {buys} buys, price slides to ${price:.2f}.",
        "Offer-side dominates; tape prints ${price:.2f} as {sells} sell orders absorb all bids.",
        "Selling pressure mounts — {agent} agents dump positions, price falls to ${price:.2f}.",
    ],
    "balanced": [
        "Tape drifts to ${price:.2f} ({change:+.2f}) as {agent} agents trade in balanced flow.",
        "Quiet session — {buys} buys, {sells} sells — market settles at ${price:.2f}.",
        "{agent_cap} agents hold the line; {momentum} drift brings price to ${price:.2f}.",
        "Mixed flow keeps price near ${price:.2f} ({change:+.2f}) with no clear directional bias.",
        "Market digests recent moves at ${price:.2f} — {buys} bids meet {sells} offers in equilibrium.",
    ],
    "volatile": [
        "Erratic price action — tape swings to ${price:.2f} ({change:+.2f}) on volatile flow.",
        "Volatility spikes as {agent} agents clash — price lurches to ${price:.2f}.",
        "Wild swings grip the session; ${price:.2f} prints as {buys} buys battle {sells} sells.",
        "Unstable tape — price gyrates to ${price:.2f} ({change:+.2f}) amid chaotic order flow.",
        "High-volatility regime: {agent} agents drive price to ${price:.2f} with conviction.",
    ],
}

_REGIME_TEMPLATES = [
    "Market character shifts from {old} to {new} at tick {tick} — repositioning underway.",
    "Tape transitions from {old} regime into {new} conditions at tick {tick}.",
    "Tick {tick}: structural break — {old} gives way to {new} market dynamics.",
    "Regime flip at tick {tick}: {old} fades as {new} takes hold across the book.",
]

_SHOCK_TEMPLATES = [
    "{shock_cap} shock at tick {tick}: price moves from ${before:.2f} to ${after:.2f} ({pct:+.1f}%).",
    "Tape shock — {shock} event detonates at tick {tick}, repricing market from ${before:.2f} to ${after:.2f}.",
    "Market dislocates on {shock} at tick {tick}: ${before:.2f} → ${after:.2f} in one move.",
    "{shock_cap} triggers cascading flow at tick {tick} — price lands at ${after:.2f}.",
]


def _rule_text(td: TickData) -> str:
    """Generate a rule-based commentary sentence from tick data."""
    ctx = {
        "tick":       td.tick,
        "price":      td.price,
        "change":     td.price_change,
        "buys":       td.buy_volume,
        "sells":      td.sell_volume,
        "agent":      td.dominant_agent_type.replace("_", " "),
        "agent_cap":  td.dominant_agent_type.replace("_", " ").capitalize(),
        "shock":      td.shock_regime.replace("_", " "),
        "shock_cap":  td.shock_regime.replace("_", " ").capitalize(),
        "momentum":   "accelerating" if abs(td.price_change) > 1.0 else "steady",
    }

    if td.shock_active:
        return random.choice(_TEMPLATES["shock"]).format(**ctx)

    imbalance = td.buy_volume - td.sell_volume
    if abs(td.price_change) > 2.0:
        key = "volatile"
    elif imbalance > 20:
        key = "buy_heavy"
    elif imbalance < -20:
        key = "sell_heavy"
    else:
        key = "balanced"

    return random.choice(_TEMPLATES[key]).format(**ctx)


# ---------------------------------------------------------------------------
# Feature extraction  (8 features per tick)
# ---------------------------------------------------------------------------

_N_FEATURES = 8


def _extract_features(td: TickData, price_history: list[float]) -> np.ndarray:
    """Convert a TickData into a fixed-length feature vector."""
    # price normalisation relative to recent window
    ref_price   = np.mean(price_history[-20:]) if price_history else td.price
    price_norm  = (td.price - ref_price) / (ref_price + 1e-8)
    change_norm = td.price_change / (ref_price + 1e-8)

    total_vol   = max(td.buy_volume + td.sell_volume, 1)
    buy_ratio   = td.buy_volume  / total_vol
    sell_ratio  = td.sell_volume / total_vol
    imbalance   = (td.buy_volume - td.sell_volume) / total_vol

    shock_flag  = 1.0 if td.shock_active else 0.0
    volatile    = 1.0 if abs(td.price_change) > 2.0 else 0.0

    return np.array([
        price_norm,
        change_norm,
        buy_ratio,
        sell_ratio,
        imbalance,
        td.hold_count / max(total_vol + td.hold_count, 1),
        shock_flag,
        volatile,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    PAD   = "<PAD>"
    UNK   = "<UNK>"
    BOS   = "<BOS>"
    EOS   = "<EOS>"

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._add(self.PAD)
        self._add(self.UNK)
        self._add(self.BOS)
        self._add(self.EOS)

    def _add(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
        return self.word2idx[word]

    def build_from_sentences(self, sentences: list[str]) -> None:
        for s in sentences:
            for w in s.lower().split():
                self._add(w)

    def encode(self, sentence: str) -> list[int]:
        bos = self.word2idx[self.BOS]
        eos = self.word2idx[self.EOS]
        unk = self.word2idx[self.UNK]
        tokens = [bos] + [self.word2idx.get(w, unk) for w in sentence.lower().split()] + [eos]
        return tokens

    def decode(self, indices: list[int]) -> str:
        skip = {self.word2idx[self.PAD], self.word2idx[self.BOS], self.word2idx[self.EOS]}
        words = [self.idx2word.get(i, self.UNK) for i in indices if i not in skip]
        text  = " ".join(words)
        if text and not text.endswith("."):
            text += "."
        return text.capitalize()

    def __len__(self) -> int:
        return len(self.word2idx)


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------

class NarratorLSTM(nn.Module):
    """
    Encoder-decoder LSTM.
    Encoder: processes tick feature sequence  → context vector
    Decoder: generates text word by word from context
    """

    def __init__(self, vocab_size: int, n_features: int = _N_FEATURES):
        super().__init__()
        # encoder
        self.encoder = nn.LSTM(
            input_size  = n_features,
            hidden_size = _HIDDEN_SIZE,
            num_layers  = _NUM_LAYERS,
            batch_first = True,
            dropout     = _DROPOUT if _NUM_LAYERS > 1 else 0.0,
        )
        # decoder
        self.embedding = nn.Embedding(vocab_size, _EMBED_DIM, padding_idx=0)
        self.decoder   = nn.LSTM(
            input_size  = _EMBED_DIM,
            hidden_size = _HIDDEN_SIZE,
            num_layers  = _NUM_LAYERS,
            batch_first = True,
            dropout     = _DROPOUT if _NUM_LAYERS > 1 else 0.0,
        )
        self.fc      = nn.Linear(_HIDDEN_SIZE, vocab_size)
        self.dropout = nn.Dropout(_DROPOUT)

    def encode(self, features: torch.Tensor):
        """features: (batch, seq_len, n_features) → hidden state"""
        _, (h, c) = self.encoder(features)
        return h, c

    def forward(
        self,
        features:    torch.Tensor,   # (B, seq_len, n_features)
        target_seq:  torch.Tensor,   # (B, text_len)
    ) -> torch.Tensor:               # (B, text_len, vocab_size)
        h, c     = self.encode(features)
        embeds   = self.dropout(self.embedding(target_seq))
        out, _   = self.decoder(embeds, (h, c))
        logits   = self.fc(self.dropout(out))
        return logits

    def generate(
        self,
        features:  torch.Tensor,   # (1, seq_len, n_features)
        vocab:     Vocabulary,
        max_len:   int         = _MAX_TEXT_LEN,
        temperature: float     = _TEMPERATURE,
    ) -> str:
        self.eval()
        with torch.no_grad():
            h, c    = self.encode(features)
            token   = torch.tensor([[vocab.word2idx[Vocabulary.BOS]]])
            words   = []
            for _ in range(max_len):
                embed   = self.embedding(token)
                out, (h, c) = self.decoder(embed, (h, c))
                logits  = self.fc(out.squeeze(1)) / max(temperature, 1e-6)
                probs   = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                if next_id == vocab.word2idx[Vocabulary.EOS]:
                    break
                words.append(next_id)
                token = torch.tensor([[next_id]])
        return vocab.decode(words)


# ---------------------------------------------------------------------------
# Training data buffer
# ---------------------------------------------------------------------------

class _DataBuffer:
    """Stores (feature_sequence, text) pairs collected during live simulation."""

    def __init__(self):
        self.feature_seqs: list[np.ndarray] = []   # each: (seq_len, n_features)
        self.texts:        list[str]         = []
        self._price_history: list[float]     = []
        self._tick_features: deque           = deque(maxlen=_SEQ_LEN)

    def add(self, td: TickData, text: str) -> None:
        self._price_history.append(td.price)
        feats = _extract_features(td, self._price_history)
        self._tick_features.append(feats)

        # only store once we have a full sequence
        if len(self._tick_features) == _SEQ_LEN:
            seq = np.stack(list(self._tick_features), axis=0)   # (seq_len, n_features)
            self.feature_seqs.append(seq)
            self.texts.append(text)

    def __len__(self) -> int:
        return len(self.texts)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"seqs": self.feature_seqs, "texts": self.texts}, f)

    def load(self, path: Path) -> None:
        if path.exists():
            with open(path, "rb") as f:
                d = pickle.load(f)
            self.feature_seqs = d["seqs"]
            self.texts        = d["texts"]
            logger.info("DataBuffer loaded %d samples from %s", len(self), path)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class _Trainer:
    def __init__(self, vocab: Vocabulary, device: torch.device):
        self.vocab  = vocab
        self.device = device

    def train(self, model: NarratorLSTM, buffer: _DataBuffer) -> float:
        """Train model on buffered data. Returns final loss."""
        if len(buffer) < _BATCH_SIZE:
            return float("inf")

        # build vocab from collected texts
        self.vocab.build_from_sentences(buffer.texts)

        # build tensors
        X_list, Y_list = [], []
        max_tlen = max(len(self.vocab.encode(t)) for t in buffer.texts)

        for seq, text in zip(buffer.feature_seqs, buffer.texts):
            tokens  = self.vocab.encode(text)
            # pad
            pad_id  = self.vocab.word2idx[Vocabulary.PAD]
            padded  = tokens + [pad_id] * (max_tlen - len(tokens))
            X_list.append(seq)
            Y_list.append(padded)

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        Y = torch.tensor(np.array(Y_list), dtype=torch.long)

        dataset    = TensorDataset(X, Y)
        loader     = DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=True)
        optimiser  = torch.optim.Adam(model.parameters(), lr=_LR)
        criterion  = nn.CrossEntropyLoss(ignore_index=self.vocab.word2idx[Vocabulary.PAD])

        model.train()
        model.to(self.device)
        last_loss = float("inf")

        for epoch in range(_EPOCHS):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # decoder input = all tokens except last
                # decoder target = all tokens except first
                dec_input  = y_batch[:, :-1]
                dec_target = y_batch[:, 1:]

                optimiser.zero_grad()
                logits = model(x_batch, dec_input)
                loss   = criterion(
                    logits.reshape(-1, len(self.vocab)),
                    dec_target.reshape(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item()

            last_loss = total_loss / max(len(loader), 1)

        logger.info("LSTM training complete — loss=%.4f, vocab=%d, samples=%d",
                    last_loss, len(self.vocab), len(buffer))
        return last_loss


# ---------------------------------------------------------------------------
# Main Narrator class
# ---------------------------------------------------------------------------

class Narrator:
    def __init__(
        self,
        interval:    int  = _DEFAULT_INTERVAL,
        history_max: int  = _HISTORY_MAX,
        enabled:     bool = True,
        # legacy args — accepted but ignored (no API used)
        model:     str           = "",
        max_tokens: int          = 0,
        api_key:   Optional[str] = None,
    ):
        self._interval   = interval
        self._enabled    = enabled
        self._device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._history:   deque[NarratorEntry] = deque(maxlen=history_max)
        self._last_regime:      str = "calm"
        self._last_regime_tick: int = 0
        self._tick_counter:     int = 0

        # training state
        self._vocab   = Vocabulary()
        self._buffer  = _DataBuffer()
        self._model:  Optional[NarratorLSTM] = None
        self._trained = False
        self._ticks_since_retrain = 0

        # try to load existing model from previous run
        self._try_load()

        if self._enabled:
            status = "LSTM ready" if self._trained else f"warmup (need {_MIN_TRAIN_TICKS} ticks)"
            print(f"[Narrator] OK: local LSTM narrator — {status} (device={self._device})")
            logger.info("Narrator initialised — trained=%s device=%s", self._trained, self._device)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _try_load(self) -> None:
        try:
            if _VOCAB_PATH.exists():
                with open(_VOCAB_PATH, "rb") as f:
                    self._vocab = pickle.load(f)

            if _DATA_PATH.exists():
                self._buffer.load(_DATA_PATH)

            if _MODEL_PATH.exists() and len(self._vocab) > 4:
                self._model = NarratorLSTM(len(self._vocab)).to(self._device)
                self._model.load_state_dict(
                    torch.load(_MODEL_PATH, map_location=self._device)
                )
                self._model.eval()
                self._trained = True
                logger.info("Narrator: loaded saved LSTM model (%d vocab)", len(self._vocab))
        except Exception as e:
            logger.warning("Narrator: could not load saved model: %s", e)
            self._trained = False

    def _save(self) -> None:
        try:
            if self._model:
                torch.save(self._model.state_dict(), _MODEL_PATH)
            with open(_VOCAB_PATH, "wb") as f:
                pickle.dump(self._vocab, f)
            self._buffer.save(_DATA_PATH)
        except Exception as e:
            logger.warning("Narrator: save failed: %s", e)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def _maybe_train(self) -> None:
        """Train if enough data collected and not yet trained, or retrain interval hit."""
        n = len(self._buffer)

        should_train = (
            (not self._trained and n >= _MIN_TRAIN_TICKS) or
            (self._trained and self._ticks_since_retrain >= _RETRAIN_EVERY)
        )

        if not should_train:
            return

        print(f"[Narrator] Training LSTM on {n} samples... ", end="", flush=True)

        # rebuild model with current vocab size
        trainer = _Trainer(self._vocab, self._device)
        # pre-build vocab
        self._vocab.build_from_sentences(self._buffer.texts)

        self._model = NarratorLSTM(len(self._vocab)).to(self._device)
        loss = trainer.train(self._model, self._buffer)

        self._trained             = True
        self._ticks_since_retrain = 0

        print(f"done (loss={loss:.4f}, vocab={len(self._vocab)})")
        self._save()

    # -----------------------------------------------------------------------
    # Text generation
    # -----------------------------------------------------------------------

    def _lstm_generate(self, td: TickData) -> Optional[str]:
        """Generate text using trained LSTM."""
        if not self._trained or self._model is None:
            return None
        try:
            # build feature sequence from buffer's recent ticks
            feats = list(self._buffer._tick_features)
            if len(feats) < _SEQ_LEN:
                return None
            seq   = np.stack(feats[-_SEQ_LEN:], axis=0)
            x     = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(self._device)
            text  = self._model.generate(x, self._vocab)
            if len(text.split()) < 4:   # too short — fallback
                return None
            return text
        except Exception as e:
            logger.warning("LSTM generate failed: %s", e)
            return None

    def _generate_text(self, td: TickData, trigger: str) -> str:
        """Get text from LSTM if trained, otherwise rule-based fallback."""
        rule_text = _rule_text(td)

        # always collect data for training
        self._buffer.add(td, rule_text)
        self._ticks_since_retrain += 1
        self._maybe_train()

        # use LSTM if ready
        lstm_text = self._lstm_generate(td)
        return lstm_text if lstm_text else rule_text

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _push(self, tick: int, text: str, trigger: str) -> NarratorEntry:
        entry = NarratorEntry(tick=tick, text=text, trigger=trigger)
        self._history.append(entry)
        logger.info("Narrator [%s] tick %d: %s", trigger, tick, text)
        return entry

    def _was_shock_last_tick(self) -> bool:
        return bool(self._history) and self._history[-1].trigger == "shock"

    # -----------------------------------------------------------------------
    # Public API  (same interface as before — drop-in replacement)
    # -----------------------------------------------------------------------

    def step(self, tick_data: TickData) -> Optional[str]:
        """Call every tick. Returns commentary string or None."""
        if not self._enabled:
            return None
        self._tick_counter += 1

        # shock fires immediately regardless of interval
        if tick_data.shock_active and not self._was_shock_last_tick():
            text = self._generate_text(tick_data, "shock")
            self._push(tick_data.tick, text, "shock")
            return text

        # normal interval
        if self._interval > 0 and self._tick_counter % self._interval != 0:
            # still collect data even on skipped ticks
            rule = _rule_text(tick_data)
            self._buffer.add(tick_data, rule)
            self._ticks_since_retrain += 1
            self._maybe_train()
            return None

        text = self._generate_text(tick_data, "auto")
        self._push(tick_data.tick, text, "auto")
        return text

    def on_shock(self, shock_data: dict) -> str:
        """Call when a shock event fires."""
        tick    = shock_data.get("tick", 0)
        s_type  = shock_data.get("shock_type", "shock").replace("_", " ")
        before  = shock_data.get("price_before", 0.0)
        after   = shock_data.get("price_after",  0.0)
        pct     = ((after - before) / max(before, 1e-8)) * 100
        text    = random.choice(_SHOCK_TEMPLATES).format(
            tick      = tick,
            shock     = s_type,
            shock_cap = s_type.capitalize(),
            before    = before,
            after     = after,
            pct       = pct,
        )
        self._push(tick, text, "shock")
        return text

    def on_regime_change(self, old_regime: str, new_regime: str, tick: int) -> str:
        """Call when market regime changes."""
        if tick - self._last_regime_tick < _REGIME_COOLDOWN:
            return ""
        text = random.choice(_REGIME_TEMPLATES).format(
            old=old_regime.replace("_", " "),
            new=new_regime.replace("_", " "),
            tick=tick,
        )
        self._push(tick, text, "regime_change")
        self._last_regime      = new_regime
        self._last_regime_tick = tick
        return text

    def manual_comment(self, tick: int, context: str) -> str:
        """One-off commentary with custom context."""
        text = f"Tick {tick}: {context}."
        self._push(tick, text, "manual")
        return text

    def market_summary(
        self,
        tick:                int,
        price:               float,
        price_change_pct:    float,
        volatility:          float,
        spread_bps:          float,
        bid_depth:           float,
        ask_depth:           float,
        imbalance:           float,
        shock_active:        bool,
        shock_regime:        str,
        vol_multiplier:      float,
        gan_regime:          Optional[str],
        active_panic_agents: int,
        total_agents:        int = 100,
    ) -> str:
        panic_pct  = (active_panic_agents / max(total_agents, 1)) * 100
        shock_note = f" {shock_regime.replace('_',' ')} shock active." if shock_active else ""
        mood       = "fearful" if panic_pct > 20 else ("cautious" if panic_pct > 5 else "composed")
        flow       = "bid-heavy" if imbalance > 0.1 else ("offer-heavy" if imbalance < -0.1 else "balanced")
        text = (
            f"Tick {tick}: price ${price:.2f} ({price_change_pct:+.2f}%), "
            f"vol {volatility:.4f}, {flow} flow, {mood} sentiment, "
            f"{panic_pct:.0f}% panic agents.{shock_note}"
        )
        self._push(tick, text, "summary")
        return text

    # -----------------------------------------------------------------------
    # Properties / utilities
    # -----------------------------------------------------------------------

    @property
    def history(self) -> list[NarratorEntry]:
        return list(self._history)

    @property
    def latest(self) -> str:
        return self._history[-1].text if self._history else ""

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def training_samples(self) -> int:
        return len(self._buffer)

    def recent(self, n: int = 10) -> list[NarratorEntry]:
        return list(self._history)[-n:]

    def clear(self) -> None:
        self._history.clear()
        self._tick_counter = 0
        logger.info("Narrator history cleared.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_narrator(
    interval:    int  = _DEFAULT_INTERVAL,
    enabled:     bool = True,
    history_max: int  = _HISTORY_MAX,
) -> Narrator:
    return Narrator(
        interval    = interval,
        history_max = history_max,
        enabled     = enabled,
    )