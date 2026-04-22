"""
gan/__init__.py
===============
Public API for the GAN Data Engine (Layer 1).

Import pattern
--------------
    from gan import build_generator, build_discriminator, build_trainer
    from gan import build_cond_generator, build_cond_discriminator
    from gan import build_dataset, generate_pipeline
    from gan import encode_regime, sample_regime, REGIMES

Architecture overview
---------------------
    data_loader.py      → OHLCV fetch, clean, normalise, sequence
    generator.py        → LSTM Generator (noise → OHLCV)
    discriminator.py    → Dual-path Discriminator (LSTM + CNN)
    trainer.py          → Adversarial training loop
    conditional_gan.py  → Regime-conditioned CondGenerator + CondDiscriminator
    generate.py         → Inference, denormalisation, export
"""

from gan.data_loader import (
    fetch_ohlcv,
    clean,
    normalise,
    to_sequences,
    save,
    load,
    build_dataset,
)

from gan.generator import (
    Generator,
    build_generator,
)

from gan.discriminator import (
    Discriminator,
    build_discriminator,
)

from gan.trainer import (
    Trainer,
    EpochMetrics,
    build_trainer,
)

from gan.conditional_gan import (
    CondGenerator,
    CondDiscriminator,
    build_cond_generator,
    build_cond_discriminator,
    encode_regime,
    encode_regime_batch,
    sample_regime,
    REGIMES,
    NUM_REGIMES,
)

from gan.generate import (
    load_generator,
    generate,
    denormalise,
    to_dataframe,
    save_synthetic,
    load_scaler,
    generate_pipeline,
)

__all__ = [
    # Data
    "fetch_ohlcv", "clean", "normalise", "to_sequences", "save", "load", "build_dataset",
    # Models
    "Generator", "build_generator",
    "Discriminator", "build_discriminator",
    # Training
    "Trainer", "EpochMetrics", "build_trainer",
    # Conditional GAN
    "CondGenerator", "CondDiscriminator",
    "build_cond_generator", "build_cond_discriminator",
    "encode_regime", "encode_regime_batch", "sample_regime",
    "REGIMES", "NUM_REGIMES",
    # Inference
    "load_generator", "generate", "denormalise",
    "to_dataframe", "save_synthetic", "load_scaler", "generate_pipeline",
]