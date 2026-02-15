"""Paloma dataloader: load allenai/paloma (e.g. C4 English) for PPL or Shapley runs."""

from __future__ import annotations

from datasets import load_dataset

PALOMA_C4_CONFIG = "c4_en"  # Paloma subset: C4 English


def load_paloma_c4(
    split: str = "val",
    max_samples: int | None = None,
    config: str | None = None,
):
    """Load Paloma C4 English subset. Requires HF login if gated (allenai/paloma).

    Returns:
        Dataset (single split) with at least a text column (e.g. "text" or "content").
    """
    configs = (config,) if config is not None else (PALOMA_C4_CONFIG, "c4-en", "c4_en")
    for cfg in configs:
        try:
            ds = load_dataset("allenai/paloma", cfg)
            if isinstance(ds, dict):
                split_key = split if split in ds else "val" if "val" in ds else "test"
                ds = ds[split_key]
            break
        except Exception:
            continue
    else:
        raise RuntimeError(
            "Could not load allenai/paloma with config c4_en or c4-en. "
            "Ensure you are logged in: huggingface-cli login"
        )
    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def get_text_column_name(ds) -> str:
    """Return the column name used for document text ('text' or 'content')."""
    if "text" in ds.column_names:
        return "text"
    if "content" in ds.column_names:
        return "content"
    raise KeyError("Dataset has no 'text' or 'content' column")


def get_paloma_c4_texts(
    split: str = "val",
    max_samples: int | None = None,
    config: str | None = None,
) -> list[str]:
    """Load Paloma C4 and return a list of document strings."""
    ds = load_paloma_c4(split=split, max_samples=max_samples, config=config)
    text_key = get_text_column_name(ds)
    return [ds[i][text_key] for i in range(len(ds))]
