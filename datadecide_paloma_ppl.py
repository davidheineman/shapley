from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from paloma_dataloader import get_text_column_name, load_paloma_c4

DEFAULT_MODELS_FILE = Path(__file__).resolve().parent / "datadecide_models.txt"


def parse_models_file(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Parse datadecide_models.txt; return (variants_with_both, 90m_ids, 1b_ids)."""
    text = path.read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    model_90m: list[str] = []
    model_1b: list[str] = []
    for line in lines:
        if line.endswith("-90M"):
            model_90m.append(line)
        elif line.endswith("-1B"):
            model_1b.append(line)

    def variant_from_id(model_id: str, suffix: str) -> str:
        return model_id[: -len(suffix)].rstrip("-")

    variants_90m = {variant_from_id(m, "-90M"): m for m in model_90m}
    variants_1b = {variant_from_id(m, "-1B"): m for m in model_1b}
    common = sorted(set(variants_90m) & set(variants_1b))
    ids_90m = [variants_90m[v] for v in common]
    ids_1b = [variants_1b[v] for v in common]
    return common, ids_90m, ids_1b


def compute_ppl_corpus(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 1024,
    stride: int | None = None,
    device: str | None = None,
) -> float:
    """Corpus-level perplexity: mean of per-token CE loss over all tokens, then exp."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    stride = stride or max_length

    with torch.no_grad():
        for text in texts:
            if not (text and str(text).strip()):
                continue
            enc = tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            # Labels: shift by 1; -100 for non-predicted
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:].clone()
            labels[:, -1] = -100
            if attention_mask is not None:
                labels = torch.where(
                    attention_mask.bool(), labels, torch.full_like(labels, -100)
                )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            n = (labels != -100).sum().item()
            if n > 0:
                total_loss += outputs.loss.item() * n
                total_tokens += n

    if total_tokens == 0:
        return float("nan")
    mean_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(mean_loss)).item())


def load_model_and_tokenizer(model_id: str, device_map: str = "auto", dtype=torch.bfloat16):
    """Load HuggingFace causal LM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model, tokenizer


def ranking_agreement(ppl_90m: list[float], ppl_1b: list[float]) -> float:
    """Fraction of pairs (i,j) where (ppl_90m[i] < ppl_90m[j]) == (ppl_1b[i] < ppl_1b[j])."""
    n = len(ppl_90m)
    assert n == len(ppl_1b)
    agreed = 0
    total = 0
    for i, j in itertools.combinations(range(n), 2):
        order_90 = ppl_90m[i] < ppl_90m[j]
        order_1b = ppl_1b[i] < ppl_1b[j]
        if order_90 == order_1b:
            agreed += 1
        total += 1
    return (agreed / total * 100.0) if total else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="DataDecide 90M/1B PPL on Paloma C4 and ranking agreement"
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=DEFAULT_MODELS_FILE,
        help="Path to datadecide_models.txt",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max Paloma C4 samples per run (default 500)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max tokens per document for PPL (default 512)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Paloma split to use: val or test (default val)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse models and load Paloma; do not load any LM",
    )
    args = parser.parse_args()

    variants, ids_90m, ids_1b = parse_models_file(args.models_file)
    print(f"Found {len(variants)} variants with both 90M and 1B models.")

    print("Loading Paloma C4...")
    ds = load_paloma_c4(split=args.split, max_samples=args.max_samples)
    text_key = get_text_column_name(ds)
    texts = [ds[i][text_key] for i in range(len(ds))]
    print(f"Using {len(texts)} documents (key={text_key}), max_length={args.max_length}.")

    if args.dry_run:
        print("Dry run: skipping model loading and PPL.")
        return

    results_90m: list[float] = []
    results_1b: list[float] = []

    for scale, model_ids, results in [("90M", ids_90m, results_90m), ("1B", ids_1b, results_1b)]:
        for idx, model_id in enumerate(model_ids):
            print(f"  [{scale}] {idx+1}/{len(model_ids)} {model_id}")
            try:
                model, tokenizer = load_model_and_tokenizer(model_id)
                ppl = compute_ppl_corpus(
                    model, tokenizer, texts, max_length=args.max_length
                )
                results.append(ppl)
                print(f"    PPL = {ppl:.4f}")
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    Error: {e}")
                results.append(float("nan"))

    # Build variant -> PPL for display; use only rows where both are valid
    def _valid_ppl(x: float) -> bool:
        return x == x and x != float("inf")

    valid = [
        i
        for i in range(len(variants))
        if _valid_ppl(results_90m[i]) and _valid_ppl(results_1b[i])
    ]
    if not valid:
        print("No valid PPL results; cannot compute ranking agreement.")
        return

    ppl_90m_valid = [results_90m[i] for i in valid]
    ppl_1b_valid = [results_1b[i] for i in valid]
    variants_valid = [variants[i] for i in valid]

    pct = ranking_agreement(ppl_90m_valid, ppl_1b_valid)
    print("\n--- Results ---")
    print("Variant (short)\tPPL 90M\tPPL 1B")
    for v, p90, p1 in zip(variants_valid, ppl_90m_valid, ppl_1b_valid):
        short = v.replace("allenai/DataDecide-", "") if v.startswith("allenai/") else v
        print(f"{short}\t{p90:.4f}\t{p1:.4f}")
    print(f"\nRanking agreement (90M vs 1B): {pct:.2f}%")
    print(f"(Fraction of model pairs with same relative order by PPL)")


if __name__ == "__main__":
    main()
