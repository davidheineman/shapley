from __future__ import annotations

import itertools
import math
import random
import threading
from contextlib import contextmanager

import torch

from transformers import PreTrainedModel, PreTrainedTokenizer

# Thread-local storage for disabled expert indices (set of int) during forward.
_disabled_experts_tls: threading.local = threading.local()

FLEXOLMO_MODEL_ID = "allenai/FlexOlmo-7x7B-1T"
FLEXOLMO_TOKENIZER_ID = "allenai/dolma2-tokenizer"


def _get_disabled_experts():
    return getattr(_disabled_experts_tls, "value", None)


def _set_disabled_experts(value: set[int] | None):
    _disabled_experts_tls.value = value


# Original FlexOlmoTopKRouter.forward; we patch it to mask disabled experts.
def _router_forward_with_mask(router, hidden_states):
    import torch.nn.functional as F

    hidden_states = hidden_states.reshape(-1, router.hidden_dim)
    router_logits = F.linear(hidden_states, router.weight)
    disabled = _get_disabled_experts()
    if disabled:
        router_logits = router_logits.clone()
        for idx in disabled:
            if 0 <= idx < router.num_experts:
                router_logits[:, idx] = float("-inf")
    router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_logits, router.top_k, dim=-1)
    if router.norm_topk_prob:
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    router_scores = router_top_value
    return router_logits, router_scores, router_indices


def _install_router_patch(model):
    """Patch all FlexOlmoTopKRouter instances in the model to respect _disabled_experts_tls."""
    from transformers.models.flex_olmo.modeling_flex_olmo import FlexOlmoTopKRouter

    # Patch the class so any router uses our forward
    if not getattr(FlexOlmoTopKRouter, "_ppl_patch_applied", False):
        def _patched_forward(self, hidden_states):
            return _router_forward_with_mask(self, hidden_states)
        FlexOlmoTopKRouter.forward = _patched_forward
        FlexOlmoTopKRouter._ppl_patch_applied = True

    # Also patch each instance so we definitely override any per-instance binding
    for _name, module in model.named_modules():
        if type(module).__name__ == "FlexOlmoTopKRouter":
            module.forward = lambda hidden_states, router=module: _router_forward_with_mask(router, hidden_states)


def load_flexolmo(
    model_id: str = FLEXOLMO_MODEL_ID,
    tokenizer_id: str | None = FLEXOLMO_TOKENIZER_ID,
    device: str | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    device_map: str | None = "auto",
    attn_implementation: str | None = "sdpa",
    **from_pretrained_kwargs,
) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    """Load FlexOlmo-7x7B-1T and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_id = tokenizer_id or model_id  # fallback to model repo if None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    load_kwargs = dict(device_map=device_map or device, attn_implementation=attn_implementation, **from_pretrained_kwargs)
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device_map is None and device != "cpu":
        model = model.to(device)
    _install_router_patch(model)
    return model, tokenizer


@contextmanager
def disabled_experts(disabled: set[int] | list[int] | None):
    """Context manager to disable given experts (by index 0..6) during forward."""
    prev = _get_disabled_experts()
    try:
        _set_disabled_experts(set(disabled) if disabled else None)
        yield
    finally:
        _set_disabled_experts(prev)


def compute_ppl(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    text: str,
    disabled_expert_indices: set[int] | list[int] | None = None,
    max_length: int | None = None,
    stride: int | None = None,
    return_loss: bool = False,
) -> float | tuple[float, float]:
    """Compute perplexity (PPL) over a string."""
    was_training = model.training
    model.eval()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    device = getattr(model, "device", None) or next(model.parameters(), torch.tensor(0)).device
    input_ids = enc["input_ids"].to(device).contiguous()
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device).contiguous()

    # Labels: shift by 1; -100 for positions we don't predict (e.g. padding)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:].clone()
    labels[:, -1] = -100
    if attention_mask is not None:
        labels = torch.where(attention_mask.bool(), labels, torch.full_like(labels, -100))

    with torch.no_grad():
        with disabled_experts(disabled_expert_indices):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
    loss = outputs.loss.item()
    if was_training:
        model.train()

    ppl = float(torch.exp(torch.tensor(loss)).item())
    return (ppl, loss) if return_loss else ppl


def compute_ppl_corpus(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    texts: list[str],
    disabled_expert_indices: set[int] | list[int] | None = None,
    max_length: int | None = None,
    return_loss: bool = False,
    batch_size: int = 1,
) -> float | tuple[float, float]:
    """Corpus-level PPL: mean per-token CE loss over all documents, then exp. Returns (ppl, mean_loss) if return_loss.
    batch_size > 1 pads and processes multiple docs per forward for speed."""
    device = getattr(model, "device", None) or next(model.parameters(), torch.tensor(0)).device
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    texts = [t for t in texts if t and str(t).strip()]
    if not texts:
        if was_training:
            model.train()
        ppl = mean_loss = float("nan")
        return (ppl, mean_loss) if return_loss else ppl

    with torch.no_grad():
        with disabled_experts(disabled_expert_indices):
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=max_length is not None,
                    max_length=max_length,
                    padding=True,
                    return_attention_mask=True,
                )
                input_ids = enc["input_ids"].to(device).contiguous()
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device).contiguous()
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:].clone()
                labels[:, -1] = -100
                if attention_mask is not None:
                    labels = torch.where(attention_mask.bool(), labels, torch.full_like(labels, -100))
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n = (labels != -100).sum().item()
                if n > 0:
                    total_loss += outputs.loss.item() * n
                    total_tokens += n

    if was_training:
        model.train()
    if total_tokens == 0:
        mean_loss = float("nan")
        ppl = float("nan")
    else:
        mean_loss = total_loss / total_tokens
        ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    return (ppl, mean_loss) if return_loss else ppl


# Convenience: expert name -> index for FlexOlmo-7x7B-1T
FLEXOLMO_EXPERT_NAMES = [
    "public",
    "math",
    "news",
    "code",
    "academic",
    "creative",
    "reddit",
]


def expert_names_to_indices(names: list[str]) -> list[int]:
    """Map expert names to indices (e.g. ['math','code'] -> [1, 3])."""
    name2idx = {n.lower(): i for i, n in enumerate(FLEXOLMO_EXPERT_NAMES)}
    return [name2idx[n.lower()] for n in names if n.lower() in name2idx]


def compute_shapley_values(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    texts: list[str],
    num_experts: int = 7,
    max_length: int | None = None,
    batch_size: int = 1,
) -> list[float]:
    """
    Compute Shapley value for each expert (0..num_experts-1).

    Value of coalition S = -mean loss over texts when only experts in S are enabled (i.e. disabled = N\\S).
    """
    n = num_experts
    N = set(range(n))

    def value_fn(disabled: set[int] | None) -> float:
        _, loss = compute_ppl_corpus(
            model,
            tokenizer,
            texts,
            disabled_expert_indices=disabled,
            max_length=max_length,
            return_loss=True,
            batch_size=batch_size,
        )
        return -loss

    # Precompute v(S) = -loss when disabled = N\S (so enabled = S)
    from tqdm import tqdm

    v_cache: dict[frozenset[int], float] = {}
    coalitions = []
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            S_set = frozenset(S)
            coalitions.append((S_set, (N - S_set) or None))
    empty_set = frozenset()
    for S_set, disabled in tqdm(coalitions, desc="Coalitions", unit="coal"):
        if S_set == empty_set:
            # v(∅)=0 by convention; all experts disabled would give NaN (softmax(-inf,...,-inf))
            v_cache[S_set] = 0.0
        else:
            v_cache[S_set] = value_fn(disabled)

    # Shapley value φ_i = sum over S ⊆ N\{i} of [ |S|!(n-1-|S|)!/n! * (v(S∪{i}) - v(S)) ]
    fact = math.factorial
    coef = [fact(s) * fact(n - 1 - s) / fact(n) for s in range(n)]

    phi = [0.0] * n
    for i in range(n):
        for r in range(n):  # |S| = r, S ⊆ N \ {i}
            for S in itertools.combinations([j for j in range(n) if j != i], r):
                S_set = frozenset(S)
                S_union_i = S_set | {i}
                marginal = v_cache[S_union_i] - v_cache[S_set]
                phi[i] += coef[r] * marginal

    return phi


def compute_shapley_values_mc(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    texts: list[str],
    n_perm: int = 20,
    num_experts: int = 7,
    max_length: int | None = None,
    batch_size: int = 1,
    seed: int | None = None,
) -> list[float]:
    """
    Monte Carlo estimate of Shapley values: sample n_perm random permutations,
    compute marginals along each, average. Uses n_perm * (n+1) coalition evals
    (e.g. 20 * 8 = 160 vs 128 for exact, but with caching we only eval each
    distinct S once per permutation so often fewer). For n_perm small (e.g. 10)
    total evals = 80, faster than exact 128.
    """
    if seed is not None:
        random.seed(seed)
    n = num_experts
    N = set(range(n))

    def value_fn(disabled: set[int] | None) -> float:
        _, loss = compute_ppl_corpus(
            model,
            tokenizer,
            texts,
            disabled_expert_indices=disabled,
            max_length=max_length,
            return_loss=True,
            batch_size=batch_size,
        )
        return -loss

    from tqdm import tqdm

    phi = [0.0] * n
    # One permutation = order in which experts are "added". Marginal for expert i = v(S ∪ {i}) - v(S).
    permutations = [
        random.sample(range(n), n) for _ in range(n_perm)
    ]
    v_cache: dict[frozenset[int], float] = {}

    for perm in tqdm(permutations, desc="MC permutations", unit="perm"):
        chain = []  # S_0 = ∅, S_1 = {perm[0]}, S_2 = {perm[0],perm[1]}, ...
        S = frozenset()
        for j in range(n + 1):
            chain.append(S)
            if j < n:
                S = S | {perm[j]}
        # Evaluate v for each set in the chain (use cache so repeated S across perms are free)
        vals = []
        empty_set = frozenset()
        for S_set in chain:
            if S_set not in v_cache:
                if S_set == empty_set:
                    # v(∅)=0 by convention; all experts disabled would give NaN (softmax(-inf,...,-inf))
                    v_cache[S_set] = 0.0
                else:
                    disabled = (N - S_set) or None
                    v_cache[S_set] = value_fn(disabled)
            vals.append(v_cache[S_set])
        # After first perm, sanity-check: v(N) should be non-zero if router patch works (v(∅) is fixed to 0)
        if len(v_cache) >= 2 and perm is permutations[0]:
            v_full = v_cache.get(frozenset(N))
            if v_full is not None and abs(v_full) < 1e-9:
                import warnings
                warnings.warn(
                    f"v(N) ≈ 0 ({v_full:.4f}). "
                    "Router masking may not be applied—Shapley values may be wrong. "
                    "Try: python flexolmo_ppl.py --text 'x' --disable-experts 0,1,2,3,4,5,6 vs no --disable-experts to check PPL change.",
                    UserWarning,
                    stacklevel=2,
                )
        for j in range(n):
            marginal = vals[j + 1] - vals[j]
            phi[perm[j]] += marginal
    for i in range(n):
        phi[i] /= n_perm
    return phi


if __name__ == "__main__":
    import argparse

    from paloma_dataloader import get_paloma_c4_texts

    parser = argparse.ArgumentParser(description="Compute PPL with FlexOlmo (optional expert disabling).")
    parser.add_argument("--text", type=str, default=None, help="Single input text (ignored if --dataset is set)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["paloma"],
        help="Use dataset instead of --text: paloma = Paloma C4 (val split)",
    )
    parser.add_argument("--paloma-split", type=str, default="val", help="Paloma split when --dataset paloma (default val)")
    parser.add_argument("--paloma-max-samples", type=int, default=None, help="Max Paloma samples (default all)")
    parser.add_argument("--disable-experts", type=str, default=None, help="Comma-separated expert indices or names, e.g. 1,3 or math,code")
    parser.add_argument("--float16", action="store_true", help="Load model in float16")
    parser.add_argument("--max-length", type=int, default=None, help="Max token length for PPL/Shapley")
    parser.add_argument("--shapley", action="store_true", help="Compute exact Shapley values (128 coalitions)")
    parser.add_argument(
        "--shapley-mc",
        type=int,
        default=None,
        metavar="N",
        help="Faster: Monte Carlo Shapley with N permutations (e.g. 10 → ~80 evals vs 128 exact).",
    )
    parser.add_argument(
        "--shapley-max-docs",
        type=int,
        default=None,
        metavar="N",
        help="Use only N random docs for Shapley (faster; default: use all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="B",
        help="Batch size for corpus PPL / Shapley (default 1 to avoid OOM; try 4 or 8 if you have GPU memory).",
    )
    args = parser.parse_args()

    if args.dataset == "paloma":
        print("Loading Paloma C4...")
        texts = get_paloma_c4_texts(split=args.paloma_split, max_samples=args.paloma_max_samples)
        print(f"Using {len(texts)} documents.")
    else:
        text = args.text or "The quick brown fox jumps over the lazy dog."
        texts = [text]

    if (args.shapley or args.shapley_mc is not None) and args.shapley_max_docs is not None and len(texts) > args.shapley_max_docs:
        texts = random.sample(texts, args.shapley_max_docs)
        print(f"Subsampled to {len(texts)} documents for Shapley.")

    print("Loading model and tokenizer...")
    model, tokenizer = load_flexolmo(
        dtype=torch.float16 if args.float16 else torch.bfloat16,
        attn_implementation="sdpa",
    )

    if args.shapley or args.shapley_mc is not None:
        n_perm = args.shapley_mc
        if n_perm is not None:
            print(f"Computing Monte Carlo Shapley (~{n_perm * 8} evals, {n_perm} permutations, batch_size={args.batch_size})...")
            phi = compute_shapley_values_mc(
                model,
                tokenizer,
                texts=texts,
                n_perm=n_perm,
                num_experts=len(FLEXOLMO_EXPERT_NAMES),
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
        else:
            print(f"Computing exact Shapley values (128 coalitions, batch_size={args.batch_size})...")
            phi = compute_shapley_values(
                model,
                tokenizer,
                texts=texts,
                num_experts=len(FLEXOLMO_EXPERT_NAMES),
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
        print("Shapley value (contribution to -loss) per expert:")
        for i, name in enumerate(FLEXOLMO_EXPERT_NAMES):
            print(f"  {name}: {phi[i]:.6f}")
        print(f"  sum: {sum(phi):.6f}")
    else:
        disabled = None
        if args.disable_experts:
            parts = [p.strip() for p in args.disable_experts.split(",")]
            if all(p.isdigit() for p in parts):
                disabled = [int(p) for p in parts]
            else:
                disabled = expert_names_to_indices(parts)
            print(f"Disabled experts: {disabled}")
        if len(texts) == 1:
            ppl, loss = compute_ppl(
                model, tokenizer, texts[0],
                disabled_expert_indices=disabled,
                max_length=args.max_length,
                return_loss=True,
            )
        else:
            ppl, loss = compute_ppl_corpus(
                model, tokenizer, texts,
                disabled_expert_indices=disabled,
                max_length=args.max_length,
                return_loss=True,
                batch_size=args.batch_size,
            )
        print(f"PPL: {ppl:.4f}  loss: {loss:.4f}")
