Expert-level performance estimate for MoEs using Shapley values

```bash
uv sync # FlexOlmo NEEDS torch 2.6.0+cu124
```

```bash
python flexolmo_ppl.py --dataset paloma --paloma-max-samples 100 --shapley-mc 10
```

```bash
python datadecide_paloma_ppl.py --max-samples 10
```