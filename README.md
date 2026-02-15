Expert-level performance estimate for MoEs using Shapley values

```bash
uv sync # FlexOlmo NEEDS torch 2.6.0+cu124
```

```bash
python flexolmo_ppl.py --dataset paloma --paloma-max-samples 50 --shapley-mc 10 --batch-size 4 # ~5 min on 2 H100s
```

<details>
<summary>Example output</summary>

```bash
(shapley) root@ceres-cs-aus-445 ~/ai2/shapley$ python flexolmo_ppl.py --dataset paloma --paloma-max-samples 50 --shapley-mc 10 --batch-size 4
Loading Paloma C4...
Using 50 documents.
Loading model and tokenizer...
Loading weights: 100%|██████████████████████████████████████████| 355/355 [00:10<00:00, 32.48it/s, Materializing param=model.norm.weight]
Computing Monte Carlo Shapley (~80 evals, 10 permutations, batch_size=4)...
MC permutations: 100%|█████████████████████████████████████████████████████████████████████████████████| 10/10 [03:15<00:00, 19.59s/perm]
Shapley value (contribution to -loss) per expert:
  public: -0.254924
  math: -1.224823
  news: -0.111676
  code: -1.265758
  academic: -3.043945
  creative: -1.027924
  reddit: -2.777939
  sum: -9.706990
```

</details>



```bash
python datadecide_paloma_ppl.py --max-samples 10
```