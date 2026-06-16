import json, statistics, os

folds = [
    ('locoa', 'nottheonion+pareidolia'),
    ('locob', 'upliftingnews+fakehistoryporn'),
    ('lococ', 'usnews+usanews+fakealbumcovers'),
]

def acc(path, *keys):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    for k in keys:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    if isinstance(d, float): return d
    if isinstance(d, dict): return d.get('accuracy')
    return None

models = {
    'CausalHeteroGNN': lambda f: acc(f'results/metrics_{f}_s42.json',          'causal','unseen','accuracy'),
    'Baseline GNN':    lambda f: acc(f'results/baselines_erm_mlp_{f}_s42.json', 'erm','unseen','accuracy'),
    'MLP':             lambda f: acc(f'results/baselines_erm_mlp_{f}_s42.json', 'mlp','unseen','accuracy'),
    'IRM':             lambda f: acc(f'results/baselines_irm_eerm_{f}_s42.json','irm','unseen','accuracy'),
    'EERM':            lambda f: acc(f'results/baselines_irm_eerm_{f}_s42.json','eerm','unseen','accuracy'),
}

print("=== SEED=42 ONLY ===")
print(f"{'Model':<20} {'loco_a':>8} {'loco_b':>8} {'loco_c':>8} {'Mean':>8}  Rank")
print('-' * 65)

rows = []
for mname, loader in models.items():
    vals = [loader(f) for f, _ in folds]
    mean = statistics.mean(v for v in vals if v is not None)
    rows.append((mname, vals, mean))

rows.sort(key=lambda x: -x[2])
for rank, (mname, vals, mean) in enumerate(rows, 1):
    vstr = '  '.join(f'{v*100:6.2f}' if v else '   N/A' for v in vals)
    print(f'{mname:<20} {vstr}  {mean*100:6.2f}   #{rank}')

print("\n=== 3 SEEDS (mean) ===")
print(f"{'Model':<20} {'loco_a':>8} {'loco_b':>8} {'loco_c':>8} {'Mean':>8}  Rank")
print('-' * 65)

models3 = {
    'CausalHeteroGNN': lambda f,s: acc(f'results/metrics_{f}_s{s}.json',          'causal','unseen','accuracy'),
    'Baseline GNN':    lambda f,s: acc(f'results/baselines_erm_mlp_{f}_s{s}.json', 'erm','unseen','accuracy'),
    'MLP':             lambda f,s: acc(f'results/baselines_erm_mlp_{f}_s{s}.json', 'mlp','unseen','accuracy'),
    'IRM':             lambda f,s: acc(f'results/baselines_irm_eerm_{f}_s{s}.json','irm','unseen','accuracy'),
    'EERM':            lambda f,s: acc(f'results/baselines_irm_eerm_{f}_s{s}.json','eerm','unseen','accuracy'),
}

rows3 = []
for mname, loader in models3.items():
    fold_means = []
    for ftag, _ in folds:
        vs = [loader(ftag, s) for s in [42, 1, 2]]
        vs = [v for v in vs if v is not None]
        fold_means.append(statistics.mean(vs)*100)
    grand = statistics.mean(fold_means)
    rows3.append((mname, fold_means, grand))

rows3.sort(key=lambda x: -x[2])
for rank, (mname, fold_means, grand) in enumerate(rows3, 1):
    vstr = '  '.join(f'{v:6.2f}' for v in fold_means)
    print(f'{mname:<20} {vstr}  {grand:6.2f}   #{rank}')
