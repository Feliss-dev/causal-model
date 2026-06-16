import json, glob, os, statistics

files = sorted(glob.glob('results/baselines_irm_eerm_loco*.json'))

results = {}
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    name = os.path.basename(f).replace('baselines_irm_eerm_','').replace('.json','')
    fold, seed = name.rsplit('_', 1)
    results[(fold, seed)] = {
        'irm':  d['irm']['unseen']['accuracy'],
        'eerm': d['eerm']['unseen']['accuracy'],
        'irm_f1':   d['irm']['unseen']['f1'],
        'eerm_f1':  d['eerm']['unseen']['f1'],
        'irm_auc':  d['irm']['unseen']['auc'],
        'eerm_auc': d['eerm']['unseen']['auc'],
    }

folds = ['locoa', 'locob', 'lococ']
seeds = ['s42', 's1', 's2']
fold_labels = {
    'locoa': 'nottheonion + pareidolia       ',
    'locob': 'upliftingnews + fakehistoryporn',
    'lococ': 'usnews+usanews+fakealbumcovers ',
}

def fmt(v): return f"{v*100:.2f}"

print("=== IRM Unseen Accuracy (%) ===")
print(f"{'Fold':<33} {'seed=42':>8} {'seed=1':>8} {'seed=2':>8} {'mean':>8} {'std':>7}")
print("-" * 77)
irm_fold_means = []
for fold in folds:
    vals = [results[(fold, s)]['irm'] for s in seeds]
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    irm_fold_means.append(m)
    print(f"{fold_labels[fold]} {fmt(vals[0]):>8} {fmt(vals[1]):>8} {fmt(vals[2]):>8} {m*100:>8.2f} {s*100:>7.2f}")
gm = statistics.mean(irm_fold_means)
gs = statistics.stdev(irm_fold_means)
print(f"{'Grand mean (across folds)':<33} {'':>8} {'':>8} {'':>8} {gm*100:>8.2f} {gs*100:>7.2f}")

print()
print("=== EERM Unseen Accuracy (%) ===")
print(f"{'Fold':<33} {'seed=42':>8} {'seed=1':>8} {'seed=2':>8} {'mean':>8} {'std':>7}")
print("-" * 77)
eerm_fold_means = []
for fold in folds:
    vals = [results[(fold, s)]['eerm'] for s in seeds]
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    eerm_fold_means.append(m)
    print(f"{fold_labels[fold]} {fmt(vals[0]):>8} {fmt(vals[1]):>8} {fmt(vals[2]):>8} {m*100:>8.2f} {s*100:>7.2f}")
gm = statistics.mean(eerm_fold_means)
gs = statistics.stdev(eerm_fold_means)
print(f"{'Grand mean (across folds)':<33} {'':>8} {'':>8} {'':>8} {gm*100:>8.2f} {gs*100:>7.2f}")

print()
print("=== Summary so sanh IRM vs EERM (mean across 3 seeds) ===")
print(f"{'Fold':<33} {'IRM mean':>10} {'EERM mean':>10}")
print("-" * 55)
for i, fold in enumerate(folds):
    print(f"{fold_labels[fold]} {irm_fold_means[i]*100:>10.2f} {eerm_fold_means[i]*100:>10.2f}")
print(f"{'Grand mean':<33} {statistics.mean(irm_fold_means)*100:>10.2f} {statistics.mean(eerm_fold_means)*100:>10.2f}")
