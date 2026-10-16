"""Figures from verified anonymous aggregates; no model selection or private I/O."""
import itertools
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'results' / 'dense_reference'
OUT = HERE / 'figures'
METRICS = ['auroc', 'auprc', 'macro_f1', 'f1', 'nez_f1', 'accuracy',
           'mrr', 'hit_at_1', 'ez_fraction_bias']
LABELS = ['AUROC', 'AP', 'Macro-F1', 'EZ-F1', 'NEZ-F1', 'ACC',
          'MRR', 'Top-1', 'EZ-frac. bias']

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.1,
    'legend.frameon': False,
})


def export(fig, name):
    svg = OUT / f'{name}.svg'
    fig.savefig(svg, facecolor='white')
    # Mechanical formatting of the generated SVG leaves its geometry intact.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    fig.savefig(OUT / f'{name}.pdf', facecolor='white')
    fig.savefig(OUT / f'{name}.png', dpi=600, facecolor='white')
    fig.savefig(OUT / f'{name}.tiff', dpi=600, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def main():
    assert json.loads((DATA / 'INTERNAL_REPLAY_VERIFICATION.json').read_text())['status'] == 'PASS'
    assert json.loads((DATA / 'INTERNAL_REPORT_STATUS.json').read_text())['status'] == 'COMPLETE_INTERNAL_ONLY_CONDITIONAL_SCREEN_VERIFIED'
    OUT.mkdir(exist_ok=True)
    cardinality = pd.read_csv(DATA / 'ASSESSMENT_CARDINALITY.csv')
    path = cardinality[cardinality.recipe == 'path'].copy()
    assert len(path) == 50 and set(path.k) == set(range(10))
    mean = []
    for k, rows in path.groupby('k'):
        w = rows.patient_appearances.to_numpy()
        mean.append(dict(k=int(k), patient_appearances=int(w.sum()),
                         **{m:float(np.average(rows[m], weights=w)) for m in METRICS}))
    pd.DataFrame(mean).to_csv(OUT / 'SOURCE_CARDINALITY.csv', index=False)
    selected = pd.read_csv(DATA / 'FROZEN_INTERNAL_SELECTION.csv')
    selected = selected[selected.recipe == 'SELECTED'].set_index('fold')
    stability = pd.read_csv(DATA / 'CONDITIONAL_SELECTION_STABILITY.csv')
    assert len(stability) == 160
    order = stability[stability.fold == 1].feature.tolist()
    frequency = stability.pivot(index='feature', columns='fold', values='selected_replicates').loc[order] / 10
    frequency.to_csv(OUT / 'SOURCE_SELECTION_FREQUENCY.csv')
    jaccard = stability.groupby('fold')[['k', 'mean_pairwise_jaccard']].first().reset_index()
    jaccard.to_csv(OUT / 'SOURCE_WEIGHT_STABILITY.csv', index=False)
    sets = [set(v.split(';')) if pd.notna(v) else set() for v in selected.features]
    pairs = [dict(fold_a=i+1, fold_b=j+1, jaccard=len(a & b) / len(a | b) if a | b else 1.)
             for (i,a),(j,b) in itertools.combinations(enumerate(sets), 2)]
    pd.DataFrame(pairs).to_csv(OUT / 'SOURCE_FINAL_FOLD_OVERLAP.csv', index=False)

    # Quantitative grid: cardinality on the left, all32 frequencies on the right.
    width_inches = 183 / 25.4
    height_inches = 157 / 25.4
    fig = plt.figure(figsize=(width_inches, height_inches), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1., 1.20], wspace=.20, hspace=.16)
    for i, (metric, label) in enumerate([('auroc', 'Patient AUROC'), ('macro_f1', 'Patient Macro-F1')]):
        ax = fig.add_subplot(grid[i, 0])
        for fold, rows in path.groupby('fold'):
            rows = rows.sort_values('k')
            ax.plot(rows.k, rows[metric], color='#B3B9BF', lw=.7, alpha=.8)
            chosen = rows[rows.k == selected.loc[fold, 'k']].iloc[0]
            ax.scatter([chosen.k], [chosen[metric]], s=25, facecolors='white',
                       edgecolors='#626A73', zorder=4, linewidths=.8)
            ax.annotate(str(fold), (chosen.k, chosen[metric]), xytext=(3, 3),
                        textcoords='offset points', fontsize=5.5)
        mm = pd.DataFrame(mean)
        ax.plot(mm.k, mm[metric], color='#326F9B', lw=1.7, marker='o', markersize=3,
                label='Appearance-weighted mean')
        ax.set(xlabel='Added physiological groups, k', ylabel=label, xticks=range(10))
        ax.text(-.12, 1.04, chr(97+i), transform=ax.transAxes, weight='bold', fontsize=9)
        if i == 0:
            ax.set_title('Own-FIT assessment, conditioned on Raw')
            ax.legend(loc='lower right', fontsize=6)
        else:
            ax.text(.03, .02, 'Grey: five original folds\nOpen points: chosen k (fold number)',
                    transform=ax.transAxes, fontsize=6, va='bottom')
    ax = fig.add_subplot(grid[:, 1])
    image = ax.imshow(frequency.to_numpy(), vmin=0, vmax=1, cmap='Blues', aspect='auto')
    pretty = [n.replace('rebuilt_', 'Graph: ').replace('partial_high_gamma', 'partial HG')
              .replace('line_length_per_sec', 'Line length / s').replace('spectral_entropy_10_125', 'Spectral entropy (10-125)')
              .replace('stable_log_', 'stable log ').replace('_sameband', '').replace('_', ' ')
              for n in order]
    ax.set(yticks=range(32), yticklabels=pretty, xticks=range(5),
           xticklabels=[f'{f}\n(k={selected.loc[f,"k"]})' for f in range(1,6)], xlabel='Original fold')
    ax.tick_params(axis='y', labelsize=5.5, length=0)
    ax.set_title('Ten training-patient weight re-selections')
    ax.text(-.08, 1.025, 'c', transform=ax.transAxes, weight='bold', fontsize=9)
    for row in range(32):
        for col in range(5):
            value = frequency.iloc[row, col]
            if value:
                ax.text(col, row, f'{value:.1f}', ha='center', va='center', fontsize=5.2,
                        color='white' if value > .55 else '#1F2F40')
    bar = fig.colorbar(image, ax=ax, shrink=.32, pad=.025, ticks=[0, .5, 1])
    bar.set_label('Selection frequency', fontsize=6)
    export(fig, '01_conditional_cardinality_stability')

    intervals = pd.read_csv(DATA / 'INTERNAL_PAIRED_INTERVALS.csv')
    contrast = intervals[intervals.reference == 'RAW_ONLY'].set_index('metric').loc[METRICS].reset_index()
    contrast.to_csv(OUT / 'SOURCE_CONDITIONAL_INCREMENT.csv', index=False)
    source = pd.read_csv(DATA / 'INTERNAL_SOURCE_RESULTS.csv')
    raw = source[source.model == 'RAW_ONLY'].set_index('center')
    chosen = source[source.model == 'SELECTED'].set_index('center').loc[raw.index]
    source_delta = chosen[['patient_appearances', 'unique_patients']].copy()
    for m in METRICS:
        source_delta[m] = chosen[m] - raw[m]
    source_delta.to_csv(OUT / 'SOURCE_SOURCE_INCREMENT.csv')
    height_inches = 80 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_inches, height_inches), layout='constrained',
                             gridspec_kw={'width_ratios': [1.05, 1.]})
    ax = axes[0]
    y = np.arange(9)
    ax.errorbar(contrast.delta, y,
                xerr=np.vstack([contrast.delta - contrast.ci_low, contrast.ci_high - contrast.delta]),
                fmt='o', markersize=3, color='#326F9B', capsize=2, lw=.9)
    ax.axvline(0, color='#8B9197', lw=.7, ls='--')
    ax.set(yticks=y, yticklabels=LABELS, xlabel='Selected minus Raw-only (ridge heads)',
           title='Internal, selection-conditioned 95% intervals')
    ax.invert_yaxis()
    ax.text(-.12, 1.06, 'a', transform=ax.transAxes, weight='bold', fontsize=9)
    ax = axes[1]
    positions = np.arange(len(source_delta))
    for off, m, name, color in [(-.13, 'auroc', 'AUROC', '#326F9B'), (.13, 'f1', 'EZ-F1', '#B67C4D')]:
        ax.scatter(source_delta[m], positions + off, s=22, marker='o' if off < 0 else 's', label=name, color=color)
    ax.axvline(0, color='#8B9197', lw=.7, ls='--')
    ax.set(yticks=positions,
           yticklabels=[f'{s} (n={int(source_delta.loc[s,"unique_patients"])})' for s in source_delta.index],
           xlabel='Selected minus Raw-only', title='Known-source strata, not held-source validation')
    ax.invert_yaxis()
    ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2)
    ax.text(-.1, 1.06, 'b', transform=ax.transAxes, weight='bold', fontsize=9)
    export(fig, '02_internal_increment_source')
    print(pd.DataFrame(mean).to_string(index=False))
    print(jaccard.to_string(index=False))
    print('Final fold mean Jaccard:', np.mean([r['jaccard'] for r in pairs]))


if __name__ == '__main__':
    main()
