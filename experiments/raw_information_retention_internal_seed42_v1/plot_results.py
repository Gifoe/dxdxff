"""Publication-size Python figures from verified anonymous first-path tables."""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_report import LABELS, COMPONENTS, load_complete

SHORT_LABELS = [
    'Window / short', 'Pre-scale / short', 'Window / long',
    'Pre-scale / long', 'Pre-scale / long + context',
    'No pre-latent reference', 'No temporal phases',
    'No seizure SD', 'No patient Z',
]
COLORS = ['#525A65', '#5680A5', '#5680A5', '#315B85', '#8C6F9B',
          '#A37654', '#A37654', '#A37654', '#A37654']
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})


def export(fig, output):
    fig.savefig(output.with_suffix('.svg'))
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=600)
    fig.savefig(output.with_suffix('.tiff'), dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def letter(ax, text):
    ax.text(-.16, 1.06, text, transform=ax.transAxes, fontweight='bold',
            fontsize=8, va='top', ha='left')


def main(root, out):
    terminal, verification, values = load_complete(root)
    absolute = pd.read_csv(root / 'FIRST_PATH_ABSOLUTE_INTERVALS.csv')
    pairs = pd.read_csv(root / 'FIRST_PATH_PAIRED_INTERVALS.csv')
    folds = pd.read_csv(root / 'FIRST_PATH_FOLD_RESULTS.csv')
    assert set(folds.arm) == set(LABELS) and len(folds) == 45
    out.mkdir(parents=True, exist_ok=True)
    n = verification['first_path_logical_patients_per_arm']
    u = verification['first_path_unique_patients']
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 115 / 25.4),
                             gridspec_kw={'width_ratios': [1.1, 1]})
    for i, arm in enumerate(LABELS):
        r = absolute[(absolute.arm == arm) & (absolute.metric == 'f1')].iloc[0]
        axes[0].errorbar(r['mean'], i, xerr=[[r['mean'] - r.low], [r.high - r['mean']]],
                         fmt='o', color=COLORS[i], ms=3.5, lw=1.1, capsize=2)
        # Show every registered fold estimate, not selected examples.
        a = folds[folds.arm == arm].sort_values('fold')
        axes[0].scatter(a.f1, np.full(5, i) + np.linspace(-.13, .13, 5),
                        color=COLORS[i], s=7, alpha=.55, marker='|', zorder=3)
        for metric, marker, color, shift in [('auroc', 's', '#315B85', -.09),
                                              ('auprc', '^', '#A37654', .09)]:
            q = absolute[(absolute.arm == arm) & (absolute.metric == metric)].iloc[0]
            axes[1].errorbar(q['mean'], i + shift,
                             xerr=[[q['mean'] - q.low], [q.high - q['mean']]],
                             fmt=marker, color=color, ms=3, lw=.9, capsize=1.5,
                             label={'auroc': 'AUROC', 'auprc': 'AP'}[metric] if i == 0 else None)
    for ax in axes:
        ax.set_ylim(8.5, -.5)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', color='#E8EBEE', lw=.5, zorder=0)
        ax.set_yticks(range(9))
        ax.axhline(4.5, color='#BDC4CC', lw=.7, ls=':')
    axes[0].set_yticklabels(SHORT_LABELS)
    axes[1].set_yticklabels([])
    axes[0].set_xlabel('Patient EZ-F1')
    axes[1].set_xlabel('Patient ranking performance')
    axes[1].legend(loc='lower left')
    axes[0].set_title('Decision performance', loc='left', fontsize=8)
    axes[1].set_title('Ranking performance', loc='left', fontsize=8)
    letter(axes[0], 'a'); letter(axes[1], 'b')
    fig.subplots_adjust(left=.29, right=.98, top=.88, bottom=.20, wspace=.15)
    fig.text(.29, .08, f'{n} patient appearances / {u} unique patients per arm\n'
             'Internal selection paths; 95% patient-ID cluster intervals, not independent test validation',
             fontsize=6.3, va='center')
    export(fig, out / 'internal_information_retention')

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 72 / 25.4))
    names = ['Pre-latent reference', 'Temporal phases', 'Seizure SD', 'Patient Z']
    for panel, (ax, metric) in enumerate(zip(axes, ['f1', 'auroc'])):
        for i, arm in enumerate(COMPONENTS):
            r = pairs[(pairs.left == arm) & (pairs.right == 'C0_WINDOW_SHORT') & (pairs.metric == metric)]
            assert len(r) == 1
            r = r.iloc[0]
            center, low, high = -r['mean'], -r.high, -r.low
            ax.errorbar(center, i, xerr=[[center - low], [high - center]],
                        fmt='o', color='#315B85', ms=4, capsize=2, lw=1.1)
        ax.set_ylim(3.5, -.5)
        ax.set_yticks(range(4), names if panel == 0 else [])
        ax.axvline(0, color='#51565C', lw=.7, ls='--')
        ax.grid(axis='x', color='#E8EBEE', lw=.5)
        ax.set_title({'f1': 'EZ-F1', 'auroc': 'AUROC'}[metric], loc='left', fontsize=8)
        ax.set_xlabel('Retained minus deleted')
        letter(ax, chr(97 + panel))
    fig.subplots_adjust(left=.24, right=.98, top=.82, bottom=.28, wspace=.17)
    fig.text(.24, .08, 'Matched end-to-end controls; 95% patient-ID cluster intervals\n'
             'Positive values favor retaining the component. Selection-conditioned internal evidence.',
             fontsize=6.3, va='center')
    export(fig, out / 'component_retention_effects')
    (out / 'FIGURE_QA.json').write_text(json.dumps({
        'status': 'SOURCE_CHECKED_PENDING_VISUAL_REVIEW',
        'backend': 'python_matplotlib', 'width_mm': 183,
        'dpi': 600, 'editable_vector_text': True, 'all_nine_arms_shown': True,
        'all_five_fold_points_shown': True, 'statistical_unit': 'patient_ID_cluster',
        'interval': '10000-draw descriptive selection-conditioned percentile 95%',
        'logical_patient_appearances_per_arm': n, 'unique_patients': u,
        'new_outer_passes': terminal['new_Outer_passes'],
        'source_files': ['FIRST_PATH_MAIN_RESULTS.csv', 'FIRST_PATH_ABSOLUTE_INTERVALS.csv',
                         'FIRST_PATH_PAIRED_INTERVALS.csv', 'FIRST_PATH_FOLD_RESULTS.csv'],
        'source_data_exclusions': 0, 'additional_raced_paths_not_mixed_with_balanced_first_path': True,
        'raster_transformations': 'none; figures rendered directly from aggregate source tables',
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.output)
