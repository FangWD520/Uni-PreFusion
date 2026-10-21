from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

out_dir = Path(__file__).resolve().parents[1] / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

metrics = ['NDS ↑', 'Map mIoU ↑', 'RayIoU ↑', 'minADE\n(Car) ↓', 'minADE\n(Ped) ↓', 'Planning\nL2 ↓']
resnet = np.array([74.2, 69.4, 48.0, 0.60, 0.39, 0.69], dtype=float)
vmamba = np.array([74.6, 69.5, 48.6, 0.60, 0.38, 0.66], dtype=float)
higher_is_better = np.array([True, True, True, False, False, False])

resnet_score = np.ones_like(resnet) * 100.0
vmamba_score = np.where(higher_is_better, vmamba / resnet * 100.0, resnet / vmamba * 100.0)

deltas = []
for r, v, hib in zip(resnet, vmamba, higher_is_better):
    if hib:
        deltas.append(f'+{v - r:.1f}')
    else:
        diff = v - r
        deltas.append('same' if abs(diff) < 1e-8 else f'{diff:.2f}')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(7.2, 2.75), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('#fbfbfb')

for i in range(len(metrics)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color='#f2f4f7', alpha=0.7, zorder=0)

ax.plot(x, resnet_score, color='#8a8f98', lw=2.2, marker='o', ms=6.5,
        mfc='white', mec='#8a8f98', mew=1.8, label='ResNet50')
ax.plot(x, vmamba_score, color='#5b6ee1', lw=2.8, marker='o', ms=7.0,
        mfc='#5b6ee1', mec='white', mew=1.4, label='VMamba')
ax.fill_between(x, resnet_score, vmamba_score, where=vmamba_score >= resnet_score,
                color='#5b6ee1', alpha=0.12, interpolate=True, zorder=1)

for i, (ys, txt) in enumerate(zip(vmamba_score, deltas)):
    ax.text(i, ys + 0.32, txt, ha='center', va='bottom',
            color='#3949ab', fontsize=8.5, fontweight='bold')

ax.axhline(100, color='#b8bdc7', lw=1.2, ls='--', zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_xlim(-0.5, len(metrics) - 0.5)
ax.set_ylim(99.2, max(vmamba_score.max() + 1.2, 105.0))
ax.grid(axis='y', color='#d9dde5', lw=0.8, alpha=0.8)
ax.grid(axis='x', visible=False)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#c6cad3')
ax.spines['bottom'].set_color('#c6cad3')

leg = ax.legend(loc='upper left', frameon=True, ncol=2, bbox_to_anchor=(0.01, 0.99),
                borderpad=0.35, handlelength=2.2, columnspacing=1.2)
leg.get_frame().set_edgecolor('#d9dde5')
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_alpha(0.95)

ax.text(0.99, 0.03, 'normalized score', transform=ax.transAxes,
        ha='right', va='bottom', color='#6b7280', fontsize=8)

plt.tight_layout(pad=0.5)
fig.savefig(out_dir / 'backbone_comparison.pdf', bbox_inches='tight')
fig.savefig(out_dir / 'backbone_comparison.png', dpi=300, bbox_inches='tight')
