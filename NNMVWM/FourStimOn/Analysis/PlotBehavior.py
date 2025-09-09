import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 1) Load and squeeze your data (shape → (15, 100))
x25  = np.load("hits_cueValid25.npy").squeeze(-1)
x50  = np.load("hits_cueValid50.npy").squeeze(-1)
x75  = np.load("hits_cueValid75.npy").squeeze(-1)
x100 = np.load("hits_cueValid100.npy").squeeze(-1)

# 2) Compute mean and SEM over the 100 samples
def mean_sem(arr):
    m = arr.mean(axis=1)
    s = arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])
    return m, s

mean25, sem25   = mean_sem(x25)
mean50, sem50   = mean_sem(x50)
mean75, sem75   = mean_sem(x75)
mean100, sem100 = mean_sem(x100)

# 3) Smooth each mean with a 5-point moving average, reflecting at the edges
sm25   = uniform_filter1d(mean25,  size=5, mode='reflect')
sm50   = uniform_filter1d(mean50,  size=5, mode='reflect')
sm75   = uniform_filter1d(mean75,  size=5, mode='reflect')
sm100  = uniform_filter1d(mean100, size=5, mode='reflect')

# 4) Prepare x-axis (Δ = 1…15)
x = np.arange(1, sm25.shape[0] + 1)

# 5) Plot publication-quality figure
plt.figure(figsize=(8, 5), dpi=300)

colors = {
    'Cue 25':  'tab:blue',
    'Cue 50':  'tab:orange',
    'Cue 75':  'tab:green',
    'Cue 100': 'tab:red'
}

plt.errorbar(x, sm25,   yerr=sem25,   label='Cue 25',
             color=colors['Cue 25'],   linestyle='-', linewidth=3, capsize=3)
plt.errorbar(x, sm50,   yerr=sem50,   label='Cue 50',
             color=colors['Cue 50'],   linestyle='-', linewidth=3, capsize=3)
plt.errorbar(x, sm75,   yerr=sem75,   label='Cue 75',
             color=colors['Cue 75'],   linestyle='-', linewidth=3, capsize=3)
plt.errorbar(x, sm100,  yerr=sem100,  label='Cue 100',
             color=colors['Cue 100'],  linestyle='-', linewidth=3, capsize=3)

plt.xlabel('Δ', fontsize=16)
plt.ylabel('Ĥ', fontsize=16)
plt.title('Smoothed Hit Rate across Δ for Different Cue Conditions',
          fontsize=16, fontweight='bold')

plt.xlim(1, sm25.shape[0])
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=14, loc='lower right')

plt.tight_layout()
plt.savefig(f"Beh_CueALL.png", dpi=150, bbox_inches='tight')
plt.show()
