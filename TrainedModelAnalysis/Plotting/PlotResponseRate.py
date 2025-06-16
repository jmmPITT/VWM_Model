import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.optimize import curve_fit


def logistic_func(x, A, B, C, D):
    return A + (1 - B) / (1 + np.exp(-C * (x - D)))

# Load data
alpha10 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual10_DeltaModulation.npy")[:, 1]
alpha09 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual09_DeltaModulation.npy")[:, 1]
alpha08 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual08_DeltaModulation.npy")[:, 1]
alpha07 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual07_DeltaModulation.npy")[:, 1]
alpha06 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual06_DeltaModulation.npy")[:, 1]
alpha05 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual05_DeltaModulation.npy")[:, 1]
alpha04 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual04_DeltaModulation.npy")[:, 1]
alpha03 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual03_DeltaModulation.npy")[:, 1]
alpha02 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual02_DeltaModulation.npy")[:, 1]
alpha01 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual01_DeltaModulation.npy")[:, 1]

# statsCue100 = np.load("ChoiceStats_CueS1100_ChangeS4_Alpha4_tChange.npy")

def jeffreys_interval(successes, trials, confidence=0.95):
    alpha = 0.5 + successes
    beta = 0.5 + trials - successes
    lower = stats.beta.ppf((1-confidence)/2, alpha, beta)
    upper = stats.beta.ppf(1-(1-confidence)/2, alpha, beta)
    return lower, upper

# Calculate hit rates and credible intervals
hits100, trials100 = alpha10[:, 1], 500
thetaH10 = hits100 / trials100
print('THETA10',thetaH10.shape)
lower10, upper10 = jeffreys_interval(hits100, trials100)

hits100, trials100 = alpha09[:, 1], 500
thetaH09 = hits100 / trials100
print('THETA10',thetaH10.shape)
lower09, upper09 = jeffreys_interval(hits100, trials100)

hits100, trials100 = alpha08[:, 1], 500
thetaH08 = hits100 / trials100
print('THETA10',thetaH08.shape)
lower08, upper08 = jeffreys_interval(hits100, trials100)

hits100, trials100 = alpha07[:, 1], 500
thetaH07 = hits100 / trials100
print('THETA06',thetaH07.shape)
lower07, upper07 = jeffreys_interval(hits100, trials100)

hits100, trials100 = alpha06[:, 1], 500
thetaH06 = hits100 / trials100
print('THETA06',thetaH06.shape)
lower06, upper06 = jeffreys_interval(hits100, trials100)


hits100, trials100 = alpha05[:, 1], 500
thetaH05 = hits100 / trials100
print('THETA05',thetaH05.shape)
lower05, upper05 = jeffreys_interval(hits100, trials100)


hits100, trials100 = alpha04[:, 1], 500
thetaH04 = hits100 / trials100
print('THETA04',thetaH04.shape)
lower04, upper04 = jeffreys_interval(hits100, trials100)


hits100, trials100 = alpha03[:, 1], 500
thetaH03 = hits100 / trials100
print('THETA03',thetaH03.shape)
lower03, upper03 = jeffreys_interval(hits100, trials100)


hits100, trials100 = alpha02[:, 1], 500
thetaH02 = hits100 / trials100
print('THETA02',thetaH02.shape)
lower02, upper02 = jeffreys_interval(hits100, trials100)


hits100, trials100 = alpha01[:, 1], 500
thetaH01 = hits100 / trials100
print('THETA01',thetaH01.shape)
lower01, upper01 = jeffreys_interval(hits100, trials100)

delta = np.linspace(0, 45, 35)

# Initial parameter guesses
initial_guess = [0.0, 0.5, 1.0, np.median(delta)]
# Define lower and upper bounds for parameters A, B, C, D
lower_bounds = [-np.inf, 0, -np.inf, -np.inf]  # No lower bounds
upper_bounds = [np.inf, 1.0, np.inf, np.inf]         # Upper bound for B is 1.0

# Fit for thetaH25 with bounds
params10, covariance10 = curve_fit(
    logistic_func, delta, thetaH10, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A10, B10, C10, D10 = params10

# Fit for thetaH25 with bounds
params09, covariance09 = curve_fit(
    logistic_func, delta, thetaH09, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A09, B09, C09, D09 = params09

# Fit for thetaH25 with bounds
params08, covariance08 = curve_fit(
    logistic_func, delta, thetaH08, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A08, B08, C08, D08 = params08


# Fit for thetaH25 with bounds
params07, covariance07 = curve_fit(
    logistic_func, delta, thetaH07, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A07, B07, C07, D07 = params07


params06, covariance06 = curve_fit(
    logistic_func, delta, thetaH06, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A06, B06, C06, D06 = params06


params05, covariance05 = curve_fit(
    logistic_func, delta, thetaH05, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A05, B05, C05, D05 = params05


params04, covariance04 = curve_fit(
    logistic_func, delta, thetaH04, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A04, B04, C04, D04 = params04


params03, covariance03 = curve_fit(
    logistic_func, delta, thetaH03, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A03, B03, C03, D03 = params03


params02, covariance02 = curve_fit(
    logistic_func, delta, thetaH02, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A02, B02, C02, D02 = params02


params01, covariance01 = curve_fit(
    logistic_func, delta, thetaH01, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A01, B01, C01, D01 = params01

colors = [(i, 0, 1 - i) for i in np.linspace(0, 1, 10)]



# Create the plot
fig, ax = plt.subplots(figsize=(13, 8))
# Plot lines and points
ax.plot(delta, thetaH10, 'o', color=colors[0], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH10, yerr=[thetaH10 - lower10, np.abs(upper10 - thetaH10)], fmt='none',
                    ecolor=colors[0], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH09, 'o', color=colors[1], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH09, yerr=[thetaH09 - lower09, np.abs(upper09 - thetaH09)], fmt='none',
                    ecolor=colors[1], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH08, 'o', color=colors[2], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH08, yerr=[thetaH08 - lower08, np.abs(upper08 - thetaH08)], fmt='none',
                    ecolor=colors[2], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH07, 'o', color=colors[3], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH07, yerr=[thetaH07 - lower07, np.abs(upper07 - thetaH07)], fmt='none',
                    ecolor=colors[3], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH06, 'o', color=colors[4], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH06, yerr=[thetaH06 - lower06, np.abs(upper06 - thetaH06)], fmt='none',
                    ecolor=colors[4], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH05, 'o', color=colors[5], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH05, yerr=[thetaH05 - lower05, np.abs(upper05 - thetaH05)], fmt='none',
                    ecolor=colors[5], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH04, 'o', color=colors[6], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH04, yerr=[thetaH04 - lower04, np.abs(upper04 - thetaH04)], fmt='none',
                    ecolor=colors[6], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH03, 'o', color=colors[7], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH03, yerr=[thetaH03 - lower03, np.abs(upper03 - thetaH03)], fmt='none',
                    ecolor=colors[7], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH02, 'o', color=colors[8], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH02, yerr=[thetaH02 - lower02, np.abs(upper02 - thetaH02)], fmt='none',
                    ecolor=colors[8], elinewidth=2, capsize=4, alpha=0.7)

# Plot lines and points
ax.plot(delta, thetaH01, 'o', color=colors[9], markersize=8, alpha=0.7)
# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH01, yerr=[thetaH01 - lower01, np.abs(upper01 - thetaH01)], fmt='none',
                    ecolor=colors[9], elinewidth=2, capsize=4, alpha=0.7)


delta_fine = np.linspace(delta.min(), delta.max(), 300)

# Generate fitted curves
fitted_curve10 = logistic_func(delta_fine, A10, B10, C10, D10)
fitted_curve09 = logistic_func(delta_fine, A09, B09, C09, D09)
fitted_curve08 = logistic_func(delta_fine, A08, B08, C08, D08)
fitted_curve07 = logistic_func(delta_fine, A07, B07, C07, D07)
fitted_curve06 = logistic_func(delta_fine, A06, B06, C06, D06)
fitted_curve05 = logistic_func(delta_fine, A05, B05, C05, D05)
fitted_curve04 = logistic_func(delta_fine, A04, B04, C04, D04)
fitted_curve03 = logistic_func(delta_fine, A03, B03, C03, D03)
fitted_curve02 = logistic_func(delta_fine, A02, B02, C02, D02)
fitted_curve01 = logistic_func(delta_fine, A01, B01, C01, D01)

# Create a color gradient from blue to red for the curves
colors = [(i, 0, 1 - i) for i in np.linspace(0, 1, 10)]

# Plot fitted logistic curves with the gradient colors
ax.plot(delta_fine, fitted_curve10, '-', color=colors[0], linewidth=5, label=r'$\alpha_1=1.0, \alpha_4=0.0$')
ax.plot(delta_fine, fitted_curve09, '-', color=colors[1], linewidth=5, label=r'$\alpha_1=0.9, \alpha_4=0.1$')
ax.plot(delta_fine, fitted_curve08, '-', color=colors[2], linewidth=5, label=r'$\alpha_1=0.8, \alpha_4=0.2$')
ax.plot(delta_fine, fitted_curve07, '-', color=colors[3], linewidth=5, label=r'$\alpha_1=0.7, \alpha_4=0.3$')
ax.plot(delta_fine, fitted_curve06, '-', color=colors[4], linewidth=5, label=r'$\alpha_1=0.6, \alpha_4=0.4$')
ax.plot(delta_fine, fitted_curve05, '-', color=colors[5], linewidth=5, label=r'$\alpha_1=0.5, \alpha_4=0.5$')
ax.plot(delta_fine, fitted_curve04, '-', color=colors[6], linewidth=5, label=r'$\alpha_1=0.4, \alpha_4=0.6$')
ax.plot(delta_fine, fitted_curve03, '-', color=colors[7], linewidth=5, label=r'$\alpha_1=0.3, \alpha_4=0.7$')
ax.plot(delta_fine, fitted_curve02, '-', color=colors[8], linewidth=5, label=r'$\alpha_1=0.2, \alpha_4=0.8$')
ax.plot(delta_fine, fitted_curve01, '-', color=colors[9], linewidth=5, label=r'$\alpha_1=0.1, \alpha_4=0.9$')

# Customize the plot
ax.set_xlabel(r'$\Delta$', fontsize=26)
ax.set_ylabel(r'Mean Response Percentages', fontsize=26)
ax.set_title(r'Parametric $\alpha$ Cue 25% $S_1$', fontsize=30)

# Set x-axis and y-axis limits and ticks
ax.set_xlim(0, 45)
ax.set_xticks(np.arange(0, 46, 5))
ax.set_xlim(delta.min(), delta.max())
ax.set_ylim(0, 1.01)
yticks = np.linspace(0, 1, 11)  # Ticks at 0.0, 0.1, ..., 1.0
ax.set_yticks(yticks)
ax.set_yticklabels(['{:.0f}%'.format(ytick * 100) for ytick in yticks])
# Increase tick width and length
ax.tick_params(axis='both', which='major', width=2, length=10, labelsize=22)
ax.tick_params(axis='both', which='minor', width=1.5, length=5)
ax.tick_params(axis='x', which='both', labelbottom=True)  # Remove x-axis labels


# Add legend
ax.legend(loc='center right', frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray', fontsize=18)

# Add grid (major and minor)
# ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
# ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('logistic_fit_RRplot_Cue25AttentionMods.pdf', dpi=300, bbox_inches='tight')
plt.show()
