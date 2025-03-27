import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.optimize import curve_fit


def logistic_func(x, A, B, C, D):
    return A + (1 - B) / (1 + np.exp(-C * (x - D)))

# Load data
statsCue25 = np.load("ChoiceStats_CueS125_ChangeS1_NoArtificialBias_DeltaModulation.npy")
statsCue25Alpha1 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual1_DeltaModulation.npy")
statsCue25Alpha4 = np.load("ChoiceStats_CueS125_ChangeS1_alpha4TchangeEqual1_DeltaModulation.npy")


statsCue25 = statsCue25[35:]
statsCue25Alpha1 = statsCue25Alpha1[35:]
statsCue25Alpha4 = statsCue25Alpha4[35:]

# statsCue100 = np.load("ChoiceStats_CueS1100_ChangeS4_Alpha4_tChange.npy")

def jeffreys_interval(successes, trials, confidence=0.95):
    alpha = 0.5 + successes
    beta = 0.5 + trials - successes
    lower = stats.beta.ppf((1-confidence)/2, alpha, beta)
    upper = stats.beta.ppf(1-(1-confidence)/2, alpha, beta)
    return lower, upper

# Calculate hit rates and credible intervals
hits25, trials25 = statsCue25[:,1], statsCue25[:,1] + statsCue25[:,0]
thetaH25 = hits25 / trials25
print('THETA25',thetaH25.shape)
lower25, upper25 = jeffreys_interval(hits25, trials25)
# print('lowerDelC40',thetaH25-lowerDelC40)
# print('upperDelC40', upperDelC40-thetaH25)

hits25Alpha1, trials25Alpha1 = statsCue25Alpha1[:,1], statsCue25Alpha1[:,1] + statsCue25Alpha1[:,0]
thetaH25Alpha1 = hits25Alpha1 / trials25Alpha1
# print('THETA25',thetaH25)
lower25Alpha1, upper25Alpha1 = jeffreys_interval(hits25Alpha1, trials25Alpha1)
# print('lowerDelC40',thetaH25-lowerDelC40)
# print('upperDelC40', upperDelC40-thetaH25)


hits25Alpha4, trials25Alpha4 = statsCue25Alpha4[:,1], statsCue25Alpha4[:,1] + statsCue25Alpha4[:,0]
thetaH25Alpha4 = hits25Alpha4 / trials25Alpha4
# print('THETA25',thetaH25)
lower25Alpha4, upper25Alpha4 = jeffreys_interval(hits25Alpha4, trials25Alpha4)
# print('lowerDelC40',thetaH25-lowerDelC40)
# print('upperDelC40', upperDelC40-thetaH25)



delta = np.linspace(0, 45, 35)

# Initial parameter guesses
initial_guess = [0.0, 0.5, 1.0, np.median(delta)]
# Define lower and upper bounds for parameters A, B, C, D
lower_bounds = [-np.inf, 0, -np.inf, -np.inf]  # No lower bounds
upper_bounds = [np.inf, 1.0, np.inf, np.inf]         # Upper bound for B is 1.0

# Fit for thetaH25 with bounds
params25, covariance25 = curve_fit(
    logistic_func, delta, thetaH25, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A25, B25, C25, D25 = params25

# Fit for thetaH25Alpha1 with bounds
params25Alpha1, covariance25Alpha1 = curve_fit(
    logistic_func, delta, thetaH25Alpha1, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A25Alpha1, B25Alpha1, C25Alpha1, D25Alpha1 = params25Alpha1

# Fit for thetaH25Alpha4 with bounds
params25Alpha4, covariance25Alpha4 = curve_fit(
    logistic_func, delta, thetaH25Alpha4, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
)
A25Alpha4, B25Alpha4, C25Alpha4, D25Alpha4 = params25Alpha4


# Set the style for a scientific look
# plt.style.use('default')
# plt.rcParams.update({
#     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
#     'font.family': 'sans-serif',
#     'mathtext.fontset': 'cm',
#     'axes.labelsize': 24,
#     'axes.titlesize': 26,
#     'xtick.labelsize': 18,
#     'ytick.labelsize': 18,
#     'legend.fontsize': 20,
#     'axes.linewidth': 1.5,
#     'grid.linewidth': 0.5,
# })

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot lines and points
line25 = ax.plot(delta, thetaH25, '-', color='black', linewidth=5, alpha=0.7, label=r'No Artificial Bias')[0]
ax.plot(delta, thetaH25, 'o', color='black', markersize=8, alpha=0.7)

line24Alpha1 = ax.plot(delta, thetaH25Alpha1, '-', color='blue', linewidth=5, alpha=0.7, label=r'$\alpha_1^{(t_{change})}=1$')[0]
ax.plot(delta, thetaH25Alpha1, 'o', color='blue', markersize=8, alpha=0.7)

line24Alpha4 = ax.plot(delta, thetaH25Alpha4, '-', color='red', linewidth=5, alpha=0.7, label=r'$\alpha_4^{(t_{change})}=1$')[0]
ax.plot(delta, thetaH25Alpha4, 'o', color='red', markersize=8, alpha=0.7)

# Add error bars separately
err25 = ax.errorbar(delta, thetaH25, yerr=[thetaH25-lower25, np.abs(upper25-thetaH25)], fmt='none',
                    ecolor='black', elinewidth=2, capsize=4, alpha=0.7)
err25Alpha1 = ax.errorbar(delta, thetaH25Alpha1, yerr=[thetaH25Alpha1-lower25Alpha1, np.abs(upper25Alpha1-thetaH25Alpha1)], fmt='none',
                    ecolor='blue', elinewidth=2, capsize=4, alpha=0.7)

err25Alpha4 = ax.errorbar(delta, thetaH25Alpha4, yerr=[thetaH25Alpha4-lower25Alpha4, np.abs(upper25Alpha4-thetaH25Alpha4)], fmt='none',
                    ecolor='red', elinewidth=2, capsize=4, alpha=0.7)

# Customize the plot
ax.set_xlabel(r'$\Delta$',fontsize=26)
ax.set_ylabel(r'$\hat{\theta}_{H}$',fontsize=26)
ax.set_title(r'Hit Rates: Cue $S_1$ 100%, Change $S_4$',fontsize=30)

# Set x-axis and y-axis limits and ticks
ax.set_xlim(0, 45)
ax.set_ylim(0, 1)
# ax.set_xticks(np.arange(0, 1.1, 0.2))
# ax.xaxis.set_minor_locator(MultipleLocator(0.05))
# ax.yaxis.set_minor_locator(MultipleLocator(0.05))

# Increase tick width and length
ax.tick_params(axis='both', which='major', width=2, length=10, labelsize=22)
ax.tick_params(axis='both', which='minor', width=1.5, length=5)

# Add legend
ax.legend(loc='center right', frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray', fontsize=20)

# Add grid (major and minor)
ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)



# Adjust layout and save
plt.tight_layout()
plt.savefig('scientific_plot_updated.png', dpi=300, bbox_inches='tight')
plt.show()


# Create the plot
fig, ax = plt.subplots(figsize=(13, 8))

# Plot lines and points
ax.plot(delta, thetaH25, 'o', color='black', markersize=8, alpha=0.7)
ax.plot(delta, thetaH25Alpha4, 'o', color='red', markersize=8, alpha=0.7)
ax.plot(delta, thetaH25Alpha1, 'o', color='blue', markersize=8, alpha=0.7)



# Plot raw data points with error bars
err25 = ax.errorbar(delta, thetaH25, yerr=[thetaH25-lower25, np.abs(upper25-thetaH25)], fmt='none',
                    ecolor='black', elinewidth=2, capsize=4, alpha=0.7)
err25Alpha4 = ax.errorbar(delta, thetaH25Alpha4, yerr=[thetaH25Alpha4-lower25Alpha4, np.abs(upper25Alpha4-thetaH25Alpha4)], fmt='none',
                    ecolor='red', elinewidth=2, capsize=4, alpha=0.7)
err25Alpha1 = ax.errorbar(delta, thetaH25Alpha1, yerr=[thetaH25Alpha1-lower25Alpha1, np.abs(upper25Alpha1-thetaH25Alpha1)], fmt='none',
                    ecolor='blue', elinewidth=2, capsize=4, alpha=0.7)



delta_fine = np.linspace(delta.min(), delta.max(), 300)

# Generate fitted curves
fitted_curve25 = logistic_func(delta_fine, A25, B25, C25, D25)
fitted_curve25Alpha4 = logistic_func(delta_fine, A25Alpha4, B25Alpha4, C25Alpha4, D25Alpha4)
fitted_curve25Alpha1 = logistic_func(delta_fine, A25Alpha1, B25Alpha1, C25Alpha1, D25Alpha1)


# Plot fitted logistic curves
ax.plot(delta_fine, fitted_curve25, '-', color='black', linewidth=5, label='No Artificial Bias')
ax.plot(delta_fine, fitted_curve25Alpha4, '-', color='red', linewidth=5, label=r'Enhanced Attention')

ax.plot(delta_fine, fitted_curve25Alpha1, '-', color='blue', linewidth=5, label=r'Attenuated Attention')

# Customize the plot
# ax.set_xlabel(r'$\Delta$', fontsize=26)
ax.set_ylabel(r'Mean Response Percentages', fontsize=26)
ax.set_title(r'Cue $S_1$ 25%, Change $S_1$', fontsize=30)

# Set x-axis and y-axis limits and ticks
ax.set_xlim(0, 45)
ax.set_xticks(np.arange(0, 46, 5))
ax.set_xlim(delta.min(), delta.max())
ax.set_ylim(0, 1)
yticks = np.linspace(0, 1, 11)  # Ticks at 0.0, 0.1, ..., 1.0
ax.set_yticks(yticks)
ax.set_yticklabels(['{:.0f}%'.format(ytick * 100) for ytick in yticks])
# Increase tick width and length
# Increase tick width and length, and remove x-axis labels
ax.tick_params(axis='both', which='major', width=2, length=10, labelsize=22)
ax.tick_params(axis='both', which='minor', width=1.5, length=5)
ax.tick_params(axis='x', which='both', labelbottom=False)  # Add this line to remove x-axis labels


# Add legend
ax.legend(loc='center right', frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray', fontsize=18)

# Add grid (major and minor)
# ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
# ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('logistic_fit_RRplot_Cue25AttentionMods.pdf', dpi=300, bbox_inches='tight')
plt.show()

# After fitting your models, you can print the fitted parameters like this:

print(f"Parameters for No Artificial Bias:")
print(f"  A = {A25:.3f}")
print(f"  B = {B25:.3f}")
print(f"  C = {C25:.3f}")
print(f"  D = {D25:.3f}\n")
perr25 = np.sqrt(np.diag(covariance25))
print('perr25',perr25)

print(f"Parameters for α₁ at t_change=1:")
print(f"  A = {A25Alpha1:.3f}")
print(f"  B = {B25Alpha1:.3f}")
print(f"  C = {C25Alpha1:.3f}")
print(f"  D = {D25Alpha1:.3f}\n")
perr25Alpha1 = np.sqrt(np.diag(covariance25Alpha1))
print('perr25Alpha1',perr25Alpha1)

print(r"Parameters for $\alpha_4^{(t_{change})}=1$:")
print(f"  A = {A25Alpha4:.3f}")
print(f"  B = {B25Alpha4:.3f}")
print(f"  C = {C25Alpha4:.3f}")
print(f"  D = {D25Alpha4:.3f}")
perr25Alpha4 = np.sqrt(np.diag(covariance25Alpha4))
print('perr25Alpha4',perr25Alpha4)

# Collect the C values and their uncertainties
C_values = [C25, C25Alpha1, C25Alpha4]
C_errors = [perr25[2], perr25Alpha1[2], perr25Alpha4[2]]

# Labels for the bars
labels = ['No Artificial Bias', r'$\alpha_1^{(t_{change})}=1$', r'$\alpha_4^{(t_{change})}=1$']

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Positions of the bars on the x-axis
x_pos = np.arange(len(labels))

# Create the bars
bars = ax.bar(x_pos, C_values, yerr=C_errors, align='center', alpha=0.8, ecolor='black', capsize=10, color=['black', 'blue', 'red'])

# Add labels and title
ax.set_ylabel('C Value', fontsize=16)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=14, rotation=45, ha='right')
ax.set_title('Comparison of C Values from Logistic Fits', fontsize=18)

# Add numerical values above the bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{C_values[i]:.2f}', ha='center', va='bottom', fontsize=18)

# Add grid
ax.yaxis.grid(True)
ax.set_ylim([0, 0.5])

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('C_values_bar_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()