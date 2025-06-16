# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats # Keep stats for jeffreys_interval if needed elsewhere, though not directly used for parameter plots

# Define the logistic function (needed for curve_fit, though not plotted directly here)
def logistic_func(x, A, B, C, D):
    """
    Defines a 4-parameter logistic function.
    A: Lower asymptote
    B: Related to the upper asymptote. The range is (1-B). Upper asymptote = A + (1-B).
    C: Growth rate (steepness)
    D: Inflection point (midpoint, corresponds to delta)
    """
    return A + (1 - B) / (1 + np.exp(-C * (x - D)))

# --- Data Loading ---
# It's assumed the .npy files are in the same directory or path is specified
# Wrap in try-except blocks for robustness if files might be missing
try:
    alpha10 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual10_DeltaModulation.npy")
    alpha09 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual09_DeltaModulation.npy")
    alpha08 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual08_DeltaModulation.npy")
    alpha07 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual07_DeltaModulation.npy")
    alpha06 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual06_DeltaModulation.npy")
    alpha05 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual05_DeltaModulation.npy")
    alpha04 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual04_DeltaModulation.npy")
    alpha03 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual03_DeltaModulation.npy")
    alpha02 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual02_DeltaModulation.npy")
    alpha01 = np.load("ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual01_DeltaModulation.npy")
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    print("Please ensure all .npy files are present in the correct directory.")
    # Exit or handle error appropriately
    exit() # Or raise an exception

# --- Data Processing and Curve Fitting ---
datasets = {
    # Store data associated with each alpha_1 value
    1.0: alpha10, 0.9: alpha09, 0.8: alpha08, 0.7: alpha07, 0.6: alpha06,
    0.5: alpha05, 0.4: alpha04, 0.3: alpha03, 0.2: alpha02, 0.1: alpha01
}

# Define the delta values (x-axis for logistic fit)
# Ensure this matches the structure of your loaded .npy files
delta = np.linspace(0, 45, 35) # Example: 35 points from 0 to 45

# Store fitted parameters and their standard errors
alpha_values = sorted(datasets.keys(), reverse=True) # Plot from alpha=1.0 down to 0.1
params_dict = {}
std_errs_dict = {}

# Initial parameter guesses and bounds (same as original script)
initial_guess = [0.0, 0.5, 0.1, np.median(delta)] # Initial guess for A, B, C, D
# Define lower and upper bounds for parameters A, B, C, D
lower_bounds = [-np.inf, -np.inf, 1e-9, -np.inf] # C (steepness) should ideally be positive
upper_bounds = [np.inf, 1.0, np.inf, np.inf]    # Upper bound for B is 1.0 as per original interpretation

print("Fitting logistic curves for each alpha value...")
for alpha_val in alpha_values:
    data = datasets[alpha_val]
    # Ensure data shape is correct (e.g., N x 2 where N is number of delta points)
    if data.shape[1] < 2:
         print(f"  Warning: Data for alpha={alpha_val:.1f} has unexpected shape {data.shape}. Skipping.")
         params_dict[alpha_val] = [np.nan] * 4
         std_errs_dict[alpha_val] = [np.nan] * 4
         continue

    hits, trials = data[:, 1], data[:, 1] + data[:, 0]

    # Check for valid trials before division
    if np.any(trials <= 0):
        print(f"  Warning: Zero or negative trials found for alpha={alpha_val:.1f}. Handling potential division by zero.")
        # Handle safely: set thetaH to 0 or NaN where trials <= 0
        thetaH = np.zeros_like(hits, dtype=float)
        valid_trials_mask = trials > 0
        thetaH[valid_trials_mask] = hits[valid_trials_mask] / trials[valid_trials_mask]
        # Optionally remove points with invalid trials if curve_fit struggles
        # delta_valid = delta[valid_trials_mask]
        # thetaH_valid = thetaH[valid_trials_mask]
        # if len(delta_valid) == 0: continue # Skip if no valid points
    else:
        thetaH = hits / trials

    # Ensure delta and thetaH have the same length after potential filtering
    if len(delta) != len(thetaH):
         print(f"  Warning: Mismatch in length between delta ({len(delta)}) and thetaH ({len(thetaH)}) for alpha={alpha_val:.1f}. Skipping.")
         params_dict[alpha_val] = [np.nan] * 4
         std_errs_dict[alpha_val] = [np.nan] * 4
         continue

    try:
        # Perform the curve fit for the current alpha value
        params, covariance = curve_fit(
            logistic_func, delta, thetaH,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000 # Increased max iterations
        )
        # Check if covariance calculation was successful
        if np.isinf(covariance).any() or np.isnan(covariance).any():
            print(f"  Warning: Covariance matrix contains NaN or Inf for alpha={alpha_val:.1f}. Setting errors to NaN.")
            std_errs = [np.nan] * 4
        else:
            # Calculate standard errors from the covariance matrix diagonal
            std_errs = np.sqrt(np.diag(covariance))

        # Store the fitted parameters and errors
        params_dict[alpha_val] = params
        std_errs_dict[alpha_val] = std_errs
        # print(f"  Alpha={alpha_val:.1f}: Params={params}, StdErrs={std_errs}") # Optional: print fits

    except RuntimeError as e:
        print(f"  Warning: Curve fit failed for alpha={alpha_val:.1f}. Skipping. Error: {e}")
        params_dict[alpha_val] = [np.nan] * 4 # Store NaNs if fit fails
        std_errs_dict[alpha_val] = [np.nan] * 4
    except ValueError as e:
         print(f"  Warning: Value error during fit for alpha={alpha_val:.1f}. Check bounds or data. Error: {e}")
         params_dict[alpha_val] = [np.nan] * 4 # Store NaNs
         std_errs_dict[alpha_val] = [np.nan] * 4


print("Curve fitting complete.")

# --- Parameter Extraction for Plotting ---
# Ensure alpha_values matches the keys used for storing results
valid_alphas = [av for av in alpha_values if not np.isnan(params_dict.get(av, [np.nan]*4)[0])]
if len(valid_alphas) != len(alpha_values):
    print("Warning: Some alpha values were skipped due to fitting errors. Plots will only show successful fits.")

# Extract each parameter type into its own list only for successful fits
param_A = [params_dict[alpha][0] for alpha in valid_alphas]
param_B = [params_dict[alpha][1] for alpha in valid_alphas]
param_C = [params_dict[alpha][2] for alpha in valid_alphas]
param_D = [params_dict[alpha][3] for alpha in valid_alphas]
print('param_D',param_D)

# Extract standard errors for each parameter for successful fits
err_A = [std_errs_dict[alpha][0] for alpha in valid_alphas]
err_B = [std_errs_dict[alpha][1] for alpha in valid_alphas]
err_C = [std_errs_dict[alpha][2] for alpha in valid_alphas]
err_D = [std_errs_dict[alpha][3] for alpha in valid_alphas]
print('err_D',err_D)
print('valid alphas', valid_alphas)
# --- Plotting ---
print("Generating parameter plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 13)) # Create a 2x2 grid of subplots
fig.suptitle(r'Logistic Fit Parameters vs. $\alpha_1$ (Cue 25% $S_1$)', fontsize=22, y=0.99) # Adjusted y slightly

# --- Add the equation text ---
equation = r'$y = A + \frac{1 - B}{1 + e^{-C(\Delta - D)}}$'
# Place the equation below the main title
fig.text(0.5, 0.90, equation, ha='center', va='bottom', fontsize=16)


# Define parameters, errors, titles, and y-labels for looping
param_data = [
    (param_A, err_A, 'Parameter A (Lower Asymptote)', 'Estimate of A'),
    (param_B, err_B, 'Parameter B (Upper Limit Factor)', 'Estimate of B'), # Clarified title for B
    (param_C, err_C, 'Parameter C (Steepness)', 'Estimate of C'),
    (param_D, err_D, 'Parameter D (Inflection Point / $\Delta$ shift)', 'Estimate of D')
]

# Colors for each parameter plot (distinct colors)
plot_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Loop through parameters and create plots
for i, (param_values, param_errors, title, ylabel) in enumerate(param_data):
    ax = axes.flat[i] # Get the current subplot axis
    color = plot_colors[i]

    # Convert lists to numpy arrays for safe nan handling in errorbar
    alpha_plot_values = np.array(valid_alphas)
    param_plot_values = np.array(param_values)
    param_plot_errors = np.array(param_errors)

    # Plot parameter estimates with error bars
    ax.errorbar(alpha_plot_values, param_plot_values, yerr=param_plot_errors, fmt='-o', color=color,
                markersize=7, capsize=5, linewidth=2.5, elinewidth=2, label=f'Parameter {chr(65+i)}', alpha=0.8) # A, B, C, D

    # Customize the subplot
    ax.set_xlabel(r'$\alpha_1$ Value', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, pad=10) # Add padding to title
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)

    # Set x-axis limits and ticks
    ax.set_xlim(0.05, 1.05) # Range from 0.05 to 1.05
    ax.set_xticks(np.linspace(0.1, 1.0, 10)) # Ticks for each alpha value

    # Adjust y-axis limits dynamically based on data, adding some padding
    # Filter out NaN values before calculating min/max
    valid_indices = ~np.isnan(param_plot_values) & ~np.isnan(param_plot_errors)
    if np.any(valid_indices):
        min_val = np.nanmin(param_plot_values[valid_indices] - param_plot_errors[valid_indices])
        max_val = np.nanmax(param_plot_values[valid_indices] + param_plot_errors[valid_indices])
        if min_val is not None and max_val is not None and not np.isnan(min_val) and not np.isnan(max_val): # Check if min/max are valid numbers
            padding = (max_val - min_val) * 0.15 # 15% padding
            ax.set_ylim(min_val - padding, max_val + padding)
        else:
             print(f"  Info: Could not determine y-limits dynamically for {title} due to only NaN values or calculation issues.")
    else:
      print(f"  Warning: No valid data points to determine y-limits for {title}.")


    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=4)

# Adjust layout to prevent overlapping titles/labels and make space for equation/title
plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect: [left, bottom, right, top]

# Save the figure
plt.savefig('logistic_parameter_trends_vs_alpha_with_equation.pdf', dpi=300, bbox_inches='tight')
print("Parameter plots saved to 'logistic_parameter_trends_vs_alpha_with_equation.pdf'")

# Display the plot
plt.show()

print("Script finished.")
