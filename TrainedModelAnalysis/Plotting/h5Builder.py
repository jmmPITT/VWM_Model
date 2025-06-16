import numpy as np
import h5py
from scipy import stats
from scipy.optimize import curve_fit


# Logistic function for fitting
def logistic_func(x, A, B, C, D):
    return A + (1 - B) / (1 + np.exp(-C * (x - D)))


# Jeffreys interval for binomial proportions
def jeffreys_interval(successes, trials, confidence=0.95):
    α = 0.5 + successes
    β = 0.5 + trials - successes
    lower = stats.beta.ppf((1 - confidence) / 2, α, β)
    upper = stats.beta.ppf(1 - (1 - confidence) / 2, α, β)
    return lower, upper


# Delta values for the raw data
delta = np.linspace(0, 45, 35)
# Fine grid for plotting/evaluating the fitted curve
delta_fine = np.linspace(delta.min(), delta.max(), 300)

# Initial guesses and bounds for [A, B, C, D]
initial_guess = [0.0, 0.5, 1.0, np.median(delta)]
lower_bounds = [-np.inf, 0.0, -np.inf, -np.inf]
upper_bounds = [np.inf, 1.0, np.inf, np.inf]

# Alpha titrations for Cue S1 25%
alpha_values = [i / 10 for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]
suffixes = [f"{i:02d}" for i in range(1, 11)]  # ["01", "02", ..., "10"]

# Create the HDF5 file
with h5py.File("AttentionTitrations.h5", "a") as h5f:
    grp_root = h5f.create_group("Cue S1 100%")
    # save the delta vectors once
    grp_root.create_dataset("Delta", data=delta)
    grp_root.create_dataset("DeltaFine", data=delta_fine)

    for alpha, suf in zip(alpha_values, suffixes):
        # load raw .npy
        filename = f"ChoiceStats_CueS1100_ChangeS1_alpha1TchangeEqual{suf}_DeltaModulation.npy"
        data = np.load(filename)  # shape (35, N)
        hits = data[:, 1]  # column 1 = hits
        trials = 500

        # compute proportions and credible intervals
        theta = hits / trials
        lower_ci, upper_ci = jeffreys_interval(hits, trials)

        # fit logistic curve
        params, cov = curve_fit(
            logistic_func, delta, theta,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=50000
        )
        param_se = np.sqrt(np.diag(cov))  # standard errors
        fitted_curve = logistic_func(delta_fine, *params)

        # create subgroup and save everything
        sub = grp_root.create_group(f"alpha1={alpha:.1f}")
        sub.create_dataset("Hits", data=hits)
        sub.create_dataset("Theta", data=theta)
        sub.create_dataset("LowerCI", data=lower_ci)
        sub.create_dataset("UpperCI", data=upper_ci)
        sub.create_dataset("FitParams", data=params)  # [A, B, C, D]
        sub.create_dataset("ParamSE", data=param_se)  # st. dev. of fit params
        sub.create_dataset("FittedCurve", data=fitted_curve)

print("AttentionTitrations.h5 written with raw data, CI, fit params, and fitted curves.")
