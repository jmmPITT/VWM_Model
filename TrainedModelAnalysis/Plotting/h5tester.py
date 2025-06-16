import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_attention_titrations(h5_filename: str, group_name: str = "Cue S1 100%"):
    # Open the HDF5 file
    with h5py.File(h5_filename, "r") as f:
        grp = f[group_name]
        delta     = grp["Delta"][()]      # raw Δ values
        delta_f   = grp["DeltaFine"][()]  # fine Δ grid for fitted curves

        # collect and sort all alpha subgroups by their numeric value
        alpha_names = sorted(
            [n for n in grp if n.startswith("alpha1=")],
            key=lambda s: float(s.split("=")[1])
        )

        # choose a colormap with as many discrete colors as alphas
        cmap = mpl.cm.get_cmap("viridis", len(alpha_names))

        fig, ax = plt.subplots(figsize=(12, 7))

        for idx, name in enumerate(alpha_names):
            ds    = grp[name]
            theta = ds["Theta"][()]
            low   = ds["LowerCI"][()]
            high  = ds["UpperCI"][()]
            fit   = ds["FittedCurve"][()]

            color = cmap(idx)
            α = float(name.split("=")[1])

            # plot raw data with error bars
            ax.errorbar(
                delta, theta,
                yerr=[np.abs(theta - low), np.abs(high - theta)],
                fmt="o", color=color, capsize=4, alpha=0.7,
                label=rf"$\alpha_1={α:.1f}$"
            )

            # plot precomputed fitted curve
            ax.plot(delta_f, fit, "-", color=color, linewidth=2)

        # formatting
        ax.set_xlabel(r"$\Delta$", fontsize=18)
        ax.set_ylabel("Mean Response (hit rate)", fontsize=18)
        ax.set_title("Cue S1 25%: Hit‐Rate vs. Δ for Various α₁", fontsize=20)
        ax.set_xlim(delta.min(), delta.max())
        ax.set_ylim(0, 1.0)
        ax.set_xticks(np.arange(0, 46, 5))
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend(
            title="α₁ Values", title_fontsize=14,
            fontsize=12, bbox_to_anchor=(1.04, 1), loc="upper left",
            frameon=True, fancybox=True, framealpha=0.8
        )
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    plot_attention_titrations("AttentionTitrations.h5")
