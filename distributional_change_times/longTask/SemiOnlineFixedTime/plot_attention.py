import numpy as np
import matplotlib.pyplot as plt
import os

# Path to your .npy file — adjust if it's elsewhere
data_path = "A1_Data.npy"

# Load the data (shape: 200×30×4×4)
x = np.load(data_path)

# Compute mean over the first axis → shape becomes (30, 4, 4)
mean_x = x.mean(axis=0)

# Create output directory for heatmaps
output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)

# Generate and save each heatmap with colorbar fixed from 0 to 1
for i in range(mean_x.shape[0]):
    plt.figure()
    plt.imshow(mean_x[i], vmin=0, vmax=1, aspect='equal')
    cbar = plt.colorbar()
    cbar.set_label('Value (0–1)')
    plt.title(f"Heatmap {i}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_{i}.png"))
    plt.close()

print(f"Saved {mean_x.shape[0]} heatmaps to '{output_dir}/heatmap_0.png' through '{output_dir}/heatmap_{mean_x.shape[0]-1}.png'")
