import numpy as np
import torch
from VAENet import VAE  # Import the VAE class from model definition file
from VAETrainer import VAERunner  # Import the VAERunner class
import h5py

def load_model(model_path, model_class=VAE):
    # Initialize the model and load the state dictionary
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Open the HDF5 file in read mode
with h5py.File('BigData.h5', 'r') as h5f:
    # Load the dataset
    noisy_train_data = h5f['noise=PixelSwap/ChangeDegree=(0,35)'][:]
    clean_train_data = h5f['noise=0/ChangeDegree=(0,35)'][:]

    # Now 'data' contains the loaded dataset and can be used for further processing

print(noisy_train_data.shape)
data = noisy_train_data.reshape((-1,25,25,1))
# print(data[17])

noisy_train_data = noisy_train_data.reshape((-1,25,25,1))
clean_train_data = clean_train_data.reshape((-1,25,25,1))

# Initialize the VAE model
model = VAE()

# Check if CUDA is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('vae_model.pth')
model.to(device)


# Initialize the VAERunner with the model and data
trainer = VAERunner(model, noisy_train_data, clean_train_data, batch_size=100, learning_rate=1e-4)

# Train the model
trainer.train(epochs=100)

