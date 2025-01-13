########################################
# main.py
########################################

import torch
from trainer import Trainer  # Your training class on the TRAIN data
from tester import Tester  # A separate testing/evaluation class on the TEST data
from Network import TransformerNetwork
from GenData import generate_trial_data  # Or wherever your data-generation function lives


def main():
    # -----------------------------
    # HYPERPARAMETERS & PATHS
    # -----------------------------
    learning_rate = 1e-6
    num_epochs_train = 200
    batch_size = 128

    # Filenames for the generated data
    train_obs_file = "trial_observations_train.npy"
    train_label_file = "trial_labels_train.npy"
    test_obs_file = "trial_observations_test.npy"
    test_label_file = "trial_labels_test.npy"

    # Number of times you want to repeat the cycle
    num_cycles = 100

    ### Initial Delta ###
    Delta_max = 65
    # -----------------------------
    # REPEATED TRAIN/TEST CYCLES
    # -----------------------------
    for cycle_index in range(num_cycles):
        print(f"===== CYCLE {cycle_index + 1} of {num_cycles} =====")

        # ----------------------------------------------------
        # 1) Generate the TRAIN data
        # ----------------------------------------------------
        print("Generating training data...")
        generate_trial_data(
            Delta_max=Delta_max,  # or another value
            model_path="vae_model.pth",  # your VAE model
            n_games=5000,  # how many episodes to generate
            T_end=7,
            save_labels=train_label_file,
            save_observations=train_obs_file
        )

        # ----------------------------------------------------
        # 2) Generate the TEST data
        # ----------------------------------------------------
        print("Generating test data...")
        generate_trial_data(
            Delta_max=Delta_max,  # or maybe a different ∆ for test
            model_path="vae_model.pth",
            n_games=5000,
            T_end=7,
            save_labels=test_label_file,
            save_observations=test_obs_file
        )

        # ----------------------------------------------------
        # 3) Instantiate Model & Trainer for TRAINING
        # ----------------------------------------------------
        print("Instantiating model & trainer...")
        model = TransformerNetwork(beta=learning_rate)

        # If you want to load a previous checkpoint each cycle, do so here:
        if cycle_index > 0:
            checkpoint = torch.load("model_save.pth", map_location=model.device)
            model.load_state_dict(checkpoint['model_state_dict'])

        trainer = Trainer(
            model=model,
            data_path=train_obs_file,
            labels_path=train_label_file,
            batch_size=batch_size,
            num_epochs=num_epochs_train,
            shuffle=True
        )

        print("Training...")
        trainer.train()

        # If you want to save a checkpoint after training
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict()
        }, f"trained_model_cycle.pth")

        # ----------------------------------------------------
        # 4) Evaluate the trained model on TEST data
        # ----------------------------------------------------
        # For testing, we can use a separate `Tester` class or a short function in `trainer.py`
        print("Testing...")
        tester = Tester(
            model=model,
            data_path=test_obs_file,
            labels_path=test_label_file,
            batch_size=batch_size,
            num_epochs=1,  # or however many passes you want just for eval
            shuffle=False
        )
        acc = tester.test()  # or tester.train() if you want the same loop, but presumably without backprop
        if acc > 85:
            Delta_max -= 3
            print('NEXT STAGE')
            print('Delta_max = ', Delta_max)
        print(f"===== END OF CYCLE {cycle_index + 1} =====\n")


if __name__ == "__main__":
    main()
