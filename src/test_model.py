from train_CNN import KittiDataset, CNN
import matplotlib.pyplot as plt
from kalman_filter import IEKF
from termcolor import cprint
import pandas as pd
import numpy as np
import argparse
import random
import torch
import utils
import h5py
import time
import os


def forward_traj(model, drive):
    # #### - Kalman Filter - #### #
    iekf = IEKF()

    with torch.no_grad():
        print(f"Drive: {drive}")
        # Load dataframes
        dataset = pd.DataFrame(pd.read_hdf(save_path, f"full_datset/{drive}"))  # Get the input DataFrame for the given date and drive
        u_df = dataset[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].copy()
        time_df = dataset[['time']].copy()

        # Export the values to an np.array
        u = u_df.values
        t = time_df.values
        v_mes0 = dataset[["ve", "vn", "vu"]].copy().iloc[0, :].values
        ang0 = dataset[["roll", "pitch", "yaw"]].copy().iloc[0, :].values

        u_net = torch.Tensor(u).to(DEVICE)

        print(f"   Forward...")
        z_cov_net = model.forward(u_net)
        z_cov = z_cov_net.cpu()                                         # Move the CNN result to 'cpu' for Kalman Filter iterations
        print(f"   IEKF...")
        kalman = iekf.run(t, u, z_cov, v_mes0, ang0)  # Run Kalman Filter
        print(f"End run drive: {drive}")
    return kalman


def test_filter(drives="full"):
    Drives = [k for k in test.seq_name()] if drives == "full" else [drives]

    model.eval()
    for drive in Drives:
        kalman = forward_traj(model, drive)
        print(f"Drive: {drive}\n\tshapes: rot [{kalman['rot'].shape}], p [{kalman['p'].shape}], P [{kalman['P'].shape}]")


# #### - Main - #### #
if __name__ == '__main__':
    start_time = time.time()

    save_path = "../data/processed/dataset.h5"                                      # Path to the .h5 dataset

    DEVICE = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = KittiDataset(save_path, 'test')

    model1 = CNN(3000)
    model1.load_state_dict(torch.load(f"../models/20220818/CNN_S3000_B4_LR0.001_E100_RTrue.pt"))
    model2 = CNN(3000)
    model2.load_state_dict(torch.load(f"../models/20220818/CNN_S3000_B4_LR0.001_E100_RFalse.pt"))

    models = [model1, model2]

    # test_filter(drives="day_2011_09_30_drive_0027_extract")
    kalmans_evo = []
    for i in range(len(models)):
        kalman = forward_traj(models[i], drive="day_2011_09_30_drive_0028_extract")
        kalmans_evo.append(utils.df_to_PosePath3D(kalman['rot'], kalman['p']))

    dataset = pd.read_hdf(save_path, f"full_datset/day_2011_09_30_drive_0028_extract")
    ground_evo = utils.df_to_PosePath3D(dataset['rot_matrix'].values, dataset[['pose_x', 'pose_y', 'pose_z']].values)

    print(f"Kalman trajectory 1:")
    print(f"APE: {utils.get_APE(ground_evo, kalmans_evo[0]).get_all_statistics()}")
    print(f"RPE: {utils.get_RPE(ground_evo, kalmans_evo[0]).get_all_statistics()}")
    utils.plot_APE(ground_evo, kalmans_evo[0])

    print(f"Kalman trajectory 2:")
    print(f"APE: {utils.get_APE(ground_evo, kalmans_evo[1]).get_all_statistics()}")
    print(f"RPE: {utils.get_RPE(ground_evo, kalmans_evo[1]).get_all_statistics()}")
    utils.plot_APE(ground_evo, kalmans_evo[1])

    utils.plot_multiple(ground_evo, kalmans_evo)

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

    plt.show()
