from torch.utils.data import Dataset
from kalman_filter import IEKF
from termcolor import cprint
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import random
import torch
import h5py
import time
import os


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'                   # Set the environement variable CUBLAS_WORKSPACE_CONFIG to ':4096:8' for the use of deterministic algorithms in convolution layers


# #### - Decorator - #### #
def timming(function):
    def wrapper(*args, **kwargs):
        start = time.time_ns()

        result = function(*args, **kwargs)

        end = time.time_ns()
        dt = end - start
        c = 0
        unit = ['ns', 'µs', 'ms', 's']
        while dt > 1000:
            dt = round(dt/1000, 3)
            c += 1
        cprint(f"Function: {function.__name__}; Execution time: {dt} {unit[c]}", 'grey')
        return result
    return wrapper


# #### - Functions - #### #
def normalize(u):
    std = u.std(0, unbiased=True)
    mean = u.mean(0)
    u_norm = (u-mean)/std
    return u_norm


def compute_delta_p(Rot, p):
    """
    Compute delta_p for Relative Position Error
    """
    list_rpe = [[], [], []]                                         # [idx_0, idx_end, pose_delta_p]

    Rot = Rot[::10]                                                 # sub-sample at 10 Hz
    Rot = Rot.cpu()
    p = p[::10]                                                     # sub-sample at 10 Hz
    p = p.cpu()

    step_size = 10                                                  # every second, 10 sub-samples = 1s
    distances = np.zeros(p.shape[0])
    # this must be ground truth
    dp = p[1:] - p[:-1]                                             # delta of position at each sub-sample
    distances[1:] = dp.norm(dim=1).cumsum(0).numpy()                # cumulative sum of delta position to get the total length

    seq_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    k_max = int(Rot.shape[0] / step_size) - 1

    for k in range(0, k_max):  # each k represent 1s pass in the dataset
        idx_0 = k * step_size  # index of the k^th second in the sub-sample sequence
        for seq_length in seq_lengths:  # run through [100, ..., 800]
            if seq_length + distances[idx_0] > distances[-1]:  # go to the next iteration if left less than 'seq_length' to the end of the sequence distance
                continue
            idx_shift = np.searchsorted(distances[idx_0:], distances[idx_0] + seq_length)  # get number of sample that represent a distance of 'seq_length'
            idx_end = idx_0 + idx_shift  # index of the end of 'seq_length' length from k^th seconde
            list_rpe[0].append(idx_0)
            list_rpe[1].append(idx_end)
        idxs_0 = list_rpe[0]  # store the indices of each start where left [100, ..., 800]m
        idxs_end = list_rpe[1]  # store the indices of each end where left [100, ..., 800]m
        delta_p = Rot[idxs_0].transpose(-1, -2).matmul(((p[idxs_end] - p[idxs_0]).float()).unsqueeze(-1)).squeeze()  # calcul the cartesian coordinates from the initial position for each set of [100, ..., 800]
        list_rpe[2] = delta_p.numpy()
    return list_rpe


def precompute_lost(Rot, p, list_rpe, N0):
    # compute the relative translational error
    N = p.shape[0]
    Rot_10_Hz = Rot[::10]
    p_10_Hz = p[::10]
    idxs_0 = torch.Tensor(np.array(list_rpe.loc[:, 'idx_0'].values, dtype=float)).long() - int(N0 / 10)
    idxs_end = torch.Tensor(np.array(list_rpe.loc[:, 'idx_end'].values, dtype=float)).long() - int(N0 / 10)
    delta_p_gt = torch.Tensor(np.vstack(list_rpe.loc[:, 'pose_delta_p'].values))
    idxs = torch.Tensor(idxs_0.shape[0]).bool()
    idxs[:] = 1
    idxs[idxs_0 < 0] = 0
    idxs[idxs_end >= int(N / 10)] = 0
    delta_p_gt = delta_p_gt[idxs]
    idxs_end_bis = idxs_end[idxs]
    idxs_0_bis = idxs_0[idxs]
    if len(idxs_0_bis) == 0:
        return None, None
    else:
        delta_p = Rot_10_Hz[idxs_0_bis].transpose(-1, -2).matmul((p_10_Hz[idxs_end_bis] - p_10_Hz[idxs_0_bis]).unsqueeze(-1)).squeeze()
        distance = delta_p_gt.norm(dim=1).unsqueeze(-1)
        iekf_pre_loss = delta_p.double() / distance.double()
        gt_pre_loss = delta_p_gt.double() / distance.double()
        return iekf_pre_loss.to(DEVICE), gt_pre_loss.to(DEVICE)


# #### - Class - #### #
class KittiDataset(Dataset):
    def __init__(self, hdf_path, seq_length, split):
        self.hdf_path = hdf_path
        self.seq_length = seq_length
        self.hdf = h5py.File(self.hdf_path, 'r')                    # Read the .h5 file
        self.split = split

        self.init = pd.DataFrame()
        self.time = pd.DataFrame()
        self.U = pd.DataFrame()
        self.GT = pd.DataFrame()

        self.h5_to_df()

    def __getitem__(self, item):
        v_mes0 = self.init.loc[["ve", "vn", "vu"], item].values
        ang0 = self.init.loc[["roll", "pitch", "yaw"], item].values
        time = self.time.loc[:, item]
        time_t = time[~time['time'].isna()].to_numpy()
        U = self.U.loc[:, item]
        U_t = torch.tensor(U[~U['ax'].isna()].to_numpy(), dtype=torch.float32)
        GT = self.GT.loc[:, item]
        GT_t = torch.tensor(GT[~GT['x'].isna()].to_numpy(), dtype=torch.float32)
        return (v_mes0, ang0), time_t, U_t, GT_t

    def __len__(self):
        return len(list(self.hdf.get(self.split)))

    def __iter__(self):
        seqs = self.hdf.get(self.split)
        for i in list(seqs.keys()):
            print(i)
            yield i, self[i]

    def h5_to_df(self):
        seqs = self.hdf.get(self.split)
        # print(f"Train:\n\t{list(seqs.keys())}")
        dict_u = {}
        dict_gt = {}
        dict_time = {}
        dict_init = {}
        for i in list(seqs.keys()):
            # get initial condictions
            init_df = pd.read_hdf(self.hdf_path, f"{self.split}/{i}/dataset").loc[:, ["ve", "vn", "vu", "roll", "pitch", "yaw"]].iloc[0, :]
            dict_init[f"{i}"] = pd.Series(init_df)  # Add the DataFrame to the dictionary

            # get time vector
            time_df = pd.read_hdf(self.hdf_path, f"{self.split}/{i}/time")
            dict_time[f"{i}"] = pd.DataFrame(time_df)  # Add the DataFrame to the dictionary

            # get IMU measurment + time
            u_df = pd.read_hdf(self.hdf_path, f"{self.split}/{i}/w_a_input")  # Get the input DataFrame for the given date and drive
            dict_u[f"{i}"] = pd.DataFrame(u_df)  # Add the DataFrame to the dictionary

            # get Ground_Truth
            gt_df = pd.read_hdf(self.hdf_path, f"{self.split}/{i}/ground_truth")  # Get the input DataFrame for the given date and drive
            dict_gt[f"{i}"] = pd.DataFrame(gt_df)  # Add the DataFrame to the dictionary

        self.init = pd.concat(dict_init, axis=1)  # Create DataFrame from dictionary, with index as dict keys
        self.time = pd.concat(dict_time, axis=1)  # Create DataFrame from dictionary, with index as dict keys
        self.U = pd.concat(dict_u, axis=1)  # Create DataFrame from dictionary, with index as dict keys
        self.GT = pd.concat(dict_gt, axis=1)  # Create DataFrame from dictionary, with index as dict keys


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, yhat, y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(yhat, y) + 1e-6)
        return loss


class CNN(torch.nn.Module):
    @timming
    def __init__(self, seq_length):
        super(CNN, self).__init__()

        self.layers = torch.nn.Sequential(
            # Convolutional part
            torch.nn.Conv1d(in_channels=6,                          # Applies a 1D convolution over an input signal composed of several input planes
                            out_channels=32,
                            kernel_size=5),
            torch.nn.ReplicationPad1d(4),                           # Pads the input tensor using replication of the input boundary
            torch.nn.ReLU(),                                        # Applies the rectified linear unit function element-wise: ReLU(x) = (x)+ = max(0, x)
            torch.nn.Dropout(p=0.5),                                # Randomly zeroes some of th eelements of the input tensor with probability p, using samples from a Bernoulli distribution

            torch.nn.Conv1d(in_channels=32,
                            out_channels=32,
                            kernel_size=5,
                            dilation=3),                            # Control spacing between the kernel points, c.f.: www.github.com/vdmoulin/conv_arithmetic
            torch.nn.ReplicationPad1d(4),                           # Pads the input tensor using replication of the input boundary
            torch.nn.ReLU(),                                        # Applies the rectified linear unit function element-wise: ReLU(x) = (x)+ = max(0, x)
            torch.nn.Dropout(p=0.5),                                # Randomly zeroes some of th eelements of the input tensor with probability p, using samples from a Bernoulli distribution

            # Fully connected part
            torch.nn.Flatten(),                                     # Flattens a contigous range of dims into a tensor
            torch.nn.Linear(32, 2),                                 # Applies a linear transformation to the incoming data: y = xA.T + b
            torch.nn.Tanh()                                         # Applies the Hyperbolic Tangent (Tanh) function element-wise: Tanh(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        )

        # torch.nn.init.kaiming_normal_(self.layers[0].weight)
        # torch.nn.init.kaiming_normal_(self.layers[4].weight)
        # torch.nn.init.kaiming_normal_(self.layers[9].weight)

    @timming
    def forward(self, x):
        out = self.layers(x)
        return out


def make_trainning(model, EPOCHS):
    # #### - Kalman Filter - #### #
    iekf = IEKF()

    # #### - Train - #### #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_bar = tqdm(total=len(train), unit="batch", desc="Training", leave=False)
    epoch_bar = tqdm(total=EPOCHS, unit="epoch", desc="Training")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        cprint("Générer les sous-séquences aléatoires", 'red')
        train_running_loss = 0.

        model.train()                                               # Make sure gradient tracking is on, and do a pass over the data

        batch_bar.reset()

        for batch_index, (seq_name, (init_cond, t, inputs, ground_truth)) in enumerate(train):
            print(batch_index, seq_name)
            print(init_cond, t.shape, inputs.shape, ground_truth.shape)
            inputs_net, ground_truth = inputs.to(DEVICE), ground_truth.to(DEVICE)

            input_net_norm = normalize(inputs_net)                  # Normalize inputs
            z_cov_net = model.forward(inputs_net)
            z_cov = z_cov_net.to('cpu')                             # Move the CNN result to 'cpu' for Kalman Filter iterations
            Rot, p = iekf.train_run(time, inputs, z_cov, init_cond[0], init_cond[1])  # Run the training Kalman Filter

            loss = criterion(iekf_out, ground_truth)  # compute loss
            loss.backward()                                         # Calculate gradients
            optimizer.step()                                        # Adjust learning weights

            train_running_loss += loss.item()

            batch_bar.set_postfix(train_loss=train_running_loss/(batch_index+1), lr=optimizer.param_groups[0]['lr'])
            batch_bar.update()
        train_loss = train_running_loss / batch_index
        train_loss_history.append(train_loss)

        # #### - Validation - #### #
        val_running_loss = 0.
        model.eval()

        with torch.no_grad():
            for batch_index, (inputs, ground_truth) in enumerate(validation):
                inputs, ground_truth = inputs.to(DEVICE), ground_truth.to(DEVICE)

                # Ajouter la normalisation de l'input
                z_cov = model.forward(inputs)
                iekf_out = []  # run l'IEKF

                loss = criterion(iekf_out, ground_truth)            # compute loss

                val_running_loss += loss.item()
            val_loss = val_running_loss / batch_index
            val_loss_history.append(val_loss)

        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=optimizer.param_groups[0]['lr'])
        epoch_bar.update()
    return train_loss_history, val_loss_history


# #### - Main - #### #
if __name__ == '__main__':
    start_time = time.time()

    random_seed = 34                                                # set random seed
    torch.backends.cudnn.enable = False                             # Disable cuDNN use of nondeterministic algorithms
    torch.use_deterministic_algorithms(True)                        # Configure PyTorch to use deterministic algorithms instead of nondeterministic ones where available, and throw an error if an operation is known to be nondeterministic (and without a deterministic alternative)
    torch.manual_seed(random_seed)                                  # Set the seed to the Random Number Generator (RNG) for all devices (both CPU and CUDA)
    rng = np.random.default_rng(random_seed)                        # Create a RNG with a fixed seed
    random.seed(random_seed)                                        # Set the Python seed

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--device", type=str, required=True, help="Which GPU to use (cuda or cpu). Ex: --device 'cuda0'"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, required=True, help="Number of epochs to train the model. Ex: --epochs 400"
    )
    # parser.add_argument(
    #     "-b", "--batch", type=int, required=True, help="Batch size for training. Ex: --batch 64"
    # )
    parser.add_argument(
        "-s", "--seq", type=int, required=True, help="Length sequence of data. Ex: --seq 2000"
    )

    args = parser.parse_args()

    EPOCHS = args.epochs
    DEVICE = args.device
    SEQ_LEN = args.seq

    print(f"Epochs: {EPOCHS}; Device: {DEVICE}; Sequence length: {SEQ_LEN}")

    save_path = "../data/processed/dataset.h5"                      # Path to the .h5 dataset

    train = KittiDataset(save_path, SEQ_LEN, 'train')
    validation = KittiDataset(save_path, SEQ_LEN, 'validation')

    _, _, u, _ = train['day_2011_10_03_drive_0042_extract']
    normalize(u)

    # Model
    model = CNN(SEQ_LEN).to(DEVICE)
    # Loss
    criterion = torch.nn.MSELoss()

    train_loss_history, val_loss_history = make_trainning(model, EPOCHS)

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

    torch.cuda.empty_cache()
