from torch.utils.tensorboard import SummaryWriter
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


# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'                   # Set the environement variable CUBLAS_WORKSPACE_CONFIG to ':4096:8' for the use of deterministic algorithms in convolution layers


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
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

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
    def __init__(self, hdf_path, seq_length, split, rng):
        self.rng = rng                                              # Set a Random Number Generator

        self.hdf_path = hdf_path
        self.seq_length = seq_length
        self.hdf = h5py.File(self.hdf_path, 'r')                    # Read the .h5 file
        self.split = split

        self.init = pd.DataFrame()
        self.time = pd.DataFrame()
        self.U = pd.DataFrame()
        self.GT = pd.DataFrame()
        self.list_RPE = pd.DataFrame()

        self.h5_to_df()

    def __getitem__(self, item):
        v_mes0 = self.init.loc[:, item].loc[:10, ["ve", "vn", "vu"]].values[0, :]
        ang0 = self.init.loc[:, item].loc[:10, ["roll", "pitch", "yaw"]].values[0, :]
        time = self.time.loc[:, item]
        time_t = time[~time['time'].isna()].to_numpy()
        U = self.U.loc[:, item]
        U_t = torch.tensor(U[~U['ax'].isna()].to_numpy(), dtype=torch.float32)
        GT = self.GT.loc[:, item]
        GT_pose_t = torch.tensor(GT[~GT['x'].isna()].loc[:, ['x', 'y', 'z']].to_numpy(), dtype=torch.float32)
        GT_rot_t = torch.tensor(GT[~GT['x'].isna()].loc[:, 'rot_matrix'], dtype=torch.float32)
        RPE = self.list_RPE.loc[:, item]
        list_rpe = RPE[~RPE[RPE.columns[0]].isna()]
        return (v_mes0, ang0), time_t, U_t, (GT_pose_t, GT_rot_t), list_rpe

    def __len__(self):
        return len(list(self.hdf.get(self.split)))

    def __iter__(self):
        seqs = self.hdf.get(self.split)
        for name in list(seqs.keys()):
            _, t, u, gt, list_rpe = self[name]
            N = self.get_start_and_end(t)
            v_mes0 = self.init.loc[:, name].loc[:, ["ve", "vn", "vu"]].values[N[0], :]  # Get the velocity at the beginning of the sub-sequence
            ang0 = self.init.loc[:, name].loc[:, ["roll", "pitch", "yaw"]].values[N[0], :]
            yield name, ((v_mes0, ang0), t[N[0]:N[1], :], u[N[0]:N[1], :], (gt[0][N[0]:N[1], :], gt[1][N[0]:N[1], :])), list_rpe, N

    def h5_to_df(self):
        seqs = self.hdf.get(self.split)
        # print(f"Train:\n\t{list(seqs.keys())}")
        dict_u = {}
        dict_gt = {}
        dict_time = {}
        dict_init = {}
        dict_rpe = {}
        for name in list(seqs.keys()):
            # get initial condictions
            init_df = pd.DataFrame(pd.read_hdf(self.hdf_path, f"{self.split}/{name}/dataset")).loc[:, ["ve", "vn", "vu", "roll", "pitch", "yaw"]]
            dict_init[f"{name}"] = init_df  # Add the DataFrame to the dictionary

            # get time vector
            time_df = pd.DataFrame(pd.read_hdf(self.hdf_path, f"{self.split}/{name}/time"))
            dict_time[f"{name}"] = time_df  # Add the DataFrame to the dictionary

            # get IMU measurment + time
            u_df = pd.DataFrame(pd.read_hdf(self.hdf_path, f"{self.split}/{name}/w_a_input"))  # Get the input DataFrame for the given date and drive
            dict_u[f"{name}"] = u_df  # Add the DataFrame to the dictionary

            # get Ground_Truth
            gt_df = pd.DataFrame(pd.read_hdf(self.hdf_path, f"{self.split}/{name}/ground_truth"))  # Get the input DataFrame for the given date and drive
            dict_gt[f"{name}"] = gt_df  # Add the DataFrame to the dictionary

            gt_pose = torch.tensor(gt_df.loc[:, ['x', 'y', 'z']].to_numpy(), dtype=torch.float32)
            gt_rot = torch.tensor(gt_df.loc[:, 'rot_matrix'], dtype=torch.float32)
            # dict_rpe[f"{name}"] = pd.DataFrame(compute_delta_p(gt_rot, gt_pose))
            dict_rpe[f"{name}"] = pd.DataFrame(compute_delta_p(gt_rot, gt_pose), index=['idx_0', 'idx_end', 'pose_delta_p']).transpose()

        self.init = pd.concat(dict_init, axis=1)            # Create DataFrame from dictionary, with index as dict keys
        self.time = pd.concat(dict_time, axis=1)            # Create DataFrame from dictionary, with index as dict keys
        self.U = pd.concat(dict_u, axis=1)                  # Create DataFrame from dictionary, with index as dict keys
        self.GT = pd.concat(dict_gt, axis=1)                # Create DataFrame from dictionary, with index as dict keys
        self.list_RPE = pd.concat(dict_rpe, axis=1)        # Create DataFrame from dictionary, with index as dict keys

    def get_start_and_end(self, X):
        """
        Generate a random sub-sequence from training dataset with size 'self.seq_length'
        :return:
        """
        if self.split == 'train':
            N_start = 10 * int(self.rng.integers(low=0, high=(X.shape[0] - self.seq_length)/10, size=1))
            N_end = N_start + self.seq_length
        else:
            N_start = 0
            N_end = X.shape[0]
        return N_start, N_end


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

        self.conv_layers = torch.nn.Sequential(
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
        )
        self.dense_layers = torch.nn.Sequential(
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
        conv_in = x.t().unsqueeze(0)
        conv_out = self.conv_layers(conv_in)
        dense_in = conv_out.transpose(0, 2).squeeze()
        out = self.dense_layers(dense_in)
        return out


def make_trainning(model, EPOCHS):
    writer = SummaryWriter(tensorboard_path)
    # #### - Kalman Filter - #### #
    iekf = IEKF()

    # #### - Train - #### #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_bar = tqdm(total=len(train), unit="batch", desc="Training", leave=False)
    epoch_bar = tqdm(total=EPOCHS, unit="epoch", desc="Training")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        train_running_loss = 0.

        model.train()                                               # Make sure gradient tracking is on, and do a pass over the data
        optimizer.zero_grad()
        batch_bar.reset()

        for batch_index, (seq_name, (init_cond, t, inputs, ground_truth), list_rpe, N) in enumerate(train):
            # print(batch_index, seq_name)
            # print(init_cond, t.shape, inputs.shape, (ground_truth[0].shape, ground_truth[1].shape))
            # print(f"Start - end: {N}")
            inputs_net = inputs.to(DEVICE)

            # with torch.autograd.set_detect_anomaly(True):           # To pointout error in the gradients propagations
            # Je sais pas s'il faut normalizer l'input car la normalisation va scale down/up des caractéristiques des signaux
            # ce qui peut fausser l'amplitude de la sortie du CNN
            # input_net_norm = normalize(inputs_net)                  # Normalize inputs
            z_cov_net = model.forward(inputs_net)
            z_cov = z_cov_net.cpu()                                 # Move the CNN result to 'cpu' for Kalman Filter iterations
            inputs = inputs.cpu()

            Rot, p = iekf.train_run(t, inputs, z_cov, init_cond[0], init_cond[1])  # Run the training Kalman Filter, result are already in torch.Tensor

            iekf_p_loss, gt_p_loss = precompute_lost(Rot, p, list_rpe, N[0])

            loss = criterion(iekf_p_loss, gt_p_loss)                # compute loss
            if not loss.isnan():
                train_running_loss += loss

            batch_bar.set_postfix(train_loss=train_running_loss.item()/(batch_index+1), lr=optimizer.param_groups[0]['lr'])
            batch_bar.update()

        train_running_loss.backward()                               # Calculate gradients
        optimizer.step()                                            # Adjust learning weights

        train_loss = train_running_loss.item() / batch_index
        train_loss_history.append(train_loss)
        writer.add_scalar('train/loss', train_loss, epoch)

        # #### - Validation - #### #
        val_running_loss = 0.
        model.eval()
        with torch.no_grad():
            for batch_index, (seq_name, (init_cond, t, inputs, ground_truth), list_rpe, N) in enumerate(validation):
                inputs_net = inputs.to(DEVICE)

                z_cov_net = model.forward(inputs_net)
                z_cov = z_cov_net.cpu()                             # Move the CNN result to 'cpu' for Kalman Filter iterations
                inputs = inputs.cpu()

                Rot, p = iekf.train_run(t, inputs, z_cov, init_cond[0], init_cond[1])  # Run the training Kalman Filter, result are already in torch.Tensor

                iekf_p_loss, gt_p_loss = precompute_lost(Rot, p, list_rpe, N[0])

                loss = criterion(iekf_p_loss, gt_p_loss)            # compute loss

                val_running_loss += loss.item()
            val_loss = val_running_loss / (batch_index+1)
            val_loss_history.append(val_loss)
            writer.add_scalar('validation/loss', val_loss, epoch)

        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=optimizer.param_groups[0]['lr'])
        epoch_bar.update()
    return train_loss_history, val_loss_history


# #### - Main - #### #
if __name__ == '__main__':
    start_time = time.time()

    random_seed = 34                                                # set random seed
    torch.backends.cudnn.enable = False                             # Disable cuDNN use of nondeterministic algorithms
    # torch.use_deterministic_algorithms(True)                        # Configure PyTorch to use deterministic algorithms instead of nondeterministic ones where available, and throw an error if an operation is known to be nondeterministic (and without a deterministic alternative)
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
    parser.add_argument(
        "-s", "--seq", type=int, required=True, help="Length sequence of data. Ex: --seq 2000"
    )

    args = parser.parse_args()

    EPOCHS = args.epochs
    DEVICE = args.device
    SEQ_LEN = args.seq

    print(f"Epochs: {EPOCHS}; Device: {DEVICE}; Sequence length: {SEQ_LEN}")

    save_path = "../data/processed/dataset.h5"                      # Path to the .h5 dataset
    run_time = time.strftime('%Y%m%d_%H%M%S')
    tensorboard_path = f"../runs/{run_time}"                                    # Path to the TensorBoard directory

    train = KittiDataset(save_path, SEQ_LEN, 'train', rng)
    validation = KittiDataset(save_path, SEQ_LEN, 'validation', rng)

    # Model
    model = CNN(SEQ_LEN).to(DEVICE)

    # # test the model and fixe seed generation
    # test_input = torch.rand((2000, 6)).to(DEVICE)
    # prediction = model(test_input)
    # print(prediction)
    # exit()

    # Loss
    criterion = torch.nn.MSELoss().to(DEVICE)

    train_loss_history, val_loss_history = make_trainning(model, EPOCHS)

    create_folder(f"../models/{run_time}")
    torch.save(model, f"../models/{run_time}/CNN.pt")

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

    torch.cuda.empty_cache()
