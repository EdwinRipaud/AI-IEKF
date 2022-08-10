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


def precompute_lost(Rot, p, list_rpe):
    # compute the relative translational error
    N = p.shape[0]
    Rot_10_Hz = Rot[::10]
    p_10_Hz = p[::10]
    idxs_0 = torch.Tensor(np.array(list_rpe.loc[:, 'idx_0'].values, dtype=float)).long()
    idxs_end = torch.Tensor(np.array(list_rpe.loc[:, 'idx_end'].values, dtype=float)).long()
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


@timming
def rot_pose_to_se3(rot, pose):
    """
    Transform rotation matrix and position to SE3 matrix
    :param rot: torch tensor or numpy array
    :param pose: torch tensor or numpy array
    :return:
    """
    se3 = []
    if type(pose).__module__ == np.__name__:
        pose = torch.tensor(pose)

    for i in range(pose.shape[0]):
        if type(rot[i]).__module__ == np.__name__:
            r = torch.tensor(rot[i][0])
        else:
            r = rot[i]
        M = torch.eye(4)
        M[:3, :3] = r
        M[:3, 3] = pose[i]
        se3.append(M)
    return se3


# #### - Class - #### #
class KittiDataset(Dataset):
    def __init__(self, hdf_path, split):
        self.hdf_path = hdf_path
        self.hdf = h5py.File(self.hdf_path, 'r')                    # Read the .h5 file
        self.split = split

    def __getitem__(self, item):
        dataset = pd.DataFrame(pd.read_hdf(self.hdf_path, f"full_datset/{item}"))
        time_vec = torch.tensor(dataset[['time']].to_numpy(), dtype=torch.float32)
        u = torch.tensor(dataset[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].to_numpy(), dtype=torch.float32)
        p_gt = torch.tensor(dataset.loc[:, ['pose_x', 'pose_y', 'pose_z']].to_numpy(), dtype=torch.float32)
        ang_gt = torch.tensor(dataset.loc[:, 'rot_matrix'], dtype=torch.float32)
        v_mes0 = dataset[["ve", "vn", "vu"]].copy().iloc[0, :].values
        ang0 = dataset[["roll", "pitch", "yaw"]].copy().iloc[0, :].values
        list_rpe = dataset.loc[:, ['idx_0', 'idx_end', 'pose_delta_p']]
        return {"init_cond": (v_mes0, ang0), "time": time_vec, "input": u, "ground_truth": (p_gt, ang_gt), "liste_RPE": list_rpe}

    def len(self):
        return len(list(self.hdf.get(f"ETV_dataset/{self.split}")))

    def len_batch(self):
        if self.split == "train":
            batch = self.hdf.get(f"ETV_dataset/{self.split}")
            return len(list(batch.get(list(batch.keys())[0])))
        else:
            return 1

    def len_seq(self):
        batch = self.hdf.get(f"ETV_dataset/{self.split}")
        seq = pd.read_hdf(self.hdf_path, f"ETV_dataset/{self.split}/{list(batch.keys())[0]}/batch_1/time")
        return seq.shape[0]

    def seq_name(self):
        batch = self.hdf.get(f"ETV_dataset/{self.split}")
        for name in list(batch.keys()):
            yield name

    def __iter__(self):
        raise NotImplementedError
        # seqs = self.hdf.get(self.split)
        # for name in list(seqs.keys()):
        #     _, t, u, gt, list_rpe = self[name]
        #     N = self.get_start_and_end(t)
        #     v_mes0 = self.init.loc[:, name].loc[:, ["ve", "vn", "vu"]].values[N[0], :]  # Get the velocity at the beginning of the sub-sequence
        #     ang0 = self.init.loc[:, name].loc[:, ["roll", "pitch", "yaw"]].values[N[0], :]
        #     yield name, ((v_mes0, ang0), t[N[0]:N[1], :], u[N[0]:N[1], :], (gt[0][N[0]:N[1], :], gt[1][N[0]:N[1], :])), list_rpe, N


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, kf_rot_p, gt_rot_p):
        """
        Compute RMSE of a trajectory relative to the ground truth using the Full Relative Position
        :param gt_rot_p: (rotation matrix, position) for the Ground-truth trajectory
        :param kf_rot_p: (rotation matrix, position) for the Kalman trajectory
        :return:
        """
        # uniformise type of inputs
        rot_gt, pose_gt = gt_rot_p
        rot_kf, pose_kf = kf_rot_p
        if type(pose_gt).__module__ == np.__name__:
            pose_gt = torch.tensor(pose_gt)
        if type(pose_kf).__module__ == np.__name__:
            pose_kf = torch.tensor(pose_kf)

        if pose_gt.shape[0] == pose_kf.shape[0]:
            s_error = torch.zeros((pose_kf.shape[0], 1))
            for i in range(pose_kf.shape[0]):
                if type(rot_gt[i]).__module__ == np.__name__:
                    if rot_gt[i].shape[0] == 1:
                        r_gt = torch.tensor(rot_gt[i][0])
                    else:
                        r_gt = torch.tensor(rot_gt[i])
                else:
                    r_gt = rot_gt[i]
                if type(rot_kf[i]).__module__ == np.__name__:
                    if rot_kf[i].shape[0] == 1:
                        r_kf = torch.tensor(rot_kf[i][0])
                    else:
                        r_kf = torch.tensor(rot_kf[i])
                else:
                    r_kf = rot_kf[i]
                # Compute SE3 matrix for Ground-truth and Kalman
                M_gt = torch.eye(4)
                M_gt[:3, :3] = r_gt
                M_gt[:3, 3] = pose_gt[i]

                # compute relative se3
                r_inv = torch.t(r_kf)
                t_inv = -r_inv.mv(pose_kf[i])
                M_kf_inv = torch.eye(4)
                M_kf_inv[:3, :3] = r_inv
                M_kf_inv[:3, 3] = t_inv

                relative_se3 = M_kf_inv.mm(M_gt)
                error = torch.linalg.norm(relative_se3-torch.eye(4))
                s_error[i] = torch.pow(error, 2)
            return torch.sqrt(torch.mean(s_error)) + 1e-8


class CNN(torch.nn.Module):
    # @timming
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

    # @timming
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)

    batch_bar = tqdm(total=train.len(), unit="batch", desc="Training", leave=False)
    epoch_bar = tqdm(total=EPOCHS, unit="epoch", desc="Training")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        # print(f"Epoch n°{epoch+1}:")
        train_running_loss = torch.zeros(1)

        model.train()                                                   # Make sure gradient tracking is on, and do a pass over the data
        optimizer.zero_grad()
        batch_bar.reset()
        # print(f"  Training:")
        for batch_index, drive in enumerate(train.seq_name()):
            train_running_loss = torch.zeros(1)
            # print(f"    batch n°{batch_index + 1}: {drive}")
            for i in range(1, BATCH_SIZE+1):
                ground_truth = pd.read_hdf(save_path, f"ETV_dataset/train/{drive}/batch_{i}/ground_truth")
                pose_gt = ground_truth.loc[:, ['pose_x', 'pose_y', 'pose_z']].values
                rot_mat_gt = ground_truth.loc[:, ['rot_matrix']].values
                # list_rpe = pd.read_hdf(save_path, f"ETV_dataset/train/{drive}/batch_{i}/list_RPE")
                inputs = torch.tensor(pd.read_hdf(save_path, f"ETV_dataset/train/{drive}/batch_{i}/input").to_numpy(), dtype=torch.float32)
                t = torch.tensor(pd.read_hdf(save_path, f"ETV_dataset/train/{drive}/batch_{i}/time").to_numpy(), dtype=torch.float32)
                init_cond = pd.read_hdf(save_path, f"ETV_dataset/train/{drive}/batch_{i}/init_cond").values

                inputs_net = inputs.to(DEVICE)

                z_cov_net = model.forward(inputs_net)
                z_cov = z_cov_net.cpu()                                 # Move the CNN result to 'cpu' for Kalman Filter iterations
                inputs = inputs.cpu()

                Rot, p = iekf.train_run(t, inputs, z_cov, init_cond[:3], init_cond[3:])  # Run the training Kalman Filter, result are already in torch.Tensor

                # iekf_p_loss, gt_p_loss = precompute_lost(Rot, p, list_rpe)
                # loss = criterion(iekf_p_loss, gt_p_loss)                # compute loss

                loss = criterion((rot_mat_gt, pose_gt), (Rot, p))

                if not loss.isnan():
                    train_running_loss += loss
                #     print(f"        sub-seq {i}: loss: {loss}")
                # else:
                #     print(f"        sub-seq {i}: loss: Nan")

            batch_bar.set_postfix(mean_batch_loss=train_running_loss.item()/BATCH_SIZE, lr=optimizer.param_groups[0]['lr'])
            batch_bar.update()

            train_running_loss.backward()                                   # Calculate gradients
        optimizer.step()                                                # Adjust learning weights

        train_loss = train_running_loss.item() / batch_index
        train_loss_history.append(train_loss)
        writer.add_scalar('train/loss', train_loss, epoch)

        if True:
            # #### - Validation - #### #
            # print(f"  Validation:")
            val_running_loss = 0.
            model.eval()
            with torch.no_grad():
                for batch_index, drive in enumerate(validation.seq_name()):
                    ground_truth = pd.read_hdf(save_path, f"ETV_dataset/validation/{drive}/ground_truth")
                    pose_gt = ground_truth.loc[:, ['pose_x', 'pose_y', 'pose_z']].values
                    rot_mat_gt = ground_truth.loc[:, ['rot_matrix']].values
                    # list_rpe = pd.read_hdf(save_path, f"ETV_dataset/train/{drive}/batch_{i}/list_RPE")
                    inputs = torch.tensor(pd.read_hdf(save_path, f"ETV_dataset/validation/{drive}/input").to_numpy(), dtype=torch.float32)
                    t = torch.tensor(pd.read_hdf(save_path, f"ETV_dataset/validation/{drive}/time").to_numpy(), dtype=torch.float32)
                    init_cond = pd.read_hdf(save_path, f"ETV_dataset/validation/{drive}/init_cond").values

                    inputs_net = inputs.to(DEVICE)

                    z_cov_net = model.forward(inputs_net)
                    z_cov = z_cov_net.cpu()                             # Move the CNN result to 'cpu' for Kalman Filter iterations
                    inputs = inputs.cpu()

                    Rot, p = iekf.train_run(t, inputs, z_cov, init_cond[:3], init_cond[3:])  # Run the training Kalman Filter, result are already in torch.Tensor

                    # iekf_p_loss, gt_p_loss = precompute_lost(Rot, p, list_rpe, N[0])

                    # loss = criterion(iekf_p_loss, gt_p_loss)            # compute loss
                    loss = criterion((rot_mat_gt, pose_gt), (Rot, p))

                    val_running_loss += loss.item()
                val_loss = val_running_loss / (batch_index+1)
                val_loss_history.append(val_loss)
                writer.add_scalar('validation/loss', val_loss, epoch)
                # print(f"    Loss: {val_loss}")

        # print(f"")
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
    # parser.add_argument(
    #     "-s", "--seq", type=int, required=True, help="Length sequence of data. Ex: --seq 2000"
    # )
    # parser.add_argument(
    #     "-b", "--batch", type=int, required=True, help="Batch size for training. Ex: --batch 32"
    # )

    args = parser.parse_args()

    save_path = "../data/processed/dataset.h5"                                  # Path to the .h5 dataset
    run_time = time.strftime('%Y%m%d_%H%M%S')
    tensorboard_path = f"../runs/{run_time}"                                    # Path to the TensorBoard directory

    train = KittiDataset(save_path, 'train')
    validation = KittiDataset(save_path, 'validation')

    EPOCHS = args.epochs
    DEVICE = args.device
    SEQ_LEN = train.len_seq()  # args.seq
    BATCH_SIZE = 8  # train.len_batch()  # args.batch

    print(f"Epochs: {EPOCHS}; Device: {DEVICE}; Sequence length: {SEQ_LEN}")
    # exit()

    if True:
        # Model
        model = CNN(SEQ_LEN).to(DEVICE)

        # # test the model and fixe seed generation
        # test_input = torch.rand((2000, 6)).to(DEVICE)
        # prediction = model(test_input)
        # print(prediction)
        # exit()

        # Loss
        # criterion = torch.nn.MSELoss().to(DEVICE)
        criterion = RMSELoss().to(DEVICE)

        train_loss_history, val_loss_history = make_trainning(model, EPOCHS)

        create_folder(f"../models/{run_time}")
        torch.save(model, f"../models/{run_time}/CNN.pt")

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

    torch.cuda.empty_cache()
