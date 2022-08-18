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
        if self.split == "train":
            return len(list(self.hdf.get(f"ETV_dataset/seq_{SEQ_LEN}/{self.split}")))
        else:
            return len(list(self.hdf.get(f"ETV_dataset/{self.split}")))

    def len_batch(self):
        if self.split == "train":
            batch = self.hdf.get(f"ETV_dataset/seq_{SEQ_LEN}/{self.split}")
            return len(list(batch.get(list(batch.keys())[0])))
        else:
            return 1

    def len_seq(self):
        if self.split == "train":
            batch = self.hdf.get(f"ETV_dataset/seq_{SEQ_LEN}/{self.split}")
            seq = pd.DataFrame(pd.read_hdf(self.hdf_path, f"ETV_dataset/seq_{SEQ_LEN}/{self.split}/{list(batch.keys())[0]}/batch_1/time"))
            return seq.shape[0]
        else:
            batch = self.hdf.get(f"ETV_dataset/{self.split}")
            seq = pd.DataFrame(pd.read_hdf(self.hdf_path, f"ETV_dataset/{self.split}/{list(batch.keys())[0]}/time"))
            return seq.shape[0]

    def seq_name(self):
        if self.split == "train":
            batch = self.hdf.get(f"ETV_dataset/seq_{SEQ_LEN}/{self.split}")
            for name in list(batch.keys()):
                yield name
        else:
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
    def __init__(self):
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
            torch.nn.Dropout(p=0.5),                                # Randomly zeroes some of the elements of the input tensor with probability p, using samples from a Bernoulli distribution
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


class Nathan_CNN(torch.nn.Module):
    """
    CNN architecture based on Nathan LAOUÉ work: https://github.com/Naatyu/MagNav
    """
    # @timming
    def __init__(self):
        super(Nathan_CNN, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            # Convolutional part
            torch.nn.Conv1d(in_channels=6,
                            out_channels=8,
                            kernel_size=3,
                            padding=1),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(in_channels=8,
                            out_channels=16,
                            kernel_size=3,
                            padding=1),
            torch.nn.LeakyReLU(),
        )
        self.dense_layers = torch.nn.Sequential(
            # Fully connected part
            torch.nn.Flatten(),                                     # Flattens a contigous range of dims into a tensor
            torch.nn.Linear(16, 8),                                 # Applies a linear transformation to the incoming data: y = xA.T + b
            torch.nn.Linear(8, 2),                                 # Applies a linear transformation to the incoming data: y = xA.T + b
            torch.nn.Tanh()                                         # Applies the Hyperbolic Tangent (Tanh) function element-wise: Tanh(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        )

        torch.nn.init.kaiming_normal_(self.conv_layers[0].weight, nonlinearity='relu')
        torch.nn.init.constant_(self.conv_layers[0].bias, 0)
        torch.nn.init.kaiming_normal_(self.conv_layers[2].weight, nonlinearity='relu')
        torch.nn.init.constant_(self.conv_layers[2].bias, 0)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # , weight_decay=1e-8)

    # if not VERBOSE:
    batch_bar = tqdm(total=BATCH_NUMBER, unit="batch", desc="Training", leave=False)
    epoch_bar = tqdm(total=EPOCHS, unit="epoch", desc="Training")

    train_loss_history = []
    val_loss_history = []

    batch_array = np.array([k for k in range(0, BATCH_NUMBER)]) % FULL_BATCH
    for epoch in range(EPOCHS):
        train_running_loss = torch.zeros(1)

        model.train()                                                   # Make sure gradient tracking is on, and do a pass over the data
        optimizer.zero_grad()
        if not VERBOSE:
            batch_bar.reset()
        else:
            print(f"Epoch n°{epoch+1}:")
            print(f"  Training:")
        for i in batch_array:
            train_running_loss = torch.zeros(1)
            if VERBOSE:
                print(f"    batch n°{i+1}")
            for batch_index, drive in enumerate(train.seq_name()):
                base_key_train = f"ETV_dataset/seq_{SEQ_LEN}/train/{drive}/batch_{i+1}"
                ground_truth = pd.DataFrame(pd.read_hdf(save_path, f"{base_key_train}/ground_truth"))
                pose_gt = ground_truth.loc[:, ['pose_x', 'pose_y', 'pose_z']].values
                rot_mat_gt = ground_truth.loc[:, ['rot_matrix']].values
                inputs = torch.tensor(pd.DataFrame(pd.read_hdf(save_path, f"{base_key_train}/input")).to_numpy(), dtype=torch.float32)
                t = torch.tensor(pd.DataFrame(pd.read_hdf(save_path, f"{base_key_train}/time")).to_numpy(), dtype=torch.float32)
                init_cond = pd.DataFrame(pd.read_hdf(save_path, f"{base_key_train}/init_cond")).values

                inputs_net = inputs.to(DEVICE)

                z_cov_net = model.forward(inputs_net)
                z_cov = z_cov_net.cpu()                                 # Move the CNN result to 'cpu' for Kalman Filter iterations
                inputs = inputs.cpu()

                Rot, p = iekf.train_run(t, inputs, z_cov, init_cond[:3, 0], init_cond[3:, 0])  # Run the training Kalman Filter, result are already in torch.Tensor

                loss = criterion((rot_mat_gt, pose_gt), (Rot, p))

                if not loss.isnan():
                    train_running_loss += loss
                if VERBOSE:
                    print(f"        sub-seq {i}: loss: {loss}")
            if not VERBOSE:
                batch_bar.set_postfix(mean_batch_loss=train_running_loss.item()/(batch_index+1), lr=optimizer.param_groups[0]['lr'])
                batch_bar.update()

            train_running_loss.backward()                               # Calculate gradients
        optimizer.step()                                                # Adjust learning weights

        train_loss = train_running_loss.item() / batch_index
        train_loss_history.append(train_loss)
        writer.add_scalar('train/loss', train_loss, epoch)

        if ROLL:
            batch_array = (batch_array + BATCH_NUMBER) % FULL_BATCH
        else:
            batch_array = np.array([k for k in range(0, BATCH_NUMBER)]) % FULL_BATCH

        # #### - Validation - #### #
        if VERBOSE:
            print(f"  Validation:")
        val_running_loss = 0.
        model.eval()
        with torch.no_grad():
            for batch_index, drive in enumerate(validation.seq_name()):
                base_key_valid = f"ETV_dataset/validation/{drive}"
                ground_truth = pd.DataFrame(pd.read_hdf(save_path, f"{base_key_valid}/ground_truth"))
                pose_gt = ground_truth.loc[:, ['pose_x', 'pose_y', 'pose_z']].values
                rot_mat_gt = ground_truth.loc[:, ['rot_matrix']].values
                inputs = torch.tensor(pd.DataFrame(pd.read_hdf(save_path, f"{base_key_valid}/input")).to_numpy(), dtype=torch.float32)
                t = torch.tensor(pd.DataFrame(pd.read_hdf(save_path, f"{base_key_valid}/time")).to_numpy(), dtype=torch.float32)
                init_cond = pd.DataFrame(pd.read_hdf(save_path, f"{base_key_valid}/init_cond")).values

                inputs_net = inputs.to(DEVICE)

                z_cov_net = model.forward(inputs_net)
                z_cov = z_cov_net.cpu()                             # Move the CNN result to 'cpu' for Kalman Filter iterations
                inputs = inputs.cpu()

                Rot, p = iekf.train_run(t, inputs, z_cov, init_cond[:3, 0], init_cond[3:, 0])  # Run the training Kalman Filter, result are already in torch.Tensor

                loss = criterion((rot_mat_gt, pose_gt), (Rot, p))

                val_running_loss += loss.item()
            val_loss = val_running_loss / (batch_index+1)
            writer.add_scalar('validation/loss', val_loss, epoch)
            if VERBOSE:
                print(f"    Loss: {val_loss}")
            if epoch > 1:
                if val_loss < min(val_loss_history):
                    torch.save(model.state_dict(), f"../models/{run_time[:-7]}/{RESEAU}_S{SEQ_LEN}_B{BATCH_NUMBER}_LR{LEARNING_RATE}_E{EPOCHS}_R{ROLL}.pt")
                    global SAVE_MODEL_BOOL
                    SAVE_MODEL_BOOL = True
            val_loss_history.append(val_loss)

        if epoch > 20:
            if val_loss >= val_loss_history[-15:]:
                with open(f"../models/{run_time[:-7]}/parameters_{RESEAU}_S{SEQ_LEN}_B{BATCH_NUMBER}_LR{LEARNING_RATE}_E{EPOCHS}_R{ROLL}.txt", "w") as f:
                    f.write(f"Epochs: \n{EPOCHS}\n\n")
                    f.write(f"Batch size:\n{BATCH_NUMBER}; rolling: {ROLL}\n\n")
                    f.write(f"Sequence length:\n{SEQ_LEN}\n\n")
                    f.write(f"Execution time:\n{round(time.time()-start_time, 1)}\n\n")
                    f.write(f"Architecture:\n{model}\n\n")
                    f.write(f"Training loss history:\n{train_loss_history}\n\n")
                    f.write(f"Validation loss history:\n{val_loss_history}\n\n")
                    f.write(f"Best model save at epoch:\n{val_loss_history.index(min(val_loss_history))}\n\n")
                exit(epoch)

        if VERBOSE:
            print(f"")
        else:
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
        "-d", "--device", type=str, default='cuda', help="Which GPU to use (cuda or cpu). Ex: --device 'cuda0'"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of epochs to train the model. Ex: --epochs 400"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=4, help="Batch size for training. Ex: --batch 32"
    )
    parser.add_argument(
        "-r", "--roll", type=str, default=False, help="Rolling through all data batch. Ex: --roll TRUE"
    )
    parser.add_argument(
        "-s", "--seq", type=int, default=3000, help="Length sequence of data. Ex: --seq 2000"
    )
    parser.add_argument(
        "-l", "--lr", type=str, default=1e-3, help="Learning rate. Ex: --lr 1e-3"
    )
    parser.add_argument(
        "-v", "--verbose", type=str, default=False, help="Print all training metadata if true, else use 'tqdm' bar. Ex: --verbose true"
    )
    parser.add_argument(
        "-c", "--cnn", type=str, default="CNN", help="Choose CNN type. Ex: --cnn Nathan_CNN"
    )

    args = parser.parse_args()

    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            pass

    EPOCHS = args.epochs
    DEVICE = args.device
    SEQ_LEN = args.seq
    LEARNING_RATE = float(str(args.lr))
    ROLL = t_or_f(args.roll)
    VERBOSE = t_or_f(args.verbose)
    RESEAU = args.cnn

    save_path = "../data/processed/dataset.h5"                                  # Path to the .h5 dataset
    run_time = time.strftime('%Y%m%d_%H%M%S')
    create_folder(f"../models/{run_time[:-7]}")

    train = KittiDataset(save_path, 'train')
    validation = KittiDataset(save_path, 'validation')

    if args.batch > train.len_batch():
        cprint(f"Batch size argument too large: {args.batch}, but dataset batch size is {train.len_batch()}", "yellow")
        BATCH_NUMBER = train.len_batch()
        print(f"batch size used is: {BATCH_NUMBER}")
    else:
        BATCH_NUMBER = args.batch

    FULL_BATCH = train.len_batch()

    print(f"Reseau: {RESEAU}, Device: {DEVICE}; Epochs: {EPOCHS}; Batch size: {args.batch}; Rolling batch: {ROLL}; Sequence length: {SEQ_LEN}; Learning rate: {LEARNING_RATE}")
    print("Verbose enable !" if VERBOSE else "Verbose disable !")

    tensorboard_path = f"../runs/{run_time}_{RESEAU}_S{SEQ_LEN}_B{BATCH_NUMBER}_LR{LEARNING_RATE}_E{EPOCHS}_R{ROLL}"            # Path to the TensorBoard directory

    # Model
    SAVE_MODEL_BOOL = False
    if RESEAU == "Nathan_CNN":
        model = Nathan_CNN().to(DEVICE)
    else:
        model = CNN().to(DEVICE)

    # # test the model and fixe seed generation
    # test_input = torch.rand((2000, 6)).to(DEVICE)
    # prediction = model(test_input)
    # print(prediction)
    # exit()

    criterion = RMSELoss().to(DEVICE)

    train_loss_history, val_loss_history = make_trainning(model, EPOCHS)

    if not SAVE_MODEL_BOOL:
        torch.save(model.state_dict(), f"../models/{run_time[:-7]}/{RESEAU}_S{SEQ_LEN}_B{BATCH_NUMBER}_LR{LEARNING_RATE}_E{EPOCHS}_R{ROLL}.pt")

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

    with open(f"../models/{run_time[:-7]}/parameters_{RESEAU}_S{SEQ_LEN}_B{BATCH_NUMBER}_LR{LEARNING_RATE}_E{EPOCHS}_R{ROLL}.txt", "w") as f:
        f.write(f"Epochs: \n{EPOCHS}\n\n")
        f.write(f"Batch size:\n{BATCH_NUMBER}; rolling: {ROLL}\n\n")
        f.write(f"Learning rate:\n{LEARNING_RATE}\n\n")
        f.write(f"Sequence length:\n{SEQ_LEN}\n\n")
        f.write(f"Execution time:\n{round(time.time()-start_time, 1)}\n\n")
        f.write(f"Architecture:\n{model}\n\n")
        f.write(f"Training loss history:\n{train_loss_history}\n\n")
        f.write(f"Validation loss history:\n{val_loss_history}\n\n")
        f.write(f"Best model save at epoch:\n{val_loss_history.index(min(val_loss_history))}\n\n")

    torch.cuda.empty_cache()
