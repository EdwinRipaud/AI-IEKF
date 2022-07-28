from torch.utils.data import Dataset, DataLoader
from termcolor import cprint
import numpy as np
import argparse
import random
import torch
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


# #### - Class - #### #
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
    parser.add_argument(
        "-b", "--batch", type=int, required=True, help="Batch size for training. Ex: --batch 64"
    )
    parser.add_argument(
        "-s", "--seq", type=int, required=True, help="Length sequence of data. Ex: --seq 2000"
    )

    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    DEVICE = args.device
    SEQ_LEN = args.seq

    print(f"Epochs: {EPOCHS}; Batch size: {BATCH_SIZE}; Device: {DEVICE}; Sequence length: {SEQ_LEN}")

    model = CNN(SEQ_LEN).to(DEVICE)
    model.name = 'CNN'

    # test the model and fixe seed generation
    test_input = torch.rand((1, 6, 500000)).to(DEVICE)
    prediction = model(test_input)
    print(prediction)

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

    torch.cuda.empty_cache()
