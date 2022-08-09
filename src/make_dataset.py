from scipy.spatial.transform import Rotation as scipyRot
from termcolor import cprint
from pandas import HDFStore
from navpy import lla2ned
import pandas as pd
import numpy as np
import argparse
import datetime
import warnings
import random
import glob
import time
import os


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
class BaseDataset:
    fields = ["lat", "lon", "alt", "roll", "pitch", "yaw", "vn", "ve", "vf", "vl", "vu", "ax", "ay", "az", "af",
              "al", "au", "wx", "wy", "wz", "wf", "wl", "wu", "pos_accuracy", "vel_accuracy", "navstat",
              "numsats", "posemode", "velmode", "orimode"]

    dataset_split = {
        'train': ["2011_09_30_drive_0018_extract",
                  "2011_09_30_drive_0020_extract",
                  "2011_09_30_drive_0027_extract",
                  "2011_09_30_drive_0033_extract",
                  "2011_09_30_drive_0034_extract",
                  "2011_10_03_drive_0027_extract",
                  "2011_10_03_drive_0034_extract",
                  "2011_10_03_drive_0042_extract"],

        'validation': ["2011_09_30_drive_0028_extract"],

        'test': ["2011_09_26_drive_0067_extract",
                 "2011_09_30_drive_0016_extract",
                 "2011_09_30_drive_0018_extract",
                 "2011_09_30_drive_0020_extract",
                 "2011_09_30_drive_0027_extract",
                 "2011_09_30_drive_0028_extract",
                 "2011_09_30_drive_0033_extract",
                 "2011_09_30_drive_0034_extract",
                 "2011_10_03_drive_0027_extract",
                 "2011_10_03_drive_0034_extract",
                 "2011_10_03_drive_0042_extract"]
    }

    @timming
    def __init__(self, raw_data_path, processed_data_path, sequence_length, batch_size,
                 maximum_sample_loss=150, minimum_sequence_length=6000, rng_seed=17):
        super(BaseDataset, self).__init__()

        self.min_seq_len = minimum_sequence_length
        self.maximum_sample_loss = maximum_sample_loss
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.frequency = 100  # sampling frequency

        self.raw_total_duration = 0
        self.process_total_duration = 0

        self.raw_data_path = raw_data_path
        self.process_data_path = processed_data_path
        self.create_folder(self.process_data_path)

        self.rng = np.random.default_rng(rng_seed)

        print(f"Runs data processing function to turn raw data from ({self.raw_data_path}) into cleaned data ready to "
              f"be analyzed (saved in {self.process_data_path})")

    @timming
    def create_dataset(self, h5_name="dataset.h5", bypass_date=None, bypass_drive=None):
        """
        Create dataset under .h5 file with data stored as pandas DataFrame.
        :param h5_name: name for the created .h5 file dataset
        :param bypass_date:
        :param bypass_drive:
        :return:
        """
        h5_name = h5_name+'.h5' if not h5_name.endswith(".h5") else h5_name
        hdf_path = os.path.join(self.process_data_path, h5_name)
        print(f"Create dataset in {hdf_path}")
        # check for older dataset file with the same name and clear them
        if os.path.exists(os.path.join(self.process_data_path, h5_name)):
            os.remove(os.path.join(self.process_data_path, h5_name))

        hdf = HDFStore(hdf_path)                   # create the HDF instance
        if bypass_date:
            if type(bypass_date) == str:
                date_dirs = [bypass_date]
            elif type(bypass_date) == int:
                date_dirs = [os.listdir(self.raw_data_path)[bypass_date]]  # To test for one specifique sequence
        else:
            date_dirs = os.listdir(self.raw_data_path)

        for n_iter, date_dir in enumerate(date_dirs):
            # get access to each sequence
            path1 = os.path.join(self.raw_data_path, date_dir)
            if not os.path.isdir(path1):
                continue

            if bypass_drive:
                if type(bypass_drive) == str:
                    date_dirs2 = [bypass_drive]
                elif type(bypass_drive) == int:
                    date_dirs2 = [os.listdir(path1)[bypass_drive]]  # To test for one specifique sequence
            else:
                date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue

                dataset = self.load_data_files(path2, date_dir2)
                if dataset is None:
                    continue
                # ## - Save full dataset - ## #
                dataset.to_hdf(hdf_path, key=f"full_datset/day_{date_dir2}")       # save full dataset

                if date_dir2 in self.dataset_split['train']:
                    init_cond_df = dataset[["ve", "vn", "vu", "roll", "pitch", "yaw"]].copy()
                    u_df = dataset[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].copy()
                    gt_df = dataset[['rot_matrix', 'pose_x', 'pose_y', 'pose_z']].copy()
                    time_df = dataset[['time']].copy()
                    for k in range(self.batch_size):                                                                    # generate 'BATCH_SIZE' random sub-sequences for training
                        base_key = f"ETV_dataset/train/day_{date_dir2}/batch_{k+1}"
                        N = 10 * self.rng.integers(low=0, high=(time_df.shape[0]-self.sequence_length)/10, size=1)[0]
                        u_df.iloc[N:N+self.sequence_length, :].to_hdf(hdf_path, key=f"{base_key}/input")
                        gt_df.iloc[N:N+self.sequence_length, :].to_hdf(hdf_path, key=f"{base_key}/ground_truth")
                        time_df.iloc[N:N+self.sequence_length, :].to_hdf(hdf_path, key=f"{base_key}/time")
                        init_cond_df.iloc[N, :].to_hdf(hdf_path, key=f"{base_key}/init_cond")

                if date_dir2 in self.dataset_split['validation']:
                    base_key = f"ETV_dataset/validation/day_{date_dir2}"

                    init_cond_df = dataset[["ve", "vn", "vu", "roll", "pitch", "yaw"]].copy()
                    init_cond_df.to_hdf(hdf_path, key=f"{base_key}/init_cond")

                    u_df = dataset[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].copy()
                    u_df.to_hdf(hdf_path, key=f"{base_key}/input")

                    gt_df = dataset[['rot_matrix', 'pose_x', 'pose_y', 'pose_z']].copy()
                    gt_df.to_hdf(hdf_path, key=f"{base_key}/ground_truth")

                    time_df = dataset[['time']].copy()
                    time_df.to_hdf(hdf_path, key=f"{base_key}/time")

                if date_dir2 in self.dataset_split['test']:
                    base_key = f"ETV_dataset/test/day_{date_dir2}"

                    init_cond_df = dataset[["ve", "vn", "vu", "roll", "pitch", "yaw"]].copy()
                    init_cond_df.to_hdf(hdf_path, key=f"{base_key}/init_cond")

                    u_df = dataset[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].copy()
                    u_df.to_hdf(hdf_path, key=f"{base_key}/input")

                    gt_df = dataset[['rot_matrix', 'pose_x', 'pose_y', 'pose_z']].copy()
                    gt_df.to_hdf(hdf_path, key=f"{base_key}/ground_truth")

                    time_df = dataset[['time']].copy()
                    time_df.to_hdf(hdf_path, key=f"{base_key}/time")

        hdf.close()
        print(f"\nInitial dataset duration: {round(self.raw_total_duration/100, 2)} s\n"
              f"Selected portion total duration: {round(self.process_total_duration/100, 2)} s\n"
              f"Propotion keep: {round(100*(self.process_total_duration/self.raw_total_duration), 2)} %\n")
        return

    def load_data_files(self, path2, date_dir2):
        """
        Read the data from the KITTI dataset
        """
        # read data
        oxts_files = sorted(glob.glob(os.path.join(path2, 'oxts', 'data', '*.txt')))

        seq_base_length = len(oxts_files)
        self.raw_total_duration += seq_base_length
        print(f"Sequence: {date_dir2}\n\tduration: {round(seq_base_length/100, 2)} s")
        if seq_base_length < self.min_seq_len:                                                                  # Check if the length of the sequence is at leas equal to the minimum require length
            cprint(f"\tSequence is too short ({round(seq_base_length / 100, 2)} s)", 'yellow')
            return None

        dataset = self.oxts_packets_2_dataframe(oxts_files)                                                     # DataFrame containing the full corrected dataset
        t = self.load_timestamps(path2)
        dataset['time'] = t

        dataset = dataset.sort_values(by='time').drop_duplicates(subset=['time'])                               # Sort timestamps by ascending order and delet duplicated timestamps

        if np.min(np.diff(dataset['time'].to_numpy())) < 0:                                                     # check if there is no remaining negative delta in time vector
            cprint("\tSequence has negative time step values", 'red')

        # Some sequences have sampling frequency outage
        if np.max(np.diff(dataset['time'].to_numpy())) > (self.maximum_sample_loss / self.frequency):
            cprint(f"\tSequence has time jumps > {self.maximum_sample_loss / self.frequency} s", 'blue')
            dataset = self.select_clear(dataset)                                                                # if there is big jump in time, we selecte the largest periode where the sampling time is within the acceptable range

        seq_process_duration = dataset['time'].shape[0]
        self.process_total_duration += seq_process_duration
        if seq_process_duration < self.min_seq_len:                                                         # Check if the length of the sequence after the cleaning process is at leas equal to the minimum require length
            cprint(f"\tSelected part is too short ({round(seq_process_duration / 100, 2)} s)", 'yellow')
            return None
        else:
            print(f"\tSelected part: {round(seq_process_duration/100, 2)} s")

        pos_gt = lla2ned(dataset.lat, dataset.lon, dataset.alt,                                                 # get the Ground Truth position from GPS tracting
                         dataset.lat.iloc[0], dataset.lon.iloc[0], dataset.alt.iloc[0],
                         latlon_unit='deg', alt_unit='m', model='wgs84')
        """
        Note on difference between ground truth and oxts solution:
            - orientation is the same
            - north and east axis are inverted
            - z axes are opposed
            - position are closed to but different
        => oxts solution is not loaded
        """
        pos_gt[:, [0, 1]] = pos_gt[:, [1, 0]]                                                                   # Swap X and Y axis as explaine in the commented note
        pos_gt[:, 2] *= -1
        dataset['pose_x'] = pos_gt[:, 0]
        dataset['pose_y'] = pos_gt[:, 1]
        dataset['pose_z'] = pos_gt[:, 2]

        roll, pitch, yaw = dataset.roll.values, dataset.pitch.values, dataset.yaw.values
        Rot_gt = []
        for i in range(roll.shape[0]):
            rot = scipyRot.from_euler('xyz', [roll[i], pitch[i], yaw[i]])
            Rot_gt.append(rot.as_matrix())  # Set initial car orientation
        dataset['rot_matrix'] = Rot_gt
        return dataset.copy()

    def oxts_packets_2_dataframe(self, oxts_files):
        data_array = np.zeros((len(oxts_files), 30))
        for i, filename in enumerate(oxts_files):
            with open(filename, 'rt') as f:
                vector = f.readline()
                data_array[i, :] = np.array(list(map(float, vector.strip().split(" "))))
        return pd.DataFrame(data_array, columns=self.fields)

    @staticmethod
    def load_timestamps(path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(path, 'oxts', 'timestamps.txt')
        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps give nanoseconds, so need to truncate
                # last 4 characters to get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                T = 3600 * t.hour + 60 * t.minute + t.second + t.microsecond / 1e6
                timestamps.append(T)
        return timestamps

    @timming
    def select_clear(self, df):
        """
        Selecte the largest time periode where there is no jump time greater than the specified
        """
        print("\tSelect clean portion")
        t = df['time']
        diff_t = np.diff(t)
        diff_idx = np.where(diff_t > (self.maximum_sample_loss / self.frequency))[0]
        diff_idx = np.insert(diff_idx, 0, 0, axis=0)
        diff_idx = np.append(diff_idx, len(diff_t))
        argmax_diff_idx = np.argmax(np.diff(diff_idx))
        ids = diff_idx[argmax_diff_idx:argmax_diff_idx + 2]
        ids[0] += 1
        return df.iloc[ids[0]:ids[1], :]

    @staticmethod
    def create_folder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)


# #### - Main - #### #
if __name__ == '__main__':
    start_time = time.time()

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    random_seed = 34                                                                                                    # set random seed
    rng = np.random.default_rng(random_seed)                                                                            # Create a RNG with a fixed seed
    random.seed(random_seed)                                                                                            # Set the Python seed

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seq", type=int, required=True, help="Length sequence of data. Ex: --seq 2000")
    parser.add_argument("-b", "--batch", type=int, required=True, help="Batch size for training. Ex: --batch 32")

    args = parser.parse_args()
    SEQ_LEN = args.seq
    BATCH_SIZE = args.batch

    path_raw_data = "../data/raw"
    path_processed_data = "../data/processed"
    kitti_dataset = BaseDataset(path_raw_data, path_processed_data, SEQ_LEN, BATCH_SIZE, maximum_sample_loss=5)
    kitti_dataset.create_dataset(h5_name="dataset.h5")  # bypass_date='2011_09_30') , bypass_drive='2011_09_30_drive_0020_extract')

    cprint(f"AJOUTER LA PROPORTION DE LONGUEUR GARDÉE", 'red')

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

