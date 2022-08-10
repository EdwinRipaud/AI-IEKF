from scipy.spatial.transform import Rotation as scipyRot
from termcolor import cprint
import numpy as np
import torch
import random
import time

np.set_printoptions(precision=2)


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


def set_parameters(original_class):
    orig_init = original_class.__init__

    def __init__(self, *args, **kwargs):
        self.filter_param = Parameters()
        attr_list = [a for a in dir(self.filter_param) if not a.startswith('__')
                     and not callable(getattr(self.filter_param, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_param, attr))

        orig_init(self, *args, **kwargs)

    original_class.__init__ = __init__
    return original_class


# #### - Class - #### #
class Parameters:
    g = torch.Tensor([0, 0, -9.80665])  # gravity vector

    P_dim = 21                      # covariance dimension

    # Process noise covariance
    Q_dim = 18                      # process noise covariance dimension

    cov_omega = 1e-3                # gyro covariance
    cov_acc = 1e-2                  # accelerometer covariance
    cov_b_omega = 6e-9              # gyro bias covariance
    cov_b_acc = 2e-4                # accelerometer bias covariance
    cov_Rot_c_i = 1e-9              # car to IMU orientation covariance
    cov_t_c_i = 1e-9                # car to IMU translation covariance

    # Pseudo-measurment covariance
    var_lat = 0.2                 # Zero lateral velocity variance
    beta_lat = 3                    # scale factor var_lat for covariance
    var_up = 300                  # Zero upward velocity variance
    beta_up = 3                     # scale factor var_up for covariance

    # State covariance
    cov_Rot0 = 1e-3                 # initial pitch and roll covariance
    cov_b_omega0 = 6e-3             # initial gyro bias covariance
    cov_b_acc0 = 4e-3               # initial accelerometer bias covariance
    cov_v0 = 1e-1                   # initial velocity covariance
    cov_Rot_c_i0 = 1e-6             # initial car to IMU pitch and roll covariance
    cov_t_c_i0 = 5e-3               # initial car to IMU translation covariance

    # numerical parameters
    n_normalize_rot = 100           # timestamp before normalizing orientation
    n_normalize_rot_c_i = 1000      # timestamp before normalizing car to IMU orientation

    def __init__(self, **kwargs):
        pass


@set_parameters  # decorate the class with the class attributes of the Class Parameters
class IEKF:
    Id2 = torch.eye(2)
    Id3 = torch.eye(3)
    Id6 = torch.eye(6)
    IdP = torch.eye(21)

    def __init__(self):
        super(IEKF, self).__init__()
        # Check if attributes of class Parameters have been set in the IEKF class
        # for key, val in self.__dict__.items():
        #     print(key, type(val))

        self.Q = torch.diag(torch.Tensor([self.cov_omega,       self.cov_omega,     self. cov_omega,         # Set the state noise matix
                                          self.cov_acc,         self.cov_acc,       self.cov_acc,
                                          self.cov_b_omega,     self.cov_b_omega,   self.cov_b_omega,
                                          self.cov_b_acc,       self.cov_b_acc,     self.cov_b_acc,
                                          self.cov_Rot_c_i,     self.cov_Rot_c_i,   self.cov_Rot_c_i,
                                          self.cov_t_c_i,       self.cov_t_c_i,     self.cov_t_c_i]))

    @timming
    def run(self, t, u, z_covs, v_mes0, ang0):
        """
        Run IEKF algorithm on input sequence
        :param t: time vector
        :param u: input measurement, u = [ax, ay, az, wx, wy, wz]
        :param z_covs: pseudo-measure covariance, [cov_v_lat, cov_v_up]
        :param v_0: initial velocity
        :param ang0: initial orientation
        :return:
        """

        t = torch.Tensor(t).cpu() if type(t).__module__ == np.__name__ else t.cpu()
        u = torch.Tensor(u).cpu() if type(u).__module__ == np.__name__ else u.cpu()
        z_covs = torch.Tensor(z_covs).cpu() if type(z_covs).__module__ == np.__name__ else z_covs.cpu()
        v_mes0 = torch.Tensor(v_mes0).cpu() if type(v_mes0).__module__ == np.__name__ else v_mes0.cpu()

        dt = t[1:] - t[:-1]                                                                                             # Set the delta time vector, that contain the delta of time for each sample
        N = u.shape[0]                                                                                                  # Sequence length
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(v_mes0, ang0, N)                                   # Initialise the states variables with initial condictions
        measurements_covs = self.z_to_cov(z_covs)

        for i in range(1, N):
            Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i = \
                self.propagate(Rot[i - 1], v[i - 1], p[i - 1], b_omega[i - 1], b_acc[i - 1], Rot_c_i[i - 1], t_c_i[i - 1], P, u[i], dt[i - 1])

            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.update(Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i, u[i], measurements_covs[i])
            # correct numerical error every second
            if i % self.n_normalize_rot == 0:
                Rot[i] = self.normalize_rot(Rot[i])
            # correct numerical error every 10 seconds
            if i % self.n_normalize_rot_c_i == 0:
                Rot_c_i[i] = self.normalize_rot(Rot_c_i[i])

        Rot = Rot.numpy() if type(Rot).__module__ == torch.__name__ else Rot
        v = v.numpy() if type(v).__module__ == torch.__name__ else v
        p = p.numpy() if type(p).__module__ == torch.__name__ else p
        b_omega = b_omega.numpy() if type(b_omega).__module__ == torch.__name__ else b_omega
        b_acc = b_acc.numpy() if type(b_acc).__module__ == torch.__name__ else b_acc
        Rot_c_i = Rot_c_i.numpy() if type(Rot_c_i).__module__ == torch.__name__ else Rot_c_i
        t_c_i = t_c_i.numpy() if type(t_c_i).__module__ == torch.__name__ else t_c_i
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def train_run(self, t, u, z_covs, v_mes0, ang0):
        """
        Run IEKF algorithm on input sequence
        :param t: time vector
        :param u: input measurement, u = [ax, ay, az, wx, wy, wz]
        :param z_covs: pseudo-measure covariance, [cov_v_lat, cov_v_up]
        :param v_mes0: initial velocity
        :param ang0: initial orientation
        :return:
        """

        t = torch.Tensor(t).cpu() if type(t).__module__ == np.__name__ else t.cpu()
        u = torch.Tensor(u).cpu() if type(u).__module__ == np.__name__ else u.cpu()
        z_covs = torch.Tensor(z_covs).cpu() if type(z_covs).__module__ == np.__name__ else z_covs.cpu()
        v_mes0 = torch.Tensor(v_mes0).cpu() if type(v_mes0).__module__ == np.__name__ else v_mes0.cpu()

        dt = t[1:] - t[:-1]                                                                                             # Set the delta time vector, that contain the delta of time for each sample
        N = u.shape[0]                                                                                                  # Sequence length
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(v_mes0, ang0, N)                                   # Initialise the states variables with initial condictions
        measurements_covs = self.z_to_cov(z_covs)

        for i in range(1, N):
            Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i = \
                self.propagate(Rot[i - 1], v[i - 1], p[i - 1], b_omega[i - 1], b_acc[i - 1], Rot_c_i[i - 1], t_c_i[i - 1], P, u[i], dt[i - 1])

            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.update(Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i, u[i], measurements_covs[i])

        return Rot, p

    def init_run(self, v_mes0, ang0, N):
        Rot = torch.zeros((N, 3, 3))                                # Rotation matrix (orientation) in car-frame
        v = torch.zeros((N, 3))                                     # Velocity vector
        p = torch.zeros((N, 3))                                     # Position vector
        b_omega = torch.zeros((N, 3))                               # Angular speed biase
        b_acc = torch.zeros((N, 3))                                 # Acceleration biase
        Rot_c_i = torch.zeros((N, 3, 3))                            # Rotation matrix from car-frame to IMU-frame
        t_c_i = torch.zeros((N, 3))                                 # Translation vector fro car-frame to IMU-frame

        rot0 = scipyRot.from_euler('xyz', [ang0[0], ang0[1], ang0[2]])
        Rot[0] = torch.from_numpy(rot0.as_matrix())                 # Set initial car orientation
        v[0] = v_mes0                                               # Set initial velocity vector
        Rot_c_i[0] = torch.eye(3)                                   # Set initial rotation between car and IMU frame to 0, i.e. identity matrix

        # cf: equation 8.18, parti 5.2, chapter 8 of the these "Deep learning, inertial measurements units, and odometry : some modern prototyping techniques for navigation based on multi-sensor fusion", by Martin BROSSARD & al.
        P = torch.zeros((self.P_dim, self.P_dim)).cpu()
        P[:2, :2] = self.cov_Rot0 * self.Id2                        # Set initial orientation covariance, with no error on initial yaw value
        P[3:5, 3:5] = self.cov_v0 * self.Id2                        # Set initial velocity (x and y) covariance
        P[5:9, 5:9] = torch.zeros((4, 4)).cpu()                     # Set initial z velocity and position covariance to 0
        P[9:12, 9:12] = self.cov_b_omega0 * self.Id3                # Set initial angular speed biase covariance
        P[12:15, 12:15] = self.cov_b_acc0 * self.Id3                # Set initial acceleration biase covariance
        P[15:18, 15:18] = self.cov_Rot_c_i0 * self.Id3              # Set initial rotation car to IMU frame covariance
        P[18:21, 18:21] = self.cov_t_c_i0 * self.Id3                # Set initial translation car to IMU frame covariance
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def z_to_cov(self, z):
        cov = torch.zeros_like(z)
        cov[:, 0] = self.var_lat**2 * 10**(self.beta_lat*z[:, 0])
        cov[:, 1] = self.var_up**2 * 10**(self.beta_up*z[:, 1])
        return cov

    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, Rot_c_i_prev, t_c_i_prev, P_prev, u, dt):
        # Propagate the state with the non-linear equations
        Rot_prev = Rot_prev.clone()
        acc_b = u[3:6] - b_acc_prev
        acc = Rot_prev.mv(acc_b) + self.g
        v = v_prev + acc * dt
        p = p_prev + v_prev.clone() * dt + 1 / 2 * acc * dt ** 2

        omega = (u[:3] - b_omega_prev) * dt
        Rot = Rot_prev.mm(self.so3exp(omega))

        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev.clone()
        t_c_i = t_c_i_prev

        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u, dt)
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov(self, P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u, dt):
        F = torch.zeros((self.P_dim, self.P_dim))
        G = torch.zeros((self.P_dim, self.Q_dim))
        Q = self.Q.clone()
        v_skew_rot = self.skew(v_prev).mm(Rot_prev)
        p_skew_rot = self.skew(p_prev).mm(Rot_prev)

        # Fill F matrix, the jacobian of f(.) with respect to X
        F[:3, 9:12] = -Rot_prev
        F[3:6, :3] = self.skew(self.g)
        F[3:6, 9:12] = -v_skew_rot
        F[3:6, 12:15] = -Rot_prev
        F[6:9, 3:6] = self.Id3
        F[6:9, 9:12] = -p_skew_rot

        # Fill G matrix, the jacobian of f(.) with respect to u
        G[:3, :3] = Rot_prev
        G[3:6, :3] = v_skew_rot
        G[3:6, 3:6] = Rot_prev
        G[6:9, :3] = p_skew_rot
        G[9:15, 6:12] = self.Id6
        G[15:18, 12:15] = self.Id3
        G[18:21, 15:18] = self.Id3

        F = F * dt
        G = G * dt
        F_square = F @ F
        F_cube = F_square @ F
        Phi = self.IdP + F + 1/2*F_square + 1/6*F_cube
        P = Phi.mm(P_prev + G.mm(Q).mm(G.t())).mm(Phi.t())
        return P

    def update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, measurement_cov):
        Rot_body = Rot.mm(Rot_c_i)                                                                                     # orientation of body frame
        v_imu = Rot.t().mv(v)                                                                                            # velocity in imu frame
        v_body = Rot_c_i.t().mv(v_imu)                                                                                   # velocity in body frame
        v_body += self.skew(t_c_i).mv((u[:3] - b_omega))                                                                 # velocity in body frame in the vehicle axis
        Omega = self.skew(u[:3] - b_omega)                                                                              # Anguar velocity correction and transform into delta roation matrix

        # Jacobian w.r.t. car frame
        H_v_imu = Rot_c_i.t().mm(self.skew(v_imu))
        H_t_c_i = -self.skew(t_c_i)

        H = torch.zeros((2, self.P_dim)).cpu()
        H[:, 3:6] = Rot_body.t()[1:]
        H[:, 15:18] = H_v_imu[1:]
        H[:, 9:12] = H_t_c_i[1:]
        H[:, 18:21] = -Omega[1:]
        r = - v_body[1:]
        R = torch.diag(measurement_cov)

        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = self.state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R)
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    def state_and_cov_update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
        S = H.mm(P).mm(H.t()) + R
        K = (torch.linalg.solve(S, (P.mm(H.t())).t())).t()
        dx = K.mv(r)

        dR, dxi = self.sen3exp(dx[:9])
        dv = dxi[:, 0]
        dp = dxi[:, 1]
        Rot_up = dR.mm(Rot)
        v_up = dR.mv(v) + dv
        p_up = dR.mv(p) + dp

        b_omega_up = b_omega + dx[9:12]
        b_acc_up = b_acc + dx[12:15]

        dR = self.so3exp(dx[15:18])
        Rot_c_i_up = dR.mm(Rot_c_i)
        t_c_i_up = t_c_i + dx[18:21]

        I_KH = self.IdP - K.mm(H)
        P_up = I_KH.mm(P).mm(I_KH.t()) + K.mm(R).mm(K.t())
        P_up = (P_up + P_up.t())/2
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    def sen3exp(self, xi):
        phi = xi[:3]
        angle = torch.linalg.norm(phi)
        # Near |phi|==0, use first order Taylor expansion
        if torch.abs(angle) < 1e-8:
            skew_phi = self.skew(phi)
            J = self.Id3 + 0.5 * skew_phi
            Rot = self.Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = self.skew(axis)
            s = torch.sin(angle)
            c = torch.cos(angle)
            J = (s / angle) * self.Id3 + (1 - s / angle) * torch.outer(axis, axis) + ((1 - c) / angle) * skew_axis
            Rot = c * self.Id3 + (1 - c) * torch.outer(axis, axis) + s * skew_axis
        x = J.mm(xi[3:].view(-1, 3).t())
        return Rot, x

    def so3exp(self, phi):
        angle = torch.linalg.norm(phi)
        # Near phi==0, use first order Taylor expansion
        if torch.abs(angle) < 1e-8:
            skew_phi = self.skew(phi)
            return self.Id3 + skew_phi
        axis = phi / angle
        skew_axis = self.skew(axis)
        s = torch.sin(angle)
        c = torch.cos(angle)
        return c * self.Id3 + (1 - c) * torch.outer(axis, axis) + s * skew_axis

    @staticmethod
    def skew(x):
        X = torch.Tensor([[0, -x[2], x[1]],
                          [x[2], 0, -x[0]],
                          [-x[1], x[0], 0]])
        return X

    @staticmethod
    def normalize_rot(rot):
        # U, S, V = torch.svd(A) returns the singular value
        # decomposition of a real matrix A of size (n x m) such that A=USV′.
        # Irrespective of the original strides, the returned matrix U will
        # be transposed, i.e. with strides (1, n) instead of (n, 1).

        # pytorch SVD seems to be inaccurate, so just move to numpy immediately
        U, _, V = torch.svd(rot)
        S = torch.eye(3)
        S[2, 2] = torch.det(U) * torch.det(V)
        return U.mm(S).mm(V.t())


# #### - Main - #### #
if __name__ == '__main__':
    start_time = time.time()

    random_seed = 34                                                                                                    # set random seed
    rng = np.random.default_rng(random_seed)                                                                            # Create a RNG with a fixed seed
    random.seed(random_seed)                                                                                            # Set the Python seed

    iekf_filter = IEKF()

    import matplotlib.pyplot as plt
    import pandas as pd
    import h5py

    save_path = "../data/processed/dataset.h5"                                      # Path to the .h5 dataset

    hdf = h5py.File(save_path, 'r')                                                 # Read the .h5 file

    # check the dataset architecture
    def get_all(name):
        if not ("/axis" in name or "/block" in name or "/index" in name or "/value" in name):
            print(name)
    # print(hdf.visit(get_all))

    date = "day_2011_09_30_drive_0033_extract"
    split = "validation"
    # date = "day_2011_09_30_drive_0018_extract"
    # split = "train"

    # Load dataframes
    dataset = pd.read_hdf(save_path, f"full_datset/{date}")   # Get the input DataFrame for the given date and drive
    u_df = dataset[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].copy()
    time_df = dataset[['time']].copy()
    X_gt = dataset[['pose_x']].values
    Y_gt = dataset[['pose_y']].values

    # Export the values to an np.array
    u = u_df.values
    t = time_df.values
    v_mes0 = dataset[["ve", "vn", "vu"]].copy().iloc[0, :].values
    ang0 = dataset[["roll", "pitch", "yaw"]].copy().iloc[0, :].values

    print(f"Initial conditions:\n\tvelocity: {v_mes0}\n\torientation: {ang0}")

    Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf_filter.run(t, u, np.ones((t.shape[0], 1))@[[1., -0.5]], v_mes0, ang0)

    plt.figure()
    plt.plot(X_gt, Y_gt, 'k-')
    plt.plot(p[:, 0], p[:, 1], 'g-.')
    plt.axis('equal')

    import utils
    ground_t = utils.df_to_PosePath3D(dataset['rot_matrix'].values, dataset[['pose_x', 'pose_y', 'pose_z']].values)
    kalman_t = utils.df_to_PosePath3D(Rot, p)
    print(utils.get_APE(ground_t, kalman_t).get_all_statistics())
    utils.plot_APE(ground_t, kalman_t)

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

    plt.show()
