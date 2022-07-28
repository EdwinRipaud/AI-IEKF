from scipy.spatial.transform import Rotation as scipyRot
from termcolor import cprint
import numpy as np
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
    g = np.array([0, 0, -9.80665])  # gravity vector

    P_dim = 21                      # covariance dimension

    Q_dim = 18                      # process noise covariance dimension

    # Process noise covariance
    cov_omega = 1e-3                # gyro covariance
    cov_acc = 1e-2                  # accelerometer covariance
    cov_b_omega = 6e-9              # gyro bias covariance
    cov_b_acc = 2e-4                # accelerometer bias covariance
    cov_Rot_c_i = 1e-9              # car to IMU orientation covariance
    cov_t_c_i = 1e-9                # car to IMU translation covariance

    cov_lat = 0.2                   # Zero lateral velocity covariance
    cov_up = 300                    # Zero upward velocity covariance

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
    Id2 = np.eye(2)
    Id3 = np.eye(3)
    Id6 = np.eye(6)
    IdP = np.eye(21)

    def __init__(self):
        super(IEKF, self).__init__()
        # Check if attributes of class Parameters have been set in the IEKF class
        # for key, val in self.__dict__.items():
        #     print(key, type(val))

        self.Q = np.diag([self.cov_omega,       self.cov_omega,     self. cov_omega,                                              # Set the state noise matix
                          self.cov_acc,         self.cov_acc,       self.cov_acc,
                          self.cov_b_omega,     self.cov_b_omega,   self.cov_b_omega,
                          self.cov_b_acc,       self.cov_b_acc,     self.cov_b_acc,
                          self.cov_Rot_c_i,     self.cov_Rot_c_i,   self.cov_Rot_c_i,
                          self.cov_t_c_i,       self.cov_t_c_i,     self.cov_t_c_i])

    def run(self, t, u, measurements_covs, v_mes0, ang0):
        """
        Run IEKF algorithm on input sequence
        :param t: time vector
        :param u: input measurement, u = [ax, ay, az, wx, wy, wz]
        :param measurements_covs: pseudo-measure covariance, [cov_v_lat, cov_v_up]
        :param v_mes0: initial velocity
        :param ang0: initial orientation
        :return:
        """
        dt = t[1:] - t[:-1]                                                                                             # Set the delta time vector, that contain the delta of time for each sample
        N = u.shape[0]                                                                                                  # Sequence length
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(v_mes0, ang0, N)                                    # Initialise the states variables with initial condictions

        for i in range(1, N):
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = self.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1], Rot_c_i[i-1], t_c_i[i-1], P, u[i], dt[i-1])

            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = self.update(Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P, u[i], measurements_covs[i])
            # correct numerical error every second
            if i % self.n_normalize_rot == 0:
                Rot[i] = self.normalize_rot(Rot[i])
            # correct numerical error every 10 seconds
            if i % self.n_normalize_rot_c_i == 0:
                Rot_c_i[i] = self.normalize_rot(Rot_c_i[i])
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_run(self, v_mes0, ang0, N):
        Rot = np.zeros((N, 3, 3))                               # Rotation matrix (orientation) in car-frame
        v = np.zeros((N, 3))                                    # Velocity vector
        p = np.zeros((N, 3))                                    # Position vector
        b_omega = np.zeros((N, 3))                              # Angular speed biase
        b_acc = np.zeros((N, 3))                                # Acceleration biase
        Rot_c_i = np.zeros((N, 3, 3))                           # Rotation matrix from car-frame to IMU-frame
        t_c_i = np.zeros((N, 3))                                # Translation vector fro car-frame to IMU-frame

        # Rot[0] = self.from_rpy(ang0[0], ang0[1], ang0[2])       # Set initial car orientation
        rot0 = scipyRot.from_euler('xyz', [ang0[0], ang0[1], ang0[2]])
        Rot[0] = rot0.as_matrix()                                  # Set initial car orientation
        v[0] = v_mes0                                           # Set initial velocity vector
        Rot_c_i[0] = np.eye(3)                                  # Set initial rotation between car and IMU frame to 0, i.e. identity matrix

        # cf: equation 8.18, parti 5.2, chapter 8 of the these "Deep learning, inertial measurements units, and odometry : some modern prototyping techniques for navigation based on multi-sensor fusion", by Martin BROSSARD
        P = np.zeros((self.P_dim, self.P_dim))
        P[:2, :2] = self.cov_Rot0 * self.Id2                    # Set initial orientation covariance, with no error on initial yaw value
        P[3:5, 3:5] = self.cov_v0 * self.Id2                    # Set initial velocity (x and y) covariance
        P[5:9, 5:9] = np.zeros((4, 4))                          # Set initial z velocity and position covariance to 0
        P[9:12, 9:12] = self.cov_b_omega0 * self.Id3            # Set initial angular speed biase covariance
        P[12:15, 12:15] = self.cov_b_acc0 * self.Id3            # Set initial acceleration biase covariance
        P[15:18, 15:18] = self.cov_Rot_c_i0 * self.Id3          # Set initial rotation car to IMU frame covariance
        P[18:21, 18:21] = self.cov_t_c_i0 * self.Id3            # Set initial translation car to IMU frame covariance
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, Rot_c_i_prev, t_c_i_prev, P_prev, u, dt):
        # Propagate the state with the non-linear equations
        acc = Rot_prev @ (u[3:6] - b_acc_prev) + self.g
        v = v_prev + acc * dt
        p = p_prev + v_prev*dt + 1/2 * acc * dt**2
        omega = u[:3] - b_omega_prev
        Rot = Rot_prev @ self.so3exp(omega * dt)
        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev
        t_c_i = t_c_i_prev

        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u, dt)
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov(self, P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u, dt):
        F = np.zeros((self.P_dim, self.P_dim))
        G = np.zeros((self.P_dim, self.Q_dim))
        v_skew_rot = self.skew(v_prev) @ Rot_prev
        p_skew_rot = self.skew(p_prev) @ Rot_prev

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
        F_square = F.dot(F)
        F_cube = F_square.dot(F)
        Phi = self.IdP + F + 1/2*F_square + 1/6*F_cube
        P = Phi @ (P_prev + G @ self.Q @ G.T) @ Phi.T
        return P

    def update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, measurement_cov):
        Rot_body = Rot.dot(Rot_c_i)                                                                                     # orientation of body frame
        v_imu = Rot.T.dot(v)                                                                                            # velocity in imu frame
        v_body = Rot_c_i.T.dot(v_imu)                                                                                   # velocity in body frame
        v_body += self.skew(t_c_i) @ (u[:3] - b_omega)                                                                 # velocity in body frame in the vehicle axis
        Omega = self.skew(u[:3] - b_omega)                                                                              # Anguar velocity correction and transform into delta roation matrix

        # Jacobian w.r.t. car frame
        H_v_imu = Rot_c_i.T.dot(self.skew(v_imu))
        H_t_c_i = -self.skew(t_c_i)

        H = np.zeros((2, self.P_dim))
        H[:, 3:6] = Rot_body.T[1:]
        H[:, 15:18] = H_v_imu[1:]
        H[:, 9:12] = H_t_c_i[1:]
        H[:, 18:21] = -Omega[1:]
        r = - v_body[1:]
        R = np.diag(measurement_cov)

        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = self.state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R)
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    def state_and_cov_update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
        S = H @ P @ H.T + R
        K = (np.linalg.solve(S, (P @ H.T).T)).T
        dx = K.dot(r)

        dR, dxi = self.sen3exp(dx[:9])
        dv = dxi[:, 0]
        dp = dxi[:, 1]
        Rot_up = dR.dot(Rot)
        v_up = dR.dot(v) + dv
        p_up = dR.dot(p) + dp

        b_omega_up = b_omega + dx[9:12]
        b_acc_up = b_acc + dx[12:15]

        dR = self.so3exp(dx[15:18])
        Rot_c_i_up = dR.dot(Rot_c_i)
        t_c_i_up = t_c_i + dx[18:21]

        I_KH = self.IdP - K @ H
        P_up = I_KH @ P @ I_KH.T + K @ R @ K.T
        P_up = (P_up + P_up.T)/2
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    def sen3exp(self, xi):
        phi = xi[:3]
        angle = np.linalg.norm(phi)
        # Near |phi|==0, use first order Taylor expansion
        if np.abs(angle) < 1e-8:
            skew_phi = self.skew(phi)
            J = self.Id3 + 0.5 * skew_phi
            Rot = self.Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = self.skew(axis)
            s = np.sin(angle)
            c = np.cos(angle)
            J = (s / angle) * self.Id3 + (1 - s / angle) * np.outer(axis, axis) + ((1 - c) / angle) * skew_axis
            Rot = c * self.Id3 + (1 - c) * np.outer(axis, axis) + s * skew_axis
        x = J @ xi[3:].reshape(-1, 3).T
        return Rot, x

    def so3exp(self, phi):
        angle = np.linalg.norm(phi)
        # Near phi==0, use first order Taylor expansion
        if np.abs(angle) < 1e-8:
            skew_phi = self.skew(phi)
            return self.Id3 + skew_phi
        axis = phi / angle
        skew_axis = self.skew(axis)
        s = np.sin(angle)
        c = np.cos(angle)
        return c * self.Id3 + (1 - c) * np.outer(axis, axis) + s * skew_axis

    @staticmethod
    def skew(x):
        X = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])
        return X

    @staticmethod
    def normalize_rot(Rot):
        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        U, _, V = np.linalg.svd(Rot, full_matrices=False)
        S = np.eye(3)
        S[2, 2] = np.linalg.det(U) * np.linalg.det(V)
        return U.dot(S).dot(V)


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
    hdf_key = list(hdf.keys())                                                      # Get the keys
    print(hdf_key)
    seq = hdf.get(list(hdf.keys())[0])                                              # Get the subfolder in the .h5 for the given date
    seq_key = list(seq.keys())                                                      # Get the keys for the the given date
    print(seq_key)
    seq_df = seq.get(list(seq.keys())[0])                                              # Get the subfolder in the .h5 for the given date
    seq_df_key = list(seq_df.keys())                                                      # Get the keys for the the given date
    print(seq_df_key)

    # Load dataframes
    dataset = pd.read_hdf(save_path, "day_2011_09_30/drive_0020_extract/dataset")   # Get the input DataFrame for the given date and drive
    u_df = pd.read_hdf(save_path, "day_2011_09_30/drive_0020_extract/w_a_input")    # Get the input DataFrame for the given date and drive
    time_df = pd.read_hdf(save_path, "day_2011_09_30/drive_0020_extract/time")      # Get the time vector DataFrame for the given date and drive
    ground_truth = pd.read_hdf(save_path, "day_2011_09_30/drive_0020_extract/ground_truth")      # Get the time vector DataFrame for the given date and drive
    X_gt = ground_truth[['x']].values
    Y_gt = ground_truth[['y']].values

    # Export the values to an np.array
    u = u_df.values
    t = time_df.values
    v_mes0 = dataset[["ve", "vn", "vu"]].copy().iloc[0, :].values
    ang0 = dataset[["roll", "pitch", "yaw"]].copy().iloc[0, :].values

    print(f"Initial conditions:\n\tvelocity: {v_mes0}\n\torientation: {ang0}")

    Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf_filter.run(t, u, np.ones((t.shape[0], 1))@[[500, 300]], v_mes0, ang0)

    plt.figure()
    plt.plot(X_gt, Y_gt, 'k-')
    plt.plot(p[:, 0], p[:, 1], 'g-.')
    plt.axis('equal')

    print(f"\n#####\nProgram run time: {round(time.time()-start_time, 1)} s\n#####")

    plt.show()
