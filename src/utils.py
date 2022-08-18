from evo.core.trajectory import PosePath3D
from evo.core import lie_algebra as lie
import matplotlib.pyplot as plt
from evo.core import metrics
from termcolor import cprint
from evo.tools import plot
import numpy as np
import torch
import copy
import time
import evo


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
@timming
def df_to_PosePath3D(rot, pose):
    se3_pose = []
    for i in range(pose.shape[0]):
        se3_pose.append(lie.se3(rot[i], pose[i, :]))
    return PosePath3D(poses_se3=se3_pose)


@timming
def get_APE(ground_truth, kalman):
    """
    Get Absolute Positional Error of a 'Kalman' trajectory, from a 'Ground-truth' trajectory
    :param ground_truth: evo.core.trajectory.PosePath3D or a tuple composed of: (rotation_matrix, position) -> type(ndarray() or list(ndarray), ndarray() or list(ndarray))
    :param kalman: evo.core.trajectory.PosePath3D or a tuple composed of: (rotation_matrix, position) -> type(ndarray() or list(ndarray), ndarray() or list(ndarray))
    :return: statistics of the Absolute Positional Error
    """
    if type(ground_truth) == tuple:
        gt_pose3D = df_to_PosePath3D(ground_truth[0], ground_truth[1])
    elif type(ground_truth) == evo.core.trajectory.PosePath3D:
        gt_pose3D = ground_truth
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    if type(ground_truth) == tuple:
        kf_pose3D = df_to_PosePath3D(kalman[0], kalman[1])
    elif type(ground_truth) == evo.core.trajectory.PosePath3D:
        kf_pose3D = kalman
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    ape_metric = metrics.APE(pose_relation=metrics.PoseRelation.full_transformation)
    ape_metric.process_data((gt_pose3D, kf_pose3D))
    return ape_metric


def plot_APE(ground_truth, kalman, cumulative=False, corr_edges=False):
    """
    Plot APE error and trajectory compare to a ground-truth one.
    :param ground_truth: PosePath3D or tuple(rotation matrix, pose)
    :param kalman: PosePath3D or tuple(rotation matrix, pose)
    :param cumulative: Boolean
    :param corr_edges: Boolean
    :return:
    """
    if type(ground_truth) == tuple:
        gt_pose3D = df_to_PosePath3D(ground_truth[0], ground_truth[1])
        N = ground_truth[1].shape[0]
    elif type(ground_truth) == evo.core.trajectory.PosePath3D:
        gt_pose3D = ground_truth
        N = ground_truth.num_poses
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    if type(kalman) == tuple:
        kf_pose3D = df_to_PosePath3D(kalman[0], kalman[1])
    elif type(kalman) == evo.core.trajectory.PosePath3D:
        kf_pose3D = kalman
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    ape_metric = get_APE(gt_pose3D, kf_pose3D)
    ape_stats = ape_metric.get_all_statistics()
    second_from_start = np.linspace(0, N/100, N)

    fig = plt.figure()
    plot.error_array(fig.gca(), ape_metric.error, x_array=second_from_start, cumulative=cumulative,
                     statistics={s: v for s, v in ape_stats.items() if s != "sse"},
                     name="APE", xlabel="$t$ (s)",
                     title=f"{'Cumulative ' if cumulative else ''}APE w.r.t {ape_metric.pose_relation.value}")

    plot_mode = plot.PlotMode.xy
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, gt_pose3D, '--', "grey", "reference")
    plot.traj_colormap(ax, kf_pose3D, ape_metric.error, plot_mode, min_map=ape_stats['min'], max_map=ape_stats['max'])
    if corr_edges:
        N = kf_pose3D.get_infos()["nr. of poses"]
        cprint("NotImplementedError: PosePath3D object has no attribute 'num_poses'", 'red')
        # plot.draw_correspondence_edges(ax, kf_pose3D.reduce_to_ids([i for i in range(0, N, int(N/100))]),
        #                                gt_pose3D.reduce_to_ids([i for i in range(0, N, int(N/100))]), plot_mode)
    ax.legend()
    return


def plot_multiple(ground_truth, kalmans):
    """
    Compare different trajectories on plot versus a ground-truth one.
    :param ground_truth: PosePath3D or tuple(rotation matrix, pose)
    :param kalmans: list[PosePath3D] or list[tuple(rotation matrix, pose)]
    :return:
    """
    if type(ground_truth) == tuple:
        gt_pose3D = df_to_PosePath3D(ground_truth[0], ground_truth[1])
    elif type(ground_truth) == evo.core.trajectory.PosePath3D:
        gt_pose3D = ground_truth
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    kf_pose3D = []
    kf_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for i in range(len(kalmans)):
        if type(kalmans[i]) == tuple:
            kf_pose3D.append(df_to_PosePath3D(kalmans[i][0], kalmans[i][1]))
        elif type(kalmans[i]) == evo.core.trajectory.PosePath3D:
            kf_pose3D.append(kalmans[i])
        else:
            raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    plot_mode = plot.PlotMode.xy
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, gt_pose3D, '-', "black", "reference")
    for k in range(len(kf_pose3D)):
        plot.traj(ax, plot_mode, kf_pose3D[k], '--', kf_colors[k % len(kf_colors)], f"Kalman {k+1}")
    ax.legend()
    return


def get_RPE(ground_truth, kalman):
    """
    Get Relative Positional Error of a 'Kalman' trajectory, from a 'Ground-truth' trajectory
    :param ground_truth: evo.core.trajectory.PosePath3D or a tuple composed of: (rotation_matrix, position) -> type(ndarray() or list(ndarray), ndarray() or list(ndarray))
    :param kalman: evo.core.trajectory.PosePath3D or a tuple composed of: (rotation_matrix, position) -> type(ndarray() or list(ndarray), ndarray() or list(ndarray))
    :return: statistics of the Relative Positional Error
    """
    if type(ground_truth) == tuple:
        gt_pose3D = df_to_PosePath3D(ground_truth[0], ground_truth[1])
    elif type(ground_truth) == evo.core.trajectory.PosePath3D:
        gt_pose3D = ground_truth
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    if type(ground_truth) == tuple:
        kf_pose3D = df_to_PosePath3D(kalman[0], kalman[1])
    elif type(ground_truth) == evo.core.trajectory.PosePath3D:
        kf_pose3D = kalman
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    rpe_metric = metrics.RPE(pose_relation=metrics.PoseRelation.translation_part,
                             delta=10, delta_unit=metrics.Unit.frames,
                             all_pairs=False)
    rpe_metric.process_data((gt_pose3D, kf_pose3D))
    return rpe_metric


def plot_RPE(ground_truth, kalman, cumulative=False, corr_edges=False):
    """
    Plot APE error and trajectory compare to a ground-truth one.
    :param ground_truth: PosePath3D or tuple(rotation matrix, pose)
    :param kalman: PosePath3D or tuple(rotation matrix, pose)
    :param cumulative: Boolean
    :param corr_edges: Boolean
    :return:
    """
    if type(ground_truth) == tuple:
        gt_pose3D = df_to_PosePath3D(ground_truth[0], ground_truth[1])
        N = ground_truth[1].shape[0]
    elif type(ground_truth) == evo.core.trajectory.PosePath3D:
        gt_pose3D = ground_truth
        N = ground_truth.num_poses
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    if type(kalman) == tuple:
        kf_pose3D = df_to_PosePath3D(kalman[0], kalman[1])
    elif type(kalman) == evo.core.trajectory.PosePath3D:
        kf_pose3D = kalman
    else:
        raise TypeError("Bad type input, needs to be a tuple or evo.core.trajectory.PosePath3D")

    rpe_metric = get_RPE(gt_pose3D, kf_pose3D)
    rpe_stats = rpe_metric.get_all_statistics()
    second_from_start = np.linspace(0, N/100, N)[::10][:len(rpe_metric.error)-1]

    fig = plt.figure()
    plot.error_array(fig.gca(), rpe_metric.error, x_array=second_from_start, cumulative=cumulative,
                     statistics={s: v for s, v in rpe_stats.items() if s != "sse"},
                     name="APE", xlabel="$t$ (s)",
                     title=f"{'Cumulative ' if cumulative else ''}RPE w.r.t {rpe_metric.pose_relation.value}")

    traj_ref_plot = copy.deepcopy(gt_pose3D)
    traj_est_plot = copy.deepcopy(kf_pose3D)
    traj_ref_plot.reduce_to_ids(rpe_metric.delta_ids)
    traj_est_plot.reduce_to_ids(rpe_metric.delta_ids)

    plot_mode = plot.PlotMode.xy
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref_plot, '--', "grey", "reference")
    plot.traj_colormap(ax, traj_est_plot, rpe_metric.error, plot_mode, min_map=rpe_stats['min'], max_map=rpe_stats['max'])
    if corr_edges:
        N = traj_est_plot.get_infos()["nr. of poses"]
        cprint("NotImplementedError: PosePath3D object has no attribute 'num_poses'", 'red')
        # plot.draw_correspondence_edges(ax, traj_est_plot.reduce_to_ids([i for i in range(0, N, int(N/100))]),
        #                                traj_ref_plot.reduce_to_ids([i for i in range(0, N, int(N/100))]), plot_mode)
    ax.legend()
    return


def torch_full_transformation_rmse(gt_rot_p, kf_rot_p):
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
        E = []
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
            E.append(error)
        squered_error = torch.pow(torch.tensor(E), 2)
        return torch.sqrt(torch.mean(squered_error)) + 1e-8


# #### - Main - #### #
if __name__ == '__main__':
    print("utils.py")
