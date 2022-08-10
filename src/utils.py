from evo.core.trajectory import PosePath3D
from evo.core import lie_algebra as lie
import matplotlib.pyplot as plt
from evo.core import metrics
from termcolor import cprint
from evo.tools import plot
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


def plot_APE(ground_truth, kalman, corr_edges=False):
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

    ape_metric = get_APE(gt_pose3D, kf_pose3D)
    ape_stats = ape_metric.get_all_statistics()

    fig = plt.figure()
    plot.error_array(fig.gca(), ape_metric.error,  # x_array=second_from_start,
                     statistics={s: v for s, v in ape_stats.items() if s != "sse"},
                     name="APE", title="APE w.r.t " + ape_metric.pose_relation.value, xlabel="$t$ (s)")

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


def plot_RPE(ground_truth, kalman, corr_edges=False):
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

    rpe_metric = get_RPE(gt_pose3D, kf_pose3D)
    rpe_stats = rpe_metric.get_all_statistics()

    fig = plt.figure()
    plot.error_array(fig.gca(), rpe_metric.error,  # x_array=second_from_start,
                     statistics={s: v for s, v in rpe_stats.items() if s != "sse"},
                     name="APE", title="APE w.r.t " + rpe_metric.pose_relation.value, xlabel="$t$ (s)")

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


# #### - Main - #### #
if __name__ == '__main__':
    print("utils.py")
