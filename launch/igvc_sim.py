import os
import shlex
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def get_path(package, dir, file):
    return os.path.join(get_package_share_directory(package), dir, file)


def _locate_sim_script():
    env_override = os.environ.get("CEV_ACKERMANN_SIM_PATH")
    if env_override:
        override_path = Path(env_override)
        if override_path.is_file():
            return override_path
        raise FileNotFoundError(f"CEV_ACKERMANN_SIM_PATH points to '{env_override}', but it does not exist.")

    launch_file = Path(__file__).resolve()
    for parent in launch_file.parents:
        direct_candidate = parent / "cev-ackermann-sim" / "sim.py"
        if direct_candidate.exists():
            return direct_candidate
        src_candidate = parent / "src" / "cev-ackermann-sim" / "sim.py"
        if src_candidate.exists():
            return src_candidate
    raise FileNotFoundError("Unable to locate cev-ackermann-sim/sim.py relative to launch file.")


def _locate_workspace_setup():
    launch_file = Path(__file__).resolve()
    for parent in launch_file.parents:
        setup_candidate = parent / "install" / "setup.bash"
        if setup_candidate.exists():
            return setup_candidate
    raise FileNotFoundError("Unable to locate install/setup.bash relative to launch file.")


def generate_launch_description():
    sim_path = _locate_sim_script()
    setup_script = _locate_workspace_setup()

    cleanup_process = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            "pkill -f trajectory_follower_node || true; "
            "pkill -f planner_node || true; "
            "pkill -f cev-ackermann-sim/sim.py || true",
        ],
        output="screen",
    )

    planner_node = Node(
        package="cev_planner_ros2",
        executable="igvc_node",
        name="cev_planner_ros2_node",
        output="screen",
        parameters=[get_path("cev_planner_ros2", "config", "igvc.yaml")],
    )

    trajectory_node = Node(
        package="trajectory_follower",
        executable="trajectory_follower_node",
        name="trajectory_node",
    )

    simulation_process = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            f"source {shlex.quote(str(setup_script))} && python3 {shlex.quote(str(sim_path))}",
        ],
        cwd=str(sim_path.parent),
        output="screen",
    )

    return LaunchDescription(
        [
            cleanup_process,
            TimerAction(
                period=0.5,
                actions=[
                    planner_node,
                    TimerAction(
                        period=0.5,
                        actions=[
                            trajectory_node,
                            TimerAction(
                                period=2.5,
                                actions=[simulation_process],
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )
