import argparse
from pathlib import Path

import unitpy

from safe_barrier_envs.car import make
from safe_barrier_envs.utils import get_default_params


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_speed", required=True, type=float, help="miles per hour")
    args = parser.parse_args()

    log_dir = Path("/tmp/car")

    params = get_default_params()["car"]
    params["env"]["do_render"] = True
    params["viewer"]["render_policy"] = False
    env_vars = make(
        use_scalar=True,  # optimization for floats, np arrays and torch tensors
        allow_np_cast=True,  # optimization for floats, np arrays and torch tensors
        allow_gui=True,
        log_dir=log_dir,  # this is where videos will be output if a gui is used
        log_on_unsafe=False,
        params=params,
    )

    env = env_vars.env
    ctrl = env_vars.safe_controller
    tgt_speed_meters_per_sec = (
        (args.tgt_speed * unitpy.Unit("mile per hour")).to("m/s").value
    )
    ctrl.ctrl_dbl_int.setpoint = tgt_speed_meters_per_sec

    # typically in RL the observation is a normalized version
    # of the state. A non-RL controller does not need this
    # normalized state so rather than denormalizing so that the
    # non-RL controller can run, the environment also provides
    # the unnormalized observation in the info dict with
    # key "safe_controller_obs"
    info = env.reset()[1]
    obs = info["safe_controller_obs"]

    while True:
        action = ctrl.get_control(obs)

        # the environment expects a normalized action between -1 and 1
        action = env.normalize(action)

        done, info = env.step(action)[2:]
        obs = info["safe_controller_obs"]

        if done:
            info = env.reset()[1]
            obs = info["safe_controller_obs"]


if __name__ == "__main__":
    main()
