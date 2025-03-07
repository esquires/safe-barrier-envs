import argparse
import time
from pathlib import Path

import numpy as np
import unitpy
from PIL import Image

from safe_barrier_envs.car import make
from safe_barrier_envs.utils import get_default_params


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    np.random.seed(3)
    log_dir = Path("/tmp/car")

    params = get_default_params()["car"]

    vp = params["viewer"]

    vp["lane_dist_text_enable"] = False
    vp["lane_dist_line_enable"] = False
    vp["car_sep_text_enable"] = False
    vp["car_sep_line_enable"] = False
    vp["vel_line_components"] = ["c"]
    vp["vel_text_components"] = []
    vp["vel_y"] = 0.20
    vp["vel_x"] = 0.08
    vp["vel_height"] = 0.20
    vp["time_enable"] = False

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
    tgt_speed_meters_per_sec = (70 * unitpy.Unit("mile per hour")).to("m/s").value
    ctrl.ctrl_dbl_int.setpoint = tgt_speed_meters_per_sec

    # typically in RL the observation is a normalized version
    # of the state. A non-RL controller does not need this
    # normalized state so rather than denormalizing so that the
    # non-RL controller can run, the environment also provides
    # the unnormalized observation in the info dict with
    # key "safe_controller_obs"
    info = env.reset()[1]

    env.states.a = np.array([150, 1.8, 22, 0])
    env.states.b = np.array([220, 5.4, 27, 0])
    env.states.c = np.array([0, 5.4, 25, np.radians(-2)])

    obs = info["safe_controller_obs"]
    action = ctrl.get_control(obs)
    action = env.normalize(action)
    env.step(action)

    if args.output:
        assert env.viewer is not None
        img = env.viewer.viewer.get_img()
        img = img[250:450, 100:1100, :]
        print(f"saving to {args.output}")
        Image.fromarray(img).save(args.output)
    else:
        print("enter ctrl-c to end")
        while True:
            time.sleep(0.01)


if __name__ == "__main__":
    main()
