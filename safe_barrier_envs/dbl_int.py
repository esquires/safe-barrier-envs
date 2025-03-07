import copy
import dataclasses
from pathlib import Path
from typing import Any, Optional

import gym
import numpy as np

from safe_barrier_envs.utils import (
    ActuatorLimits,
    BarrierConstraint,
    EpisodeMonitor,
    MathCalcs,
    NpFloat,
    NumType,
    RewType,
    T,
    TimeTracker,
    get_default_params,
    make_box,
)

InfoDict = dict[str, Any]


class A_tilde:
    def __init__(self, a_minus: float, a_plus: Optional[float], calcs: MathCalcs):
        assert a_minus < 0
        if a_plus is not None:
            assert 0 < a_plus
        self.a_minus = a_minus
        self.a_plus = a_plus
        self.calcs = calcs

    def get_a(self, v: T, _type: Optional[NumType] = None) -> T | float:
        if self.a_plus is None:
            return self.a_minus

        v, v_cast = self.calcs.scalar_cast(v, _type)  # type: ignore
        if v_cast:
            return self.a_minus if v >= 0 else self.a_plus
        else:
            m = self.calcs.get_module(v, _type)
            a = m.where(v >= 0, self.a_minus, self.a_plus)
            return a  # type: ignore

    def __repr__(self) -> str:
        return (
            f"A_tilde(a_minus={self.a_minus}, a_plus={self.a_plus}, calcs={self.calcs})"
        )


class Rho:
    def __init__(self, calcs: MathCalcs):
        self.calcs = calcs

    def get_rho(self, state: T, _type: Optional[NumType] = None) -> T:
        return self.calcs.get_element(state, 1, 2, _type)[0]


class DblIntSafeController:
    def __init__(self, dt: float, a_tilde: A_tilde, setpoint: float, calcs: MathCalcs):
        self.dt = dt
        self.a_tilde = a_tilde
        self.setpoint = setpoint
        self.calcs = calcs

    def __repr__(self) -> str:
        return (
            f"DblIntSafeController("
            f"dt={self.dt}, "
            f"a_tilde={self.a_tilde}, "
            f"setpoint={self.setpoint}, "
            f"calcs={self.calcs}"
            ")"
        )

    def get_control(self, state: T, _type: Optional[NumType] = None) -> T:
        v, v_cast = self.calcs.get_element(state, 1, 2, _type)

        if v_cast:
            err = v - self.setpoint
            if err >= 0:
                return max(-err / self.dt, self.a_tilde.a_minus)  # type: ignore
            else:
                return min(-err / self.dt, self.a_tilde.a_plus)  # type: ignore
        else:

            m = self.calcs.get_module(state)

            err = v - self.setpoint

            u = m.where(
                err >= 0,
                self.calcs.clamp(-err / self.dt, low=self.a_tilde.a_minus),
                self.calcs.clamp(-err / self.dt, high=self.a_tilde.a_plus),
            )
            return u  # type: ignore


def get_N(
    v: T,
    a_tilde: A_tilde,
    dt: float,
    calcs: MathCalcs,
    _type: Optional[NumType] = None,
    a: Optional[T] = None,
) -> T:
    m = calcs.get_module(v, _type)
    if a is None:
        a = a_tilde.get_a(v, _type)  # type: ignore
    return m.floor(calcs.abs(v / (dt * a), _type))  # type: ignore


# pylint: disable=too-many-positional-arguments
def get_eta(
    p: T,
    v: T,
    a_tilde: Optional[A_tilde],
    dt: float,
    calcs: MathCalcs,
    _type: Optional[NumType] = None,
    N: Optional[T] = None,
    a: Optional[T] = None,
) -> T:
    if a is None:
        assert a_tilde is not None
        a = a_tilde.get_a(v, _type)  # type: ignore
    if N is None:
        assert a_tilde is not None
        N = get_N(v, a_tilde, dt, calcs, _type=_type, a=a)

    term1 = dt * N * v
    term2 = (N * (N - 1) / 2) * (dt**2) * a  # type: ignore
    term3 = dt * (v + dt * N * a)  # type: ignore

    out = p + term1 + term2 + term3
    return out  # type: ignore


class DblIntBarrierFunction:
    def __init__(
        self,
        lim: float,
        lim_is_min: bool,
        a_tilde: A_tilde,
        dt: float,
        calcs: MathCalcs,
    ):
        self.lim = lim
        self.lim_is_min = lim_is_min
        self.a_tilde = a_tilde
        self.dt = dt
        self.calcs = calcs

    def __repr__(self) -> str:
        return (
            "DblIntBarrierFunction("
            f"lim={self.lim}, "
            f"lim_is_min={self.lim_is_min}, "
            f"a_tilde={self.a_tilde}, "
            f"dt={self.dt}, "
            f"calcs={self.calcs}"
            ")"
        )

    def get_h(self, state: T) -> T:
        p, v = self.calcs.split(state, 2)
        eta = get_eta(p, v, self.a_tilde, self.dt, self.calcs)

        if self.lim_is_min:
            p_min = self.lim
            return self.calcs.clamp(p, high=eta) - p_min  # type: ignore
        else:
            p_max = self.lim
            return p_max - self.calcs.clamp(p, low=eta)  # type: ignore


class DblIntDynamics:
    def __init__(self, dt: float, act_lims: ActuatorLimits, calcs: MathCalcs):
        self.dt = dt
        self.act_lims = act_lims
        self.calcs = calcs

    def get_next_state(self, state: T, action: T) -> T:
        p, v = self.calcs.split(state, 2)
        action = self.act_lims.clamp(action)

        action = self.calcs.scalar_cast(action)[0]  # type: ignore

        next_p = p + self.dt * v
        next_v = v + self.dt * action

        return self.calcs.concat([next_p, next_v])  # type: ignore

    def __repr__(self) -> str:
        return f"DblIntDynamics(dt={self.dt}, act_lims={self.act_lims})"


# pylint: disable=too-many-instance-attributes,abstract-method
class DblIntEnv(gym.Env):
    def __init__(
        self,
        bf: DblIntBarrierFunction,
        dynamics: DblIntDynamics,
        init_pos_bounds: tuple[float, float],
        init_vel_bounds: tuple[float, float],
        max_timesteps: int,
        viewer: Optional[Any],
        monitor: EpisodeMonitor,
        rew_type: RewType,
        use_safe_controller_obs: bool,
        start_barrier_safe: bool,
    ):
        self.bf = bf
        self.dynamics = dynamics
        self.init_pos_bounds = init_pos_bounds
        self.init_vel_bounds = init_vel_bounds
        self.time_tracker = TimeTracker(max_timesteps, 0)
        self.viewer = viewer
        self.monitor = monitor
        self.rho = Rho(dynamics.calcs)

        self.safe_controller_obs_dim = 2
        obs_dim = self.safe_controller_obs_dim if use_safe_controller_obs else 3
        self.observation_space = make_box((1.0), obs_dim)

        self.action_space = make_box((1.0), 1)

        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.closest = np.inf

        self.denormalize = self.dynamics.act_lims.denormalize
        self.normalize = self.dynamics.act_lims.normalize
        self.unsafe_counts: list[bool] = []

        self.enable_viewer = True
        self.rew_type = rew_type
        self.use_safe_controller_obs = use_safe_controller_obs
        self.start_barrier_safe = start_barrier_safe

    # pylint: disable=unused-argument
    def reset(self, **kwargs: Any) -> tuple[NpFloat, InfoDict]:
        def _sample() -> NpFloat:
            _unif = np.random.uniform
            return np.hstack(
                (_unif(*self.init_pos_bounds), _unif(*self.init_vel_bounds)),
                dtype=np.float32,
            )

        if self.viewer is not None:
            self.viewer.reset()
        self.monitor.reset()

        self.time_tracker.num_timesteps = 0

        self.state = _sample()

        if self.start_barrier_safe:
            h = self.bf.get_h(self.state)
            while h < 0:
                self.state = _sample()
                h = self.bf.get_h(self.state)

        self.closest = self.state[0]
        obs, info = self.get_obs()
        self.monitor.add_obs(
            info["safe_controller_obs"], info["rho"], info["safety_margins"]
        )
        return obs, info

    def step(self, action: NpFloat) -> tuple[NpFloat, float, bool, InfoDict]:
        self.monitor.add_action(action)
        action = np.clip(float(action), -1, 1)

        if self.enable_viewer and self.viewer is not None:
            t_beg_step = self.time_tracker.num_timesteps * self.dynamics.dt
            obs = self.get_obs()[0]
            done = (
                self.time_tracker.num_timesteps + 1
            ) >= self.time_tracker.max_timesteps or self.state[0] < 0.0
            self.viewer.render(self.state, action, t_beg_step, done, obs)

        self.time_tracker.num_timesteps += 1
        done = (
            self.time_tracker.num_timesteps >= self.time_tracker.max_timesteps
            or self.state[0] < 0.0
        )

        self.state = np.array(
            self.dynamics.get_next_state(self.state, action), dtype=np.float32
        )

        obs, info = self.get_obs()
        info["cost"] = float(self.state[0] < 0)

        if self.rew_type == RewType.NORMAL:
            rew = -self.state[0]
        elif self.rew_type == RewType.NEG_COST:
            rew = -info["cost"]
        else:
            rew = info["rho"]

        self.monitor.add_obs(
            info["safe_controller_obs"], info["rho"], info["safety_margins"]
        )
        if done:
            info["episode_unsafe"] = self.closest < 0
        return obs, rew, done, info

    def get_obs(self) -> tuple[NpFloat, InfoDict]:
        info: InfoDict = {}
        info["safe_controller_obs"] = self.state.copy()
        info["constraint_violation"] = float(
            self.closest < -self.monitor.floating_point_err_thresh
        )
        info["safety_margins"] = np.array([self.state[0]])
        info["rho"] = self.rho.get_rho(self.state)

        t = self.time_tracker.num_timesteps * self.dynamics.dt

        if self.use_safe_controller_obs:
            obs = self.dynamics.calcs.symlog(info["safe_controller_obs"], NumType.NUMPY)
            return obs, info
        else:
            return np.hstack((self.state, t), dtype=np.float32), info


@dataclasses.dataclass
class DblIntVars:
    dynamics: DblIntDynamics
    barrier_function: DblIntBarrierFunction
    bf_constraint: BarrierConstraint
    safe_controller: DblIntSafeController
    env: DblIntEnv
    params: dict[str, Any]
    calcs: MathCalcs


# pylint: disable=too-many-locals
def make(
    use_scalar: bool,
    allow_np_cast: bool,
    allow_gui: bool,
    log_on_unsafe: bool,
    log_dir: Optional[Path],
    params: Optional[dict[str, Any]],
) -> DblIntVars:

    if params is None:
        params = get_default_params()["dbl-int"]

    calcs = MathCalcs(use_scalar, allow_np_cast)
    act_lims = ActuatorLimits(calcs=calcs, **params["actuator_lims"])

    dt = params["dynamics"]["dt"]
    dynamics = DblIntDynamics(dt, act_lims, calcs)
    a_tilde = A_tilde(calcs=calcs, **params["a_tilde"])
    ctrl = DblIntSafeController(dt, a_tilde, 0.0, calcs)
    bf = DblIntBarrierFunction(0, True, a_tilde, dt, calcs)
    bf_constraint = BarrierConstraint(
        params["barrier_constraint"]["lmbda"], dynamics, bf, calcs=calcs
    )

    if allow_gui and params["do_render"] and log_dir is not None:
        from safe_barrier_envs.dbl_int_viewer import DblIntViewer, DblIntViewerParams
        from safe_barrier_envs.viewer_utils import FrameSaver, PolicyGraphParams

        policy_graph_params = PolicyGraphParams(**params["policy_graph_params"])
        dbl_int_viewer_params = DblIntViewerParams(
            policy_graph_params=policy_graph_params, **params["viewer"]
        )
        frame_saver = FrameSaver(log_dir, 5, 60) if params["save_frames"] else None
        viewer = DblIntViewer(act_lims, dbl_int_viewer_params, frame_saver)
    else:
        viewer = None

    monitor = EpisodeMonitor(
        log_dir=log_dir,
        log_on_unsafe=log_on_unsafe,
        floating_point_err_thresh=float(params["floating_point_err_thresh"]),
    )
    env_params_copy = copy.deepcopy(params["env"])
    env_params_copy["rew_type"] = RewType[params["env"]["rew_type"].upper()]

    env = DblIntEnv(bf, dynamics, viewer=viewer, monitor=monitor, **env_params_copy)

    return DblIntVars(dynamics, bf, bf_constraint, ctrl, env, params, calcs)
