# pylint: disable=too-many-lines
import copy
import dataclasses
import enum
import math
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Optional

import gym
import numpy as np
import scipy.integrate
import torch
import unitpy

from safe_barrier_envs.dbl_int import A_tilde, DblIntSafeController, get_eta
from safe_barrier_envs.utils import (
    TF,
    ActuatorLimits,
    BarrierFunction,
    EpisodeMonitor,
    EpisodeStats,
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


def mph_to_mps(mph: float) -> float:
    return float((mph * unitpy.Unit("mile per hour")).to("m/s").value)


def mps_to_mph(mps: float) -> float:
    return float((mps * unitpy.Unit("m/s")).to("mile per hour").value)


@dataclasses.dataclass
class CarParams:
    width: float
    length_rear: float
    length_front: float

    def get_beta(self, u2: TF, _type: NumType) -> TF:
        lr = self.length_rear
        lf = self.length_front

        _tan, _atan = self._get_funcs(_type)

        beta = _atan(_tan(u2) * lr / (lf + lr))
        return beta  # type: ignore

    def get_beta_inv(self, val: T, _type: NumType) -> T:
        lr = self.length_rear
        lf = self.length_front

        _tan, _atan = self._get_funcs(_type)
        beta_inv = _tan(_atan(((lf + lr) / lr) * val))
        return beta_inv  # type: ignore

    @staticmethod
    def _get_funcs(
        _type: Optional[NumType] = None,
    ) -> tuple[Callable, Callable]:  # type: ignore
        if _type == NumType.SCALAR:
            return math.tan, math.atan
        elif _type == NumType.NUMPY:
            return np.tan, np.arctan
        else:
            return torch.tan, torch.arctan


@dataclasses.dataclass
class SafetySpec:
    D_lead: float
    tau: float
    v_lim: float


@dataclasses.dataclass
class AllVehState:
    a: NpFloat
    b: NpFloat
    c: NpFloat

    def to_vec(self) -> NpFloat:
        return np.hstack((self.a, self.b, self.c), dtype=np.float32)


class CarDynamics:
    def __init__(
        self,
        dt: float,
        car_params: CarParams,
        act_lims: ActuatorLimits,
        calcs: MathCalcs,
    ):
        self.dt = dt
        self.car_params = car_params
        self.act_lims = act_lims
        self.calcs = calcs

    def __repr__(self) -> str:
        return (
            "CarDynamics("
            f"dt={self.dt}, "
            f"car_params={self.car_params}, "
            f"act_lims={self.act_lims}, "
            f"calcs={self.calcs}"
            ")"
        )

    # pylint: disable=too-many-locals
    def get_next_state(self, state: T, action: T) -> T:
        """
        model is from equation (2 a-e) of
        Polack, Philip, et al. "The kinematic bicycle model: A
        consistent model for planning feasible trajectories for
        autonomous vehicles?." 2017 IEEE intelligent vehicles
        symposium (IV). IEEE, 2017.
        https://ieeexplore.ieee.org/abstract/document/7995816
        """

        x, y, v, psi = self.calcs.split(state, 4)
        action = self.act_lims.clamp(action)
        u1, u2 = self.calcs.split(action, 2)

        _type = self.calcs.get_type(x)
        _sin, _cos = self.get_trig_funcs(_type)

        beta_u2 = self.car_params.get_beta(u2, _type)

        next_x = self.get_next_x(x, v, psi, beta_u2, _cos)
        next_y = self.get_next_y(y, v, psi, beta_u2, _sin)
        next_v = self.get_next_v(v, u1)
        next_psi = self.get_next_psi(v, psi, beta_u2, _sin)

        next_state = self.calcs.concat([next_x, next_y, next_v, next_psi])
        return next_state

    @staticmethod
    def get_trig_funcs(_type: NumType) -> tuple[Callable, Callable]:  # type: ignore
        if _type == NumType.SCALAR:
            return math.sin, math.cos
        elif _type == NumType.TORCH:
            return torch.sin, torch.cos
        else:
            return np.sin, np.cos

    def get_next_x(
        self, x: T, v: T, psi: T, beta_u2: TF, _cos: Callable  # type: ignore
    ) -> T:
        return x + self.dt * v * _cos(psi + beta_u2)  # type: ignore

    def get_next_y(
        self, y: T, v: T, psi: T, beta_u2: TF, _sin: Callable  # type: ignore
    ) -> T:
        return y + self.dt * v * _sin(psi + beta_u2)  # type: ignore

    def get_next_v(self, v: T, u1: T) -> T:
        return v + self.dt * u1  # type: ignore

    def get_next_psi(
        self,
        v: T,
        psi: T,
        beta_u2: TF,
        _sin: Callable,  # type: ignore
    ) -> T:
        lr = self.car_params.length_rear
        return psi + self.dt * (v / lr) * _sin(beta_u2)  # type: ignore


# pylint: disable=too-many-instance-attributes
@dataclasses.dataclass
class RhoMargins:
    margins: Any
    headway_margin: Any
    road_margin: Any
    speed_margin: Any
    collision_margin: Any

    offset: Any
    in_lane_a: Any
    in_lane_b: Any

    headway_thresh: Any
    dist_a: Any
    dist_b: Any

    headway_margin_a: Any
    headway_margin_b: Any
    collision_margin_a: Any
    collision_margin_b: Any


@dataclasses.dataclass
class RhoViolations:
    crash_a: Any
    crash_b: Any
    crash: Any

    headway_a: Any
    headway_b: Any
    headway: Any

    speed: Any
    lane: Any

    violation: Any


class Rho:
    def __init__(
        self, spec: SafetySpec, params: CarParams, W_lane: float, calcs: MathCalcs
    ):
        self.spec = spec
        self.params = params
        self.W_lane = W_lane
        self.calcs = calcs

    # pylint: disable=too-many-locals
    def get_margins(self, state: T, _type: Optional[NumType] = None) -> RhoMargins:
        if _type is None:
            _type = self.calcs.get_type(state)
        state_a, state_b, state_c = self.calcs.split(state, 3, 12, _type)

        x_a = self.calcs.split(state_a, 4, 4, _type)[0]
        x_b = self.calcs.split(state_b, 4, 4, _type)[0]
        x_c, y_c, v_c, psi_c = self.calcs.split(state_c, 4, 4, _type)

        offset = get_car_offset(self.params, psi_c, self.calcs)

        road_margin = self.calcs.min(  # type: ignore
            self.calcs.concat(
                [y_c - offset, 2 * self.W_lane - y_c - offset], _type=_type
            ),
            _type=_type,
        )

        headway_thresh = self.spec.D_lead + self.spec.tau * self.calcs.clamp(
            v_c, low=0.0, _type=_type
        )

        in_lane_a = y_c - offset <= self.W_lane
        in_lane_b = y_c + offset >= self.W_lane

        dist_a = x_a - x_c
        dist_b = x_b - x_c

        headway_margin_a = dist_a - headway_thresh
        headway_margin_b = dist_b - headway_thresh

        collision_margin_a = dist_a - self.spec.D_lead
        collision_margin_b = dist_b - self.spec.D_lead

        def _get_margin_lead(_margin_lead_a: Any, _margin_lead_b: Any) -> Any:
            if _type == NumType.TORCH:
                return torch.where(
                    torch.logical_and(in_lane_a, in_lane_b),  # type: ignore
                    torch.minimum(_margin_lead_a, _margin_lead_b),
                    torch.where(  # type: ignore
                        in_lane_a,
                        _margin_lead_a,
                        _margin_lead_b,
                    ),
                )
            else:
                if in_lane_a and in_lane_b:
                    return min(_margin_lead_a, _margin_lead_b)
                elif in_lane_a:
                    return _margin_lead_a
                else:
                    return _margin_lead_b

        headway_margin = _get_margin_lead(headway_margin_a, headway_margin_b)
        collision_margin = _get_margin_lead(collision_margin_a, collision_margin_b)
        speed_margin = self.spec.v_lim - v_c

        margins = self.calcs.concat(
            [headway_margin, road_margin, speed_margin, collision_margin], _type
        )

        return RhoMargins(
            margins=margins,
            headway_margin=headway_margin,
            road_margin=road_margin,
            speed_margin=speed_margin,
            collision_margin=collision_margin,
            offset=offset,
            in_lane_a=in_lane_a,
            in_lane_b=in_lane_b,
            headway_thresh=headway_thresh,
            dist_a=dist_a,
            dist_b=dist_b,
            headway_margin_a=headway_margin_a,
            headway_margin_b=headway_margin_b,
            collision_margin_a=collision_margin_a,
            collision_margin_b=collision_margin_b,
        )

    def get_rho(self, state: T, _type: Optional[NumType] = None) -> T:
        if _type is None:
            _type = self.calcs.get_type(state)

        margins = self.get_margins(state, _type)
        return self.calcs.min(margins.margins)  # type: ignore

    def get_violations(
        self, margins: RhoMargins, eps: float, _type: Optional[NumType]
    ) -> RhoViolations:
        if _type is None:
            _type = self.calcs.get_type(margins.margins)

        m = self.calcs.get_module(margins.margins, _type)

        def _too_close(_key: str) -> Any:
            _suffix = _key[-1]
            _in_lane = getattr(margins, f"in_lane_{_suffix}")
            _val = getattr(margins, _key)
            return m.logical_and(_in_lane, _val < -eps)

        crash_a = _too_close("collision_margin_a")
        crash_b = _too_close("collision_margin_b")
        crash = crash_a or crash_b

        headway_a = _too_close("headway_margin_a")
        headway_b = _too_close("headway_margin_b")
        headway = headway_a or headway_b

        speed = margins.speed_margin < -eps
        lane = margins.road_margin < -eps

        violation = m.logical_or(
            crash, m.logical_or(headway, m.logical_or(speed, lane))
        )

        return RhoViolations(
            crash_a=crash_a,
            crash_b=crash_b,
            crash=crash,
            headway_a=headway_a,
            headway_b=headway_b,
            headway=headway,
            speed=speed,
            lane=lane,
            violation=violation,
        )


class CarContinuousDynamics:
    def __init__(
        self,
        dt: float,
        car_params: CarParams,
        act_lims: ActuatorLimits,
        override: Optional[Callable[[NpFloat, NpFloat], NpFloat]],
    ):
        self.dt = dt
        self.car_params = car_params
        self.act_lims = act_lims
        self.override = override

    def get_next_state(self, state: NpFloat, action: NpFloat) -> NpFloat:

        action_a = action[:2]
        action_b = action[2:4]
        action_c = action[4:]

        def _f_single_veh(
            _single_veh_action: NpFloat, _t: float, _single_veh_state: NpFloat
        ) -> tuple[float, float, float, float]:
            _x, _y, _v, _psi = _single_veh_state
            _u1, _u2 = _single_veh_action

            _beta_u2 = self.car_params.get_beta(_u2, NumType.SCALAR)

            _x_dot = _v * math.cos(_psi + _beta_u2)
            _y_dot = _v * math.sin(_psi + _beta_u2)
            _v_dot = _u1
            _psi_dot = (_v / self.car_params.length_rear) * math.sin(_beta_u2)

            return _x_dot, _y_dot, _v_dot, _psi_dot

        def _f(_t: float, _state: NpFloat) -> NpFloat:
            _state_a = _state[:4]
            _state_b = _state[4:8]
            _state_c = _state[8:]

            _a_dot = _f_single_veh(action_a, _t, _state_a)
            _b_dot = _f_single_veh(action_b, _t, _state_b)

            if self.override is None:
                _u_c = action_c
            else:
                _u_c = self.override(_state, action_c)

            _c_dot = _f_single_veh(_u_c, _t, _state_c)

            _next_state = _a_dot + _b_dot + _c_dot
            return _next_state  # type: ignore

        sln = scipy.integrate.solve_ivp(_f, (0, self.dt), state, method="RK45")
        next_state = sln.y[:, -1]
        return next_state  # type: ignore


class CtrlSteer:
    def __init__(
        self, dt: float, u2_max: float, car_params: CarParams, calcs: MathCalcs
    ):
        self.dt = dt
        _type = NumType.SCALAR
        self.b_min = math.sin(car_params.get_beta(-u2_max, _type))
        self.b_max = math.sin(car_params.get_beta(u2_max, _type))
        self.car_params = car_params
        self.calcs = calcs

    def __repr__(self) -> str:
        return (
            "CtrlSteer("
            f"dt={self.dt}, "
            f"b_min={self.b_min}, "
            f"b_max={self.b_max}, "
            f"car_params={self.car_params}, "
            f"calcs={self.calcs}"
            ")"
        )

    def get_control(self, v: T, psi: T, _type: Optional[NumType] = None) -> T:
        lr = self.car_params.length_rear

        if _type is None:
            _type = self.calcs.get_type(v)

        v_cast, v_was_cast = self.calcs.scalar_cast(v, _type)
        psi_cast, psi_was_cast = self.calcs.scalar_cast(psi, _type)

        if v_was_cast and psi_was_cast:
            if v_cast == 0:
                return 0.0  # type: ignore
            else:
                val = -psi_cast * lr / (self.dt * v_cast)
                if val < 0:
                    val = max(val, self.b_min)  # type: ignore
                else:
                    val = min(val, self.b_max)  # type: ignore
                val = self.car_params.get_beta_inv(math.asin(val), NumType.SCALAR)
                return val  # type: ignore
        else:

            m = self.calcs.get_module(v, _type)
            eps = 1e-6
            mask = m.abs(v) < eps
            zero_vel = v == 0

            # where v == 0 is mostly just to avoid dividing by zero, we later
            # set the control to zero for this case
            v = m.where(zero_vel, eps, m.where(mask, eps * m.sign(v), v))
            val = -psi * self.car_params.length_rear / (self.dt * v)
            val = self.calcs.clamp(val, low=self.b_min, high=self.b_max)
            val = self.car_params.get_beta_inv(m.arcsin(val), _type)

            val = m.where(zero_vel, 0.0, val)
            return val  # type: ignore


class CarSafeController:
    def __init__(
        self,
        ctrl_dbl_int: DblIntSafeController,
        ctrl_steer: CtrlSteer,
        calcs: MathCalcs,
    ):
        self.ctrl_dbl_int = ctrl_dbl_int
        self.ctrl_steer = ctrl_steer
        self.calcs = calcs

    def __repr__(self) -> str:
        return (
            f"CarSafeController("
            f"ctrl_dbl_int={self.ctrl_dbl_int}, "
            f"ctrl_steer={self.ctrl_steer}, "
            f"calcs={self.calcs}"
            ")"
        )

    def get_control(self, state: T, _type: Optional[NumType] = None) -> T:
        if _type is None:
            _type = self.calcs.get_type(state)

        state, to_np = self.calcs.np_cast(state)  # type: ignore
        try:
            v, psi = self.calcs.split(state, 4, _type=_type)[2:]
        except (ValueError, AssertionError):
            all_veh_state = state
            state_c = self.calcs.split(all_veh_state, 3, 12, _type)[2]
            v, psi = self.calcs.split(state_c, 4, _type=_type)[2:]

        u_dbl = self.ctrl_dbl_int.get_control(v, _type)
        u_steer = self.ctrl_steer.get_control(v, psi, _type)

        if to_np:
            return torch.tensor([[u_dbl, u_steer]])  # type: ignore
        else:
            return self.calcs.concat([u_dbl, u_steer], _type)


def get_car_offset(
    car_params: CarParams, psi: T, calcs: MathCalcs, _type: Optional[NumType] = None
) -> T:
    L = car_params.length_front
    W = car_params.width

    if _type is None:
        _type = calcs.get_type(psi)

    if _type == NumType.SCALAR:
        _abs, _sin, _cos = abs, math.sin, math.cos
    elif _type == NumType.NUMPY:
        _abs, _sin, _cos = np.abs, np.sin, np.cos  # type: ignore
    else:
        _abs, _sin, _cos = torch.abs, torch.sin, torch.cos  # type: ignore

    offset = L * _abs(_sin(psi)) + (W / 2) * _abs(_cos(psi))
    return offset  # type: ignore


class LaneRho(Generic[T]):
    def __init__(self, L1: T, H1: T, L2: T, H2: T):
        self.L1: T = L1
        self.H1: T = H1
        self.L2: T = L2
        self.H2: T = H2

    def __repr__(self) -> str:
        calcs = MathCalcs(True, True)
        out = calcs.concat([self.L1, self.H1, self.L2, self.H2])
        return out.__repr__()  # type: ignore

    def __getitem__(self, idx: int) -> T:
        match idx:
            case 0:
                return self.L1
            case 1:
                return self.H1
            case 2:
                return self.L2
            case 3:
                return self.H2
            case _:
                raise IndexError("only indexes 0-3 are supported")

    # pylint: disable=too-many-positional-arguments
    @classmethod
    def make(
        cls,
        y: T,
        psi: T,
        W_lane: float,
        car_params: CarParams,
        calcs: MathCalcs,
        _type: Optional[NumType] = None,
    ) -> "LaneRho":  # type: ignore
        offset = get_car_offset(car_params, psi, calcs, _type)

        rho_L1 = y - offset
        rho_H1 = W_lane - offset - y
        rho_L2 = y - offset - W_lane
        rho_H2 = 2 * W_lane - offset - y

        return cls(rho_L1, rho_H1, rho_L2, rho_H2)  # type: ignore


class CarBFLane:
    # pylint: disable=# pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        W_lane: float,
        dt: float,
        dynamics: CarDynamics,
        ctrl_dbl_int: DblIntSafeController,
        ctrl_steer: CtrlSteer,
        calcs: MathCalcs,
    ):
        self.W_lane = W_lane
        self.dt = dt
        self.dynamics = dynamics
        self.ctrl_dbl_int = ctrl_dbl_int
        self.ctrl_steer = ctrl_steer
        self.calcs = calcs
        self.num_steps: Optional[Any] = None

    def __repr__(self) -> str:
        return (
            "CarBFLane("
            f"W_lane={self.W_lane}, "
            f"dt={self.dt}, "
            f"dynamics={self.dynamics}, "
            f"ctrl_dbl_int={self.ctrl_dbl_int}, "
            f"ctrl_steer={self.ctrl_steer}"
            ")"
        )

    # pylint: disable=too-many-locals
    def get_h(self, state: T, save_num_steps: bool = False) -> LaneRho[T]:
        # for computational efficiency, we do one integration and keep track of
        # the four rhos

        def _get_done(_done: bool | T, _v: T, _psi: T, _is_scalar: bool) -> T:
            _eps = 1e-6
            if _is_scalar:
                return _done or (abs(_v) < _eps or abs(_psi) < _eps)  # type: ignore
            else:
                assert m is not None  # type: ignore
                _or = m.logical_or  # type: ignore
                _done = _or(
                    done, _or(m.abs(_v) < _eps, m.abs(_psi) < _eps)  # type: ignore
                )
                return _done  # type: ignore

        def _update_rho(
            _done: T, _rho1: LaneRho[T], _rho2: LaneRho[T], _is_scalar: bool, _m: Any
        ) -> LaneRho[T]:

            if _is_scalar:
                if _done:
                    return _rho1
                else:
                    return LaneRho(
                        L1=min(_rho1.L1, _rho2.L1),  # type: ignore
                        H1=min(_rho1.H1, _rho2.H1),  # type: ignore
                        L2=min(_rho1.L2, _rho2.L2),  # type: ignore
                        H2=min(_rho1.H2, _rho2.H2),  # type: ignore
                    )

            else:

                def __min(_v1: T, _v2: T) -> T:
                    return _m.where(_done, _v1, _m.minimum(_v1, _v2))  # type: ignore

                return LaneRho(
                    L1=__min(_rho1.L1, _rho2.L1),
                    H1=__min(_rho1.H1, _rho2.H1),
                    L2=__min(_rho1.L2, _rho2.L2),
                    H2=__min(_rho1.H2, _rho2.H2),
                )

        def _all_done(_done: T, _is_scalar: bool, _m: Any) -> T:
            return _done if _is_scalar else _m.all(_done)

        car_params = self.dynamics.car_params

        y, v, psi = self.calcs.split(state, 4)[1:]
        state_type = self.calcs.get_type(y)
        is_scalar = state_type == NumType.SCALAR

        if is_scalar:
            m = None
            done = False
        else:
            m = self.calcs.get_module(y)
            if state_type == NumType.NUMPY:
                done = np.zeros(y.shape, dtype=np.int32)  # type: ignore
            else:
                done = torch.zeros(  # type: ignore
                    y.shape, dtype=torch.int32, device=y.device
                )

        rho = LaneRho.make(y, psi, self.W_lane, car_params, self.calcs, state_type)
        done = _get_done(done, v, psi, is_scalar)  # type: ignore

        if save_num_steps:
            if is_scalar:
                self.num_steps = 0
            else:
                self.num_steps = m.zeros(y.shape, dtype=m.int32)  # type: ignore

        _sin, _cos = self.dynamics.get_trig_funcs(state_type)

        while not _all_done(done, is_scalar, m):  # type: ignore
            u1 = self.ctrl_dbl_int.get_control(v, state_type)
            u2 = self.ctrl_steer.get_control(v, psi, state_type)
            beta_u2 = car_params.get_beta(u2, state_type)

            y = self.dynamics.get_next_y(y, v, psi, beta_u2, _sin)
            v = self.dynamics.get_next_v(v, u1)
            psi = self.dynamics.get_next_psi(v, psi, beta_u2, _sin)

            curr_rho = LaneRho.make(
                y, psi, self.W_lane, car_params, self.calcs, state_type
            )

            rho = _update_rho(done, rho, curr_rho, is_scalar, m)  # type: ignore
            if save_num_steps:
                if is_scalar:
                    self.num_steps += 1  # type: ignore
                else:
                    self.num_steps = m.where(  # type: ignore
                        done, self.num_steps, self.num_steps + 1  # type: ignore
                    )
            done = _get_done(done, v, psi, is_scalar)  # type: ignore

        return rho


class CarBFLead:
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        D_lead: float,
        tau: float,
        a_tilde: A_tilde,
        dt: float,
        act_lims: ActuatorLimits,
        calcs: MathCalcs,
    ):
        self.D_lead = D_lead
        self.tau = tau
        self.a_tilde = a_tilde
        self.dt = dt
        self.act_lims = act_lims
        self.calcs = calcs

    def __repr__(self) -> str:
        return (
            "CarBFLead("
            f"D_lead={self.D_lead}, "
            f"tau={self.tau}, "
            f"a_tilde={self.a_tilde}, "
            f"dt={self.dt}, "
            f"calcs={self.calcs}, "
            ")"
        )

    def get_h(self, all_car_state: T) -> tuple[T, T]:
        state_a, state_b, state_c = self.calcs.split(all_car_state, 3, 12)
        x_c, _, v_c, _ = self.calcs.split(state_c, 4)

        _type = self.calcs.get_type(x_c)
        m = self.calcs.get_module(state_a, _type)
        tilde_v_c = self.calcs.clamp(v_c, low=0.0, _type=_type)

        eta_c = get_eta(x_c, tilde_v_c, self.a_tilde, self.dt, self.calcs, _type=_type)

        def _get_h_j(_state_j: T) -> T:
            _x_j, _, _v_j, _psi_j = self.calcs.split(_state_j, 4)
            _a_j = self.act_lims.low[0]
            _v_j = self.calcs.clamp(_v_j, high=tilde_v_c, _type=_type)
            _N_j = m.floor(-_v_j / (self.dt * _a_j))
            _eta_j = get_eta(_x_j, _v_j, None, self.dt, self.calcs, _type, _N_j, _a_j)

            _d_j = _eta_j - eta_c

            _h_j = _d_j - self.D_lead - self.tau * tilde_v_c
            return _h_j  # type: ignore

        h_a = _get_h_j(state_a)
        h_b = _get_h_j(state_b)

        return h_a, h_b


class CarBFSpeed:
    def __init__(self, v_lim: float, calcs: MathCalcs):
        self.v_lim = v_lim
        self.calcs = calcs

    def __repr__(self) -> str:
        return f"CarBFSpeed(v_lim={self.v_lim}, calcs={self.calcs})"

    def get_h(self, state_or_v: T) -> T:
        v = self.calcs.get_element(state_or_v, 2, 4)[0]
        return self.v_lim - self.calcs.clamp(v, low=0.0)  # type: ignore


class CarBF:
    def __init__(
        self,
        h_lane: CarBFLane,
        h_lead: CarBFLead,
        h_speed: CarBFSpeed,
        calcs: MathCalcs,
    ):
        self.h_lane = h_lane
        self.h_lead = h_lead
        self.h_speed = h_speed
        self.calcs = calcs

    # pylint: disable=too-many-locals
    def get_h(self, all_car_state: T) -> T:
        all_car_state, to_np = self.calcs.np_cast(all_car_state)  # type: ignore
        state_c = self.calcs.split(all_car_state, 3, 12)[2]

        y_c, v_c = self.calcs.split(state_c, 4)[1:3]
        _type = self.calcs.get_type(y_c)

        h_speed = self.h_speed.get_h(v_c)
        h_lane = self.h_lane.get_h(state_c)
        h_lead_a, h_lead_b = self.h_lead.get_h(all_car_state)

        def _min(_vals: list[Any]) -> Any:
            return self.calcs.min(self.calcs.concat(_vals, _type), _type)

        def _max(_vals: list[Any]) -> Any:
            return self.calcs.max(self.calcs.concat(_vals, _type), _type)

        bracket1 = _min([h_speed, h_lane.L1, h_lane.H2])

        term1 = _min([h_lane.H1, h_lead_a])
        term2 = _min([h_lane.L2, h_lead_b])
        term3 = _min([h_lead_a, h_lead_b])

        bracket2 = _max([term1, term2, term3])

        h = _min([bracket1, bracket2])

        if to_np:
            h = torch.tensor([[h]], dtype=torch.float32)

        return h  # type: ignore


class LeadCarMode(enum.Enum):
    BRAKE = 0
    CRUISE = 1
    ACCELERATE = 2


class LeadCar:
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        lane_idx: Literal["a", "b"],
        ctrl: DblIntSafeController,
        W_lane: float,
        speed_range_mph: tuple[float, float],
        cruise_interval: tuple[float, float],
        start_x_lims: tuple[float, float],
        calcs: MathCalcs,
    ):
        self.lane_idx = lane_idx
        self.W_lane = W_lane

        self.speed_range_mph = np.array(speed_range_mph)
        self.speed_range_mps = (
            (self.speed_range_mph * unitpy.Unit("mile per hour")).to("m/s").value
        )
        self.cruise_interval = cruise_interval
        self.start_x_lims = start_x_lims
        self.calcs = calcs

        self.ctrl = ctrl
        self.mode = LeadCarMode.CRUISE
        self.time_end_cruise = np.inf
        self.is_max_decel = False
        self.enable_respawn = True

    def __repr__(self) -> str:
        return (
            "LeadCar("
            f"lane_idx={self.lane_idx}, "
            f"ctrl={self.ctrl}, "
            f"W_lane={self.W_lane}, "
            f"speed_range_mph={self.speed_range_mph}, "
            f"cruise_interval={self.cruise_interval}, "
            f"start_x_lims={self.start_x_lims}, "
            f"calcs={self.calcs}"
            ")"
        )

    def reset(self, state: NpFloat) -> None:
        state[0] = np.random.uniform(*self.start_x_lims)
        state[1] = self.W_lane * (0.5 if self.lane_idx == "a" else 1.5)
        state[2] = np.random.uniform(*self.speed_range_mps)

        self.mode = LeadCarMode.CRUISE
        self.ctrl.setpoint = state[2]
        self.time_end_cruise = np.random.uniform(*self.cruise_interval)

    def step(self, state: NpFloat, x_follow: float, t: float) -> NpFloat:
        x, _, v, _ = state
        vel_close = abs(v - self.ctrl.setpoint) < 1

        if x < x_follow and self.enable_respawn:
            # print('spawning')
            self.reset(state)
            state[0] += x_follow

        if self.is_max_decel:
            self.ctrl.setpoint = 0.0
        else:
            if self.mode == LeadCarMode.CRUISE and t >= self.time_end_cruise:
                self.mode = LeadCarMode.BRAKE
                self.ctrl.setpoint = self.ctrl.setpoint * np.random.uniform(0, 1)

            elif self.mode == LeadCarMode.BRAKE and vel_close:
                self.mode = LeadCarMode.ACCELERATE
                self.ctrl.setpoint = np.random.uniform(*self.speed_range_mps)

            elif self.mode == LeadCarMode.ACCELERATE and vel_close:
                self.mode = LeadCarMode.CRUISE
                self.time_end_cruise = t + np.random.uniform(*self.cruise_interval)

        action = self.ctrl.get_control(v, NumType.SCALAR)
        return action  # type: ignore


# pylint: disable=abstract-method,too-many-instance-attributes
class CarEnv(gym.Env):  # type: ignore
    # pylint: disable=too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        car_dynamics: CarDynamics,
        lead_car_a: LeadCar,
        lead_car_b: LeadCar,
        W_lane: float,
        safety_spec: SafetySpec,
        speed_tgt_mph: float,
        max_timesteps: int,
        viewer: Optional[Any],
        bf: BarrierFunction,
        calcs: MathCalcs,
        monitor: EpisodeMonitor,
        continuous_dynamics: CarContinuousDynamics,
        use_continuous_dynamics: bool,
        done_on_catastrophic_unsafe: bool,
        done_on_any_unsafe: bool,
        rew_type: RewType,
        use_safe_controller_obs: bool,
        start_barrier_safe: bool,
        early_stop_when_below_vel: float,
        pct_max_decel: float,
        rho: Rho,
    ):
        self.car_dynamics = car_dynamics
        self.continuous_dynamics = continuous_dynamics
        self.use_continuous_dynamics = use_continuous_dynamics
        self.done_on_catastrophic_unsafe = done_on_catastrophic_unsafe
        self.done_on_any_unsafe = done_on_any_unsafe
        self.lead_car_a = lead_car_a
        self.lead_car_b = lead_car_b
        self.W_lane = W_lane
        self.safety_spec = safety_spec
        self.speed_tgt_mph = speed_tgt_mph
        self.speed_tgt = (speed_tgt_mph * unitpy.Unit("mile per hour")).to("m/s").value
        self.viewer = viewer
        self.bf = bf
        self.calcs = calcs
        self.monitor = monitor
        self.pct_max_decel = pct_max_decel

        self.episode_stats = EpisodeStats()

        self.states = AllVehState(*[np.zeros(4, dtype=np.float32) for _ in range(3)])

        self.time_tracker = TimeTracker(max_timesteps, 0)

        self.safe_controller_obs_dim = 12
        obs_dim = self.safe_controller_obs_dim if use_safe_controller_obs else 8
        self.observation_space = make_box((1.0), obs_dim)
        self.action_space = make_box((1.0), 2)

        self.denormalize = self.car_dynamics.act_lims.denormalize
        self.normalize = self.car_dynamics.act_lims.normalize
        self.unsafe_counts: list[bool] = []
        self.speed_deviations: list[float] = []

        self.rew_type = rew_type
        self.use_safe_controller_obs = use_safe_controller_obs
        self.start_barrier_safe = start_barrier_safe
        self.early_stop_when_below_vel = early_stop_when_below_vel
        self.rho = rho

        self.num_episodes = 0

    # pylint: disable=unused-argument
    def reset(self, **kwargs: Any) -> tuple[NpFloat, InfoDict]:
        self.num_episodes += 1

        if self.viewer is not None:
            self.viewer.reset()

        self.monitor.reset()

        while True:
            self.lead_car_a.reset(self.states.a)
            self.lead_car_b.reset(self.states.b)

            offset = self.car_dynamics.car_params.width
            eps = 1e-3 * self.W_lane  # to avoid floating point errors
            y_c = np.random.uniform(offset + eps, 2 * self.W_lane - offset - eps)

            if self.start_barrier_safe:
                self.states.c[:] = [0.0, y_c, 0.95 * self.speed_tgt, 0]
                safe_controller_obs = np.hstack(
                    (self.states.a, self.states.b, self.states.c), dtype=np.float32
                )
                bf_val = self.bf.get_h(safe_controller_obs)
                if bf_val > 0:
                    break

            else:
                speed = np.random.uniform(0, self.speed_tgt)
                ang = np.radians(np.random.uniform(-5, 5))
                self.states.c[:] = [0.0, y_c, speed, ang]

                if self.rho.get_rho(self.states.to_vec()) >= 0:
                    break

        is_max_decel = np.random.uniform() < self.pct_max_decel
        self.lead_car_a.is_max_decel = is_max_decel
        self.lead_car_b.is_max_decel = is_max_decel

        self.time_tracker.num_timesteps = 0
        self.speed_deviations = []

        margins = self.rho.get_margins(self.states.to_vec(), NumType.NUMPY)
        violations = self.rho.get_violations(
            margins, self.monitor.floating_point_err_thresh, NumType.NUMPY
        )
        info = self._update_constraint_info(margins, violations, {})
        info["rho"] = np.min(margins.margins)
        self.unsafe_counts = [bool(info["constraint_violation"])]

        obs = self.get_obs()
        self.tot_rew = 0.0

        # self.psi_lims = []
        self.monitor.add_obs(
            info["safe_controller_obs"], info["rho"], info["safety_margins"]
        )

        return obs, info

    def _make_all_veh_state(self, state: NpFloat) -> AllVehState:
        state_a, state_b, state_c = self.calcs.split(state, 3, 12, NumType.NUMPY)
        return AllVehState(state_a, state_b, state_c)

    def step(self, action: NpFloat) -> tuple[NpFloat, float, bool, InfoDict]:
        # if self.num_episodes >= 2 and self.time_tracker.num_timesteps > 300:
        #     import lvdb; lvdb.set_trace()  # fmt: skip

        self.monitor.add_action(action)
        action = np.clip(action, -1, 1)
        orig_action = action.copy()
        action = self.denormalize(action)
        self.time_tracker.num_timesteps += 1
        dynamics = self.car_dynamics
        t = self.time_tracker.num_timesteps * dynamics.dt
        # print(self.t, action, self.state_c[2])

        # state_a, state_b, state_c = split(self.all_veh_state, 3, 12)
        x_c = self.states.c[0]
        # print('psi: ', np.rad2deg(self.states.c[3]))
        action_a = np.hstack((self.lead_car_a.step(self.states.a, x_c, t), 0))
        action_b = np.hstack((self.lead_car_b.step(self.states.b, x_c, t), 0))

        if self.use_continuous_dynamics:
            states = np.hstack([self.states.a, self.states.b, self.states.c])
            actions = np.hstack([action_a, action_b, action])
            next_states = self.continuous_dynamics.get_next_state(states, actions)

            # don't allow negative speeds since assumedly we aren't allowing the
            # car to change gears
            next_state_c = next_states[8:]
            if next_state_c[2] < 0:
                next_state_c[2] = 0

            self.states.a = next_states[:4]
            self.states.b = next_states[4:8]
            self.states.c = next_state_c

        else:

            self.states.a = dynamics.get_next_state(self.states.a, action_a)
            self.states.b = dynamics.get_next_state(self.states.b, action_b)

            # don't allow negative speeds since assumedly we aren't allowing the
            # car to change gears
            min_allowed_acc = max(
                dynamics.act_lims.low[0], -self.states.c[2] / dynamics.dt
            )
            action[0] = max(min_allowed_acc, action[0])

            self.states.c = dynamics.get_next_state(self.states.c, action)

        self.speed_deviations.append(abs(self.speed_tgt - self.states.c[2]))
        # self.psi_lims.append(abs(np.rad2deg(self.states.c[3])))

        margins = self.rho.get_margins(self.states.to_vec(), NumType.NUMPY)
        violations = self.rho.get_violations(
            margins, self.monitor.floating_point_err_thresh, NumType.NUMPY
        )

        done, info = self.get_done(violations)
        info = self._update_constraint_info(margins, violations, info)
        info["rho"] = np.min(margins.margins)

        if self.rew_type == RewType.NORMAL:
            rew = self.get_rew()
        elif self.rew_type == RewType.NEG_COST:
            rew = -info["cost"]
        else:
            rew = info["rho"]

        self.tot_rew += rew

        if done:
            # print(self.tot_rew)
            # print(max(self.psi_lims))
            info["pct_unsafe"] = float(np.mean(self.unsafe_counts))
            info["speed_deviation"] = float(np.mean(self.speed_deviations))

            info.update({f"episode_{k}": v for k, v in info.items()})
            self.episode_stats.update(info)

        obs = self.get_obs()

        if self.viewer is not None:
            if self.num_episodes >= 0:
                self.viewer.render(
                    self.states, orig_action, margins, violations, t, done, info, obs
                )

        self.monitor.add_obs(
            info["safe_controller_obs"], info["rho"], info["safety_margins"]
        )
        return obs, rew, done, info

    def get_rew(self) -> float:
        v_c = self.states.c[2]
        speed_deviation = abs(v_c - self.speed_tgt)
        rew = 1 - speed_deviation / self.speed_tgt
        return float(rew)

    def get_done(self, violations: RhoViolations) -> tuple[bool, InfoDict]:
        info = {}
        exit_time = self.time_tracker.num_timesteps >= self.time_tracker.max_timesteps

        if self.done_on_any_unsafe:
            done = violations.violation or exit_time
        elif self.done_on_catastrophic_unsafe:
            done = violations.crash or violations.lane or exit_time
        else:
            done = exit_time

        exit_early_stop_below_vel = (
            self.early_stop_when_below_vel
            and self.states.c[2] < self.early_stop_when_below_vel
        )
        if exit_early_stop_below_vel:
            done = True

        if done:
            info["exit_crash"] = (
                float(violations.crash) and self.done_on_catastrophic_unsafe
            )
            info["exit_off_road"] = (
                float(violations.lane) and self.done_on_catastrophic_unsafe
            )
            info["exit_t_thresh"] = float(exit_time)
            info["exit_too_close"] = violations.headway and self.done_on_any_unsafe
            info["exit_too_fast"] = violations.speed and self.done_on_any_unsafe
            info["exit_early_stop_below_vel"] = exit_early_stop_below_vel

            # import pprint
            #
            # pprint.pprint(violations)
            # if violations.violation:
            #     import lvdb; lvdb.set_trace()  # fmt: skip

        return done, info

    def _update_constraint_info(
        self, margin: RhoMargins, violation: RhoViolations, info: InfoDict
    ) -> InfoDict:
        info["safe_controller_obs"] = self.states.to_vec()

        info["constraint_violation"] = float(violation.violation)
        info["safety_margins"] = margin.margins
        info["cost"] = float(violation.violation)

        return info

    def get_obs(self) -> NpFloat:
        # state_a, state_b, state_c = split(self.all_veh_state, 3, 12)
        x_c, y_c, v_c, psi_c = self.states.c

        dist_a = self.states.a[0] - x_c
        dist_b = self.states.b[0] - x_c
        start_x = self.lead_car_a.start_x_lims[1]

        def _normalize_dist(_d: float) -> float:
            _half = start_x / 2
            return (_d - _half) / _half

        if self.use_safe_controller_obs:
            temp_states = copy.deepcopy(self.states)
            temp_states.a[0] -= temp_states.c[0]
            temp_states.b[0] -= temp_states.c[0]
            temp_states.c[0] = 0
            obs = temp_states.to_vec()

            obs = self.car_dynamics.calcs.symlog(obs, NumType.NUMPY)

        else:
            obs = np.array(
                [
                    _normalize_dist(dist_a),
                    _normalize_dist(dist_b),
                    (self.states.a[2] - self.speed_tgt) / self.speed_tgt,
                    (self.states.b[2] - self.speed_tgt) / self.speed_tgt,
                    (y_c - self.W_lane) / self.W_lane,
                    (v_c - self.speed_tgt) / self.speed_tgt,
                    np.sin(psi_c),
                    np.cos(psi_c),
                ],
                dtype=np.float32,
            )

        return obs


class CarBFConstraint:
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        lmbda: float,
        dynamics: CarDynamics,
        barrier_function: BarrierFunction,
        lead_ctrl_a: DblIntSafeController,
        lead_ctrl_b: DblIntSafeController,
        calcs: MathCalcs,
    ):
        self.lmbda = lmbda
        self.dynamics = dynamics
        self.barrier_function = barrier_function
        self.calcs = calcs

        # create a copy so that we can freeze the setpoint
        self.lead_ctrl_a = copy.deepcopy(lead_ctrl_a)
        self.lead_ctrl_b = copy.deepcopy(lead_ctrl_b)

        self.lead_ctrl_a.setpoint = 0.0
        self.lead_ctrl_b.setpoint = 0.0

    # pylint: disable=too-many-locals
    def get_constraint(
        self, all_car_state: T, action: TF, h: Optional[T] = None
    ) -> T:
        all_car_state, state_to_np = self.calcs.np_cast(all_car_state)  # type: ignore
        action, action_to_np = self.calcs.np_cast(action)  # type: ignore

        _type = self.calcs.get_type(all_car_state)

        assert (state_to_np and action_to_np) or (not state_to_np and not action_to_np)
        state_a, state_b, state_c = self.calcs.split(all_car_state, 3, 12, _type=_type)

        v_a = self.calcs.split(state_a, 4, _type=_type)[2]
        v_b = self.calcs.split(state_b, 4, _type=_type)[2]

        acc_a = self.lead_ctrl_a.get_control(v_a, _type)
        acc_b = self.lead_ctrl_b.get_control(v_b, _type)

        if _type == NumType.TORCH:
            action_a = torch.cat(
                [acc_a, torch.zeros_like(acc_a)], dim=1  # type: ignore
            )
            action_b = torch.cat(
                [acc_b, torch.zeros_like(acc_b)], dim=1  # type: ignore
            )
        else:
            action_a = np.hstack((acc_a, 0))  # type: ignore
            action_b = np.hstack((acc_b, 0))  # type: ignore

        if h is None:
            h = self.barrier_function.get_h(all_car_state)

        next_state_a = self.dynamics.get_next_state(state_a, action_a)  # type: ignore
        next_state_b = self.dynamics.get_next_state(state_b, action_b)  # type: ignore
        next_state_c = self.dynamics.get_next_state(state_c, action)  # type: ignore

        next_state = self.calcs.concat(  # type: ignore
            [next_state_a, next_state_b, next_state_c], _type=_type
        )
        next_h = self.barrier_function.get_h(next_state)  # type: ignore
        constraint = next_h - (1 - self.lmbda) * h  # type: ignore

        if state_to_np:
            constraint = torch.tensor(
                [[constraint]], dtype=torch.float32  # type: ignore
            )

        return constraint  # type: ignore


@dataclasses.dataclass
class CarVars:
    car_dynamics: CarDynamics
    barrier_function: CarBF
    bf_constraint: CarBFConstraint
    safe_controller: CarSafeController
    a_tilde: A_tilde
    safety_spec: SafetySpec
    env: CarEnv
    params: dict[str, Any]
    calcs: MathCalcs


# pylint: disable=too-many-locals,too-many-positional-arguments
def make(
    use_scalar: bool,
    allow_np_cast: bool,
    allow_gui: bool,
    log_on_unsafe: bool,
    log_dir: Optional[Path],
    params: Optional[dict[str, Any]],
) -> CarVars:

    if params is None:
        params = get_default_params()["car"]

    calcs = MathCalcs(use_scalar, allow_np_cast)
    car_params = CarParams(**params["car_params"])

    act_lims = ActuatorLimits(calcs=calcs, **params["actuator_lims"])
    act_lims.set_idx(1, np.deg2rad(act_lims.get_idx(1)))  # type: ignore

    dt = params["dynamics"]["dt"]
    car_dynamics = CarDynamics(dt, car_params, act_lims, calcs)

    continuous_dynamics = CarContinuousDynamics(dt, car_params, act_lims, None)

    safety_spec_dict = params["safety_spec"].copy()
    safety_spec_dict["v_lim"] = (
        (safety_spec_dict["v_lim_miles_per_hour"] * unitpy.Unit("mile per hour"))
        .to("m/s")
        .value
    )
    del safety_spec_dict["v_lim_miles_per_hour"]
    safety_spec = SafetySpec(**safety_spec_dict)
    a_tilde = A_tilde(calcs=calcs, **params["a_tilde"])
    assert a_tilde.a_minus >= act_lims.low[0] and a_tilde.a_plus <= act_lims.high[0]

    W_lane = params["W_lane"]

    ctrl_dbl_int = DblIntSafeController(dt, a_tilde, setpoint=0, calcs=calcs)
    ctrl_steer = CtrlSteer(dt, act_lims.high[1], car_params, calcs=calcs)
    ctrl = CarSafeController(ctrl_dbl_int, ctrl_steer, calcs=calcs)

    h_lane = CarBFLane(W_lane, dt, car_dynamics, ctrl_dbl_int, ctrl_steer, calcs=calcs)
    h_lead = CarBFLead(
        safety_spec.D_lead, safety_spec.tau, a_tilde, dt, act_lims, calcs=calcs
    )
    h_speed = CarBFSpeed(safety_spec.v_lim, calcs=calcs)

    bf = CarBF(h_lane, h_lead, h_speed, calcs=calcs)

    def _make_lead_car(
        _lane_idx: Literal["a", "b"],
    ) -> tuple[DblIntSafeController, LeadCar]:
        _u1_min = act_lims.low[0]
        _u1_max = act_lims.high[0]
        _ctrl = DblIntSafeController(dt, A_tilde(_u1_min, _u1_max, calcs), 0, calcs)

        if not params["env"]["start_barrier_safe"]:
            # allow the vehicles to start arbitrarily close
            params["env"]["lead_veh"]["start_x_lims"][0] = 0

        return _ctrl, LeadCar(
            _lane_idx, _ctrl, W_lane, calcs=calcs, **params["env"]["lead_veh"]
        )

    lead_ctrl_a, lead_car_a = _make_lead_car("a")
    lead_ctrl_b, lead_car_b = _make_lead_car("b")

    bf_constraint = CarBFConstraint(
        params["barrier_constraint"]["lmbda"],
        car_dynamics,
        bf,
        lead_ctrl_a=lead_ctrl_a,
        lead_ctrl_b=lead_ctrl_b,
        calcs=calcs,
    )

    if allow_gui and params["env"]["do_render"]:
        from safe_barrier_envs.car_viewer import (
            CarEnvViewer,
            CarEnvViewerParams,
            LegendParams,
        )
        from safe_barrier_envs.viewer_utils import FrameSaver, PolicyGraphParams

        policy_graph_params = PolicyGraphParams(**params["policy_graph_params"])
        legend_params = LegendParams(**params["legend_params"])
        car_env_viewer_params = CarEnvViewerParams(
            policy_graph_params=policy_graph_params,
            legend_params=legend_params,
            **params["viewer"],
        )
        assert log_dir is not None
        frame_saver = (
            FrameSaver(log_dir, 5, 60) if params["env"]["save_frames"] else None
        )
        viewer = CarEnvViewer(
            W_lane,
            safety_spec,
            lead_car_a.start_x_lims[1],
            act_lims,
            car_env_viewer_params,
            frame_saver,
        )
    else:
        viewer = None

    monitor = EpisodeMonitor(
        log_dir=log_dir,
        log_on_unsafe=log_on_unsafe,
        floating_point_err_thresh=float(params["floating_point_err_thresh"]),
    )

    rho = Rho(safety_spec, car_params, W_lane, calcs)
    env = CarEnv(
        car_dynamics=car_dynamics,
        lead_car_a=lead_car_a,
        lead_car_b=lead_car_b,
        W_lane=W_lane,
        safety_spec=safety_spec,
        speed_tgt_mph=params["env"]["speed_tgt_mph"],
        max_timesteps=params["env"]["max_timesteps"],
        viewer=viewer,
        monitor=monitor,
        bf=bf,
        calcs=calcs,
        continuous_dynamics=continuous_dynamics,
        use_continuous_dynamics=params["use_continuous_dynamics"],
        done_on_catastrophic_unsafe=params["env"]["done_on_catastrophic_unsafe"],
        rew_type=RewType[params["env"]["rew_type"].upper()],
        use_safe_controller_obs=params["env"]["use_safe_controller_obs"],
        start_barrier_safe=params["env"]["start_barrier_safe"],
        done_on_any_unsafe=params["env"]["done_on_any_unsafe"],
        early_stop_when_below_vel=mph_to_mps(
            params["env"]["early_stop_when_below_mph"]
        ),
        pct_max_decel=params["env"]["pct_max_decel"],
        rho=rho,
    )

    return CarVars(
        car_dynamics, bf, bf_constraint, ctrl, a_tilde, safety_spec, env, params, calcs
    )
