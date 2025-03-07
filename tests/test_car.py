import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import unitpy

from safe_barrier_envs.car import CarVars, LeadCarMode, get_car_offset, make
from safe_barrier_envs.dbl_int import A_tilde
from safe_barrier_envs.utils import BarrierConstraint, NpFloat, NumType, T


def test_beta() -> None:

    for use_scalar in [False, True]:
        car_vars = make(use_scalar, True, False, False, None, None)
        calcs = car_vars.env.calcs
        act_lims = car_vars.car_dynamics.act_lims
        car_params = car_vars.car_dynamics.car_params

        for u2 in np.linspace(act_lims.low[1], act_lims.high[1]):
            _type = calcs.get_type(u2)
            out = car_params.get_beta_inv(car_params.get_beta(u2, _type), _type)
            assert np.isclose(out, u2, atol=1e-5)


def test_dynamics() -> None:

    for use_scalar in [False, True]:
        car_vars = make(use_scalar, True, False, False, None, None)

        dynamics = car_vars.car_dynamics

        state = [1, 2, 3, np.pi / 3]
        if not use_scalar:
            state = np.array(state)

        _type = NumType.SCALAR if use_scalar else NumType.NUMPY
        action = car_vars.safe_controller.get_control(state, _type)
        next_state = dynamics.get_next_state(state, action)

        next_x, next_y, next_v, next_psi = next_state

        if use_scalar:
            assert isinstance(next_x, float)
            assert isinstance(next_y, float)
            assert isinstance(next_v, float)
            assert isinstance(next_psi, float)

        assert next_x >= 1
        assert next_y >= 2
        assert next_v <= 3
        assert next_psi <= np.pi / 3


class BFWrapper:
    def __init__(self, bf: Any, idx: int):
        self.bf = bf
        self.idx = idx

    def __repr__(self) -> str:
        return f"BFWrapper(bf={self.bf}, idx={self.idx})"

    def get_h(self, state: T) -> T:
        return self.bf.get_h(state)[self.idx]


def test_bf_lane() -> None:

    psi_lim = np.deg2rad(1)
    for use_scalar in [False, True]:
        np.random.seed(0)
        car_vars = make(use_scalar, True, False, False, None, None)
        calcs = car_vars.env.calcs
        bf_lane = car_vars.barrier_function.h_lane
        dynamics = car_vars.car_dynamics
        W_car = dynamics.car_params.width

        def _check_constraint(_state: NpFloat, _y_offset: NpFloat, _idx: int) -> None:
            _bf = BFWrapper(bf_lane, _idx)
            _y_offset = calcs.scalar_cast(_y_offset)[0]
            _state = np.array([_state[0], _state[1] + _y_offset, state[2], state[3]])
            _constraint = BarrierConstraint(
                car_vars.bf_constraint.lmbda, dynamics, _bf, calcs
            )
            _ac = car_vars.safe_controller.get_control(state)
            assert _constraint.get_constraint(_state, _ac) >= -1e-6

        num_steps = []

        for _ in tqdm.tqdm(range(1000)):
            y = np.random.uniform(W_car, bf_lane.W_lane - W_car)
            v = np.random.uniform(0, car_vars.safety_spec.v_lim)
            idxs = (2, 3) if y >= bf_lane.W_lane else (0, 1)

            # first check that the barrier is nonnegative when not turning
            state = np.array([0, y, v, 0])
            h_lane = bf_lane.get_h(state, False)
            assert all(h_lane[i] >= 0 for i in idxs)

            # now check when there is turning
            psi = np.random.uniform(-psi_lim, psi_lim)
            state = np.array([0, y, v, psi])
            h_lane = bf_lane.get_h(state, True)

            num_steps.append(bf_lane.num_steps)

            _check_constraint(state, -h_lane[0], 0)
            _check_constraint(state, h_lane[1], 1)
            _check_constraint(state, -h_lane[2], 2)
            _check_constraint(state, h_lane[3], 3)

        print(np.max(num_steps), np.mean(num_steps))


def test_bf_lead() -> None:

    def _sample_state(_v_lim: float) -> NpFloat:
        _velocities = np.random.uniform(0, _v_lim, 3)
        _states = [np.array([0, 0, _v, 0]) for _v in _velocities]
        return np.hstack(_states)

    def _check_constraint(
        _all_veh_state: NpFloat, _x_offset: float, _idx: int, _car_vars: CarVars
    ) -> None:

        _act_lims = _car_vars.car_dynamics.act_lims
        _calcs = _act_lims.calcs

        _all_veh_state = _all_veh_state.copy()
        _x_c_idx = 8
        _all_veh_state[_x_c_idx] += _calcs.scalar_cast(_x_offset)[0]

        _constraint = copy.deepcopy(_car_vars.bf_constraint)
        _constraint.barrier_function = BFWrapper(
            _car_vars.barrier_function.h_lead, _idx
        )

        # get control
        _state_a, _state_b, _state_c = _calcs.split(all_veh_state, 3, 12)
        _a_tilde_j = A_tilde(_act_lims.low[0], _act_lims.high[0], _calcs)

        _ac = _car_vars.safe_controller.get_control(_state_c)
        assert _constraint.get_constraint(_all_veh_state, _ac) >= -1e-6

    for use_scalar in [False, True]:
        np.random.seed(0)
        car_vars = make(use_scalar, True, False, False, None, None)

        n = 1000
        for _ in tqdm.tqdm(range(1000)):
            all_veh_state = _sample_state(car_vars.safety_spec.v_lim)
            bf_vals = car_vars.barrier_function.h_lead.get_h(all_veh_state)

            _check_constraint(all_veh_state, bf_vals[0], 0, car_vars)
            _check_constraint(all_veh_state, bf_vals[1], 1, car_vars)


def test_bf_speed() -> None:

    for use_scalar in [False, True]:
        car_vars = make(use_scalar, True, False, False, None, None)
        calcs = car_vars.env.calcs
        h_speed = car_vars.barrier_function.h_speed
        constraint = BarrierConstraint(
            car_vars.bf_constraint.lmbda, car_vars.car_dynamics, h_speed, calcs
        )

        for v in [0, car_vars.safety_spec.v_lim]:
            state = np.array([0, 0, v, np.pi / 3])
            action = car_vars.safe_controller.get_control(state)

            h_val = h_speed.get_h(v)
            assert h_val >= 0
            assert constraint.get_constraint(state, action) > 0


def test_lead_car() -> None:

    for use_scalar in [False, True]:
        np.random.seed(0)
        car_vars = make(use_scalar, True, False, False, None, None)

        assert car_vars.env.lead_car_a.lane_idx == "a"
        assert car_vars.env.lead_car_b.lane_idx == "b"

        lead_car = car_vars.env.lead_car_a

        state = np.zeros(4)
        lead_car.reset(state)

        assert state[0] > 0
        assert np.isclose(state[1], car_vars.env.W_lane / 2)
        assert state[2] > 0
        assert np.isclose(state[3], 0)
        assert lead_car.mode == LeadCarMode.CRUISE

        action = lead_car.step(state, 0, 0)
        assert lead_car.mode == LeadCarMode.CRUISE
        assert np.allclose(action, [0, 0])

        setpoint = lead_car.ctrl.setpoint
        action = lead_car.step(state, 0, lead_car.time_end_cruise + 1e-9)
        assert lead_car.mode == LeadCarMode.BRAKE
        assert not np.isclose(setpoint, lead_car.ctrl.setpoint)

        dynamics = car_vars.car_dynamics

        def _loop(_state: NpFloat, _next_mode: LeadCarMode) -> None:

            _prev_err = abs(_state[2] - lead_car.ctrl.setpoint)
            _curr_mode = lead_car.mode
            while True:

                _prev_setpoint = lead_car.ctrl.setpoint
                _action = np.hstack((lead_car.step(_state, 0, 0), 0))
                _state = dynamics.get_next_state(_state, _action)

                if lead_car.mode != _curr_mode:
                    assert lead_car.mode == _next_mode
                    break

                _err = abs(_state[2] - _prev_setpoint)
                assert _err < _prev_err
                _prev_err = _err
                assert lead_car.mode == _curr_mode

            return _state

        state = _loop(state, LeadCarMode.ACCELERATE)
        state = _loop(state, LeadCarMode.CRUISE)


def test_env_vel() -> None:

    for use_scalar in [False, True]:
        np.random.seed(0)
        car_vars = make(use_scalar, True, False, False, None, None)

        env = car_vars.env
        env.reset()

        noop = np.array([0, 0])
        decel = np.array([-1, 0])
        accel = np.array([1, 0])

        rew_noop = env.step(noop)[1]

        rew_decel = env.step(decel)[1]
        assert rew_decel < rew_noop

        rew_accel = env.step(accel)[1]
        assert rew_decel < rew_accel


def test_env_crash() -> None:

    for use_scalar in [False, True]:
        np.random.seed(0)
        car_vars = make(use_scalar, True, False, False, None, None)

        env = car_vars.env
        env.reset()

        accel = np.array([1, 0])
        done = False

        for _ in range(1000):
            done, info = env.step(accel)[2:]

            if done:
                assert info["constraint_violation"]
                assert info["exit_crash"]
                break

        if not done:
            raise RuntimeError("the environment should have exited")


def test_lane_change() -> None:
    # verify the barrier is not active given it the car has
    # just entered the top lane and is pointing at a reasonable angle
    reasonable_angle = np.deg2rad(1)

    for use_scalar in [False, True]:
        car_vars = make(use_scalar, True, False, False, None, None)
        env = car_vars.env
        env.reset()

        ctrl_dbl_int = copy.deepcopy(car_vars.safe_controller.ctrl_dbl_int)
        a_tilde = ctrl_dbl_int.a_tilde
        dt = env.car_dynamics.dt
        car_params = env.car_dynamics.car_params

        v = (70.0 * unitpy.Unit("mile per hour")).to("m/s").value
        offset = get_car_offset(car_params, reasonable_angle, env.calcs)
        y = env.W_lane + offset

        bf_lane = car_vars.barrier_function.h_lane

        state = np.array([0, y, v, reasonable_angle])
        excess = bf_lane.get_h(state).H2
        print(excess)
        assert excess > 0


def test_replay() -> None:
    path = Path("~/logs/exp/foo/car/trpo/s1-7/episode_hx-0.p").expanduser()
    with open(path, "rb") as f:
        obs_hx, rho_hx, ac_hx, margin_hx = pickle.load(f)

    worst_margin = margin_hx.min(axis=0)
    print(worst_margin)
    car_vars = make(True, True, True, False, Path("tmp/dbl-int"), None)
    bf_constraint = car_vars.bf_constraint

    for _, (obs, ac) in enumerate(zip(obs_hx, ac_hx)):
        h = bf_constraint.barrier_function.get_h(obs)
        constraint = bf_constraint.get_constraint(obs, ac, h)

        if constraint < 1e-6:
            import lvdb; lvdb.set_trace()  # fmt: skip


if __name__ == "__main__":
    test_beta()
    test_dynamics()
    test_bf_lane()
    test_bf_lead()
    test_bf_speed()
    test_lead_car()
    test_env_vel()
    test_env_crash()
    test_lane_change()
    # test_replay()
