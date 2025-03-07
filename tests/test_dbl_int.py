import pickle
from pathlib import Path

import numpy as np
import tqdm

from safe_barrier_envs.dbl_int import (
    A_tilde,
    DblIntBarrierFunction,
    DblIntDynamics,
    DblIntSafeController,
    get_eta,
    get_N,
    make,
)
from safe_barrier_envs.utils import ActuatorLimits, BarrierConstraint, MathCalcs, NpAny


def test_a_tilde() -> None:

    for use_scalar in [False, True]:
        a_tilde = A_tilde(-0.5, 1, MathCalcs(use_scalar, True))

        assert np.isclose(a_tilde.get_a(1), a_tilde.a_minus)
        assert np.isclose(a_tilde.get_a(-1), a_tilde.a_plus)


def test_safe_controller() -> None:

    def _make_state(_full_state: bool, _v: float | int) -> NpAny:
        if _full_state:
            return np.array([0, _v])
        else:
            return np.array([_v])

    for use_scalar in [False, True]:
        for full_state in [False, True]:

            calcs = MathCalcs(use_scalar, True)
            dt = 0.1
            a_tilde = A_tilde(-0.5, 1, calcs)

            ctrl = DblIntSafeController(dt=dt, a_tilde=a_tilde, setpoint=0, calcs=calcs)

            large_vel = -2 * dt * a_tilde.a_minus
            u = ctrl.get_control(_make_state(full_state, large_vel))
            assert np.isclose(u, a_tilde.a_minus)

            small_vel = -dt * a_tilde.a_minus / 2
            u = ctrl.get_control(_make_state(full_state, small_vel))
            assert np.isclose(u, -small_vel / dt)

            large_neg_vel = -2 * dt * a_tilde.a_plus
            u = ctrl.get_control(_make_state(full_state, large_neg_vel))
            assert np.isclose(u, a_tilde.a_plus)

            small_neg_vel = -dt * a_tilde.a_plus / 2
            u = ctrl.get_control(_make_state(full_state, small_neg_vel))
            assert np.isclose(u, -small_neg_vel / dt)


def test_get_N() -> None:

    for use_scalar in [False, True]:
        calcs = MathCalcs(use_scalar, True)
        a_tilde = A_tilde(-1, 1, calcs)
        dt = 0.1

        def _get_N(_vel: float) -> float:
            return float(get_N(_vel, a_tilde, dt, calcs))

        thresh = dt * a_tilde.a_plus

        assert _get_N(0) == 0
        assert _get_N(thresh / 2) == 0
        assert _get_N(0.99 * thresh) == 0
        assert _get_N(thresh) == 1
        assert _get_N(1.01 * thresh) == 1
        assert _get_N(2.01 * thresh) == 2


def test_dynamics() -> None:
    dt = 0.1

    for use_scalar in [False, True]:

        calcs = MathCalcs(use_scalar, True)
        act_lims = ActuatorLimits([-1], [1], calcs)
        dynamics = DblIntDynamics(dt=dt, act_lims=act_lims, calcs=calcs)
        state = np.array([0, 1])
        action = 1

        next_state = dynamics.get_next_state(state, action)
        assert np.isclose(next_state[0], state[0] + dt * state[1])
        assert np.isclose(next_state[1], state[1] + dt * action)


def test_get_eta() -> None:

    def _test_helper(_vel: float, _a_tilde: A_tilde, _calcs: MathCalcs) -> None:
        _dt = 0.1
        _ctrl = DblIntSafeController(dt=_dt, a_tilde=_a_tilde, setpoint=0, calcs=_calcs)

        act_lims = ActuatorLimits([-1], [1], calcs=calcs)
        dynamics = DblIntDynamics(dt=_dt, act_lims=act_lims, calcs=calcs)
        state = np.array([0, _vel])
        N = int(get_N(_vel, _a_tilde, _dt, _calcs))

        orig_eta = get_eta(0, _vel, _a_tilde, _dt, _calcs)

        for _ in range(N):
            action = _ctrl.get_control(state)
            next_state = dynamics.get_next_state(state, action)
            next_p, next_v = next_state
            eta = get_eta(next_p, next_v, _a_tilde, _dt, _calcs)

            # bullet 3 of lemma 4
            assert np.isclose(eta, orig_eta)

            # bullet 2 of lemma 4
            if _vel >= 0:
                assert next_state[0] <= eta
            else:
                assert next_state[0] >= eta

            state = next_state

        # bullet 1 of lemma 4
        action = _ctrl.get_control(state)
        next_state = dynamics.get_next_state(state, action)

    for use_scalar in [False, True]:
        np.random.seed(0)

        calcs = MathCalcs(use_scalar, True)
        a_tilde = A_tilde(-1, 1, calcs)

        _test_helper(0.1, a_tilde, calcs)
        _test_helper(0.15, a_tilde, calcs)
        _test_helper(0.2, a_tilde, calcs)
        _test_helper(0.1, a_tilde, calcs)
        _test_helper(0.15, a_tilde, calcs)

        n = 1000
        for _ in tqdm.tqdm(range(n)):
            vel = np.random.uniform(-10, 10)
            a_tilde = A_tilde(
                np.random.uniform(-1, -0.01), np.random.uniform(0.01, 1), calcs
            )

            _test_helper(vel, a_tilde, calcs)


def test_env() -> None:
    dbl_int_vars = make(True, True, True, True, Path("tmp/dbl-int"), None)
    env = dbl_int_vars.env
    env.reset()
    env.step(0.1)
    env.step(0.1)
    env.step(0.1)


def test_bf() -> None:

    n = 10000

    for use_scalar in [False, True]:
        np.random.seed(0)
        for _ in tqdm.tqdm(range(n)):
            calcs = MathCalcs(use_scalar, True)
            lim = np.random.uniform(-1, 1)
            lim_is_min = bool(np.random.randint(2))
            a_tilde = A_tilde(
                np.random.uniform(-1, -0.01), np.random.uniform(0.01, 1), calcs
            )
            lmbda = np.random.uniform(0.1, 0.9)

            state = np.array([np.random.uniform(-1, 1), np.random.uniform(-10, 10)])
            dt = 0.1

            bf = DblIntBarrierFunction(lim, lim_is_min, a_tilde, dt=dt, calcs=calcs)
            act_lims = ActuatorLimits([-1], [1], calcs=calcs)
            dynamics = DblIntDynamics(dt, act_lims=act_lims, calcs=calcs)
            bf_constraint = BarrierConstraint(lmbda, dynamics, bf, calcs=calcs)
            ctrl = DblIntSafeController(dt=dt, a_tilde=a_tilde, setpoint=0, calcs=calcs)

            # put it on the edge of the safety boundary
            bf_val = bf.get_h(state)
            if not use_scalar:
                bf_val = float(bf_val[0])
            state[0] = float(state[0]) + (-bf_val if lim_is_min else bf_val)
            bf_val = bf.get_h(state)
            action = ctrl.get_control(state)

            constraint = bf_constraint.get_constraint(state, action)
            assert constraint >= -1e-9


def test_replay() -> None:
    path = Path("~/logs/exp/foo/dbl-int/trpo/s1-16/episode_hx-0.p").expanduser()
    with open(path, "rb") as f:
        obs_hx, rho_hx, ac_hx, margin_hx = pickle.load(f)

    worst_margin = margin_hx.min(axis=0)
    print(worst_margin)

    dbl_int_vars = make(True, True, True, True, Path("tmp/dbl-int"), None)
    bf_constraint = dbl_int_vars.bf_constraint

    for obs, ac in zip(obs_hx, ac_hx):
        h = bf_constraint.barrier_function.get_h(obs)
        constraint = bf_constraint.get_constraint(obs, ac, h)

        if constraint < 1e-6:
            print(f"constraint {constraint}")


if __name__ == "__main__":
    test_a_tilde()
    test_safe_controller()
    test_get_N()
    test_dynamics()
    test_bf()
    test_get_eta()
    test_env()
    # test_replay()
