import casadi
from mujoco_py import load_model_from_path, MjSim
import os
from util import MuJoCo2CasADi, render_simulation, simulate_with_control, MjSimCasADi, plot_solution
from casadi import MX, vec, sumsqr
import numpy as np
import matplotlib.pyplot as plt

repo_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    model_name = "PogoStick"
    model_xml_path = os.path.join(repo_path, "model/" + model_name + ".xml")
    sim = MjSimCasADi(model_xml_path)

    T = 0.6
    N = int(T*100)
    muj_dt = sim.sim.model.opt.timestep
    n_steps = int(T/N/muj_dt)

    opts = {"enable_fd": True, "fd_method": "forward"}

    f = sim.get_function(name="f", n_steps=n_steps, opts=opts)

    opti = casadi.Opti()

    x = opti.variable(f.states_n, N+1)
    u = opti.variable(f.inputs_n, N)

    # Start Position
    # opti.subject_to(x[:, 0] == MX([0]*f.states_n))
    # opti.subject_to(x[0, 0] == MX(0.5))
    opti.subject_to(x[1, 0] == MX(0.0))
    opti.subject_to(x[2, 0] == MX(0.0))
    opti.subject_to(x[3, 0] == MX(0.0))

    # Middle Position
    # opti.subject_to(x[2, int(x.shape[1]/2)] == MX(0.0))

    # End Position
    # opti.subject_to(x[:, -1] == MX([0] * f.states_n))
    # opti.subject_to(x[0, -1] == MX(0.5))
    # opti.subject_to(x[1, -1] == MX(0.0))
    # opti.subject_to(x[2, -1] == MX(0.0))
    # opti.subject_to(x[3, -1] == MX(0.0))

    # Periodicity
    opti.subject_to(x[:, 0] == x[:, -1])

    # Transitions
    for k in range(1, N+1):
        opti.subject_to(x[:, k] == f(x[:, k-1], u[:, k-1]))

    # Cost Function
    J = sumsqr(u)
    opti.minimize(J)

    # Warm Start
    sim.simulate(n_waypoints=N, n_steps=n_steps)
    opti.set_initial(x, sim.sim_states[0::n_steps].T)

    opti.solver('ipopt', {}, {"max_iter": 300})

    try:
        sol = opti.solve()
    except RuntimeError:
        print("No optimal solution has been found")
        sol_u = opti.debug.value(u).T
        sol_x = opti.debug.value(x).T
    else:
        sol_u = sol.value(u).T
        sol_x = sol.value(x).T

    sim.simulate(n_waypoints=N, n_steps=n_steps, render=True, x_init=sol_x[0, :])

    sim_fig = sim.plot_simulation()

    sim_fig.suptitle("Simulation with no control")
    sim_fig.show()

    sim.render_simulation(model_name + "_T{}_N{}_no_control".format(T, N))

    sol_figs = plot_solution(T, muj_dt, n_steps, sol_x, sol_u, f)

    sol_figs[0].suptitle("Optimization Results | States")
    sol_figs[0].show()

    sol_figs[1].suptitle("Optimization Results | Control")
    sol_figs[1].show()

    sim.simulate(n_waypoints=N, n_steps=n_steps, x_init=sol_x[0, :], u=sol_u, render=True)

    sim_fig = sim.plot_simulation()

    sim_fig.suptitle("Simulation with optimized control")
    sim_fig.show()

    sim.render_simulation(model_name + "_T{}_N{}".format(T, N))

    print("Done!")
