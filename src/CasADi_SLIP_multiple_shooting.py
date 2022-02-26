import casadi
from mujoco_py import load_model_from_path, MjSim, ignore_mujoco_warnings
import os
from util import MjSimCasADi
from casadi import MX, vec, sumsqr
import numpy as np
import matplotlib.pyplot as plt

repo_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    model_name = "SLIP"
    model_xml_path = os.path.join(repo_path, "model/" + model_name + ".xml")
    sim = MjSimCasADi(model_xml_path)

    T = 1.5

    N = int(T*100)
    muj_dt = sim.sim.model.opt.timestep
    n_steps = int(T/N/muj_dt)

    opts = {"enable_fd": True, "fd_method": "forward"}

    f = sim.get_function(name="f", n_steps=n_steps, opts=opts)

    opti = casadi.Opti()

    x = opti.variable(f.states_n, N+1)
    u = opti.variable(f.inputs_n, N)

    ##################
    # Start Position
    ##################

    # X
    opti.subject_to(x[0, 0] == -1.5)

    # Z
    opti.subject_to(x[1, 0] > 0.5)

    # Spring rot
    opti.subject_to(x[2, 0] > np.deg2rad(-30.0))
    opti.subject_to(x[2, 0] < np.deg2rad(-5.0))

    # X_p
    opti.subject_to(x[4, 0] > 1.0)

    # Z_p
    opti.subject_to(x[5, 0] == 0.0)

    ##################
    # Feasibility
    ##################
    max_penetration = 0.01

    # Contact between floor and body
    opti.subject_to(vec(x[1, :]) > -1.2 - max_penetration)

    # Contact between floor and spring
    opti.subject_to(vec(x[1, :]) > vec(-(1.45 - (1.45 - x[3, :]) * casadi.cos(x[2, :])) - max_penetration))

    ##################
    # End Position
    ##################

    # X
    opti.subject_to(x[0, -1] == 1.5)

    # Cycle
    opti.subject_to(vec(x[1:, -1]) == vec(x[1:, 0]))

    # Transitions
    for k in range(1, N+1):
        opti.subject_to(x[:, k] == f(x[:, k-1], u[:, k-1]))

    # Limit Control
    opti.subject_to(vec(u) < 1000)
    opti.subject_to(vec(u) > -1000)

    # Cost Function
    J = sumsqr(u)
    opti.minimize(J)

    opti.solver('ipopt', {}, {"max_iter": 50})

    with ignore_mujoco_warnings():
        try:
            sol = opti.solve()
        except RuntimeError:
            print("No optimal solution has been found")
            sol_u = opti.debug.value(u).T
            sol_x = opti.debug.value(x).T
        else:
            sol_u = sol.value(u).T
            sol_x = sol.value(x).T

    sol_figs = sim.plot_solution(sol_x, sol_u)

    sol_figs[0].savefig(fname=model_name + "solution_states" + "_T{}_N{}".format(T, N), format="png")
    sol_figs[1].savefig(fname=model_name + "solution_control" + "_T{}_N{}".format(T, N), format="png")

    sim.simulate(n_waypoints=N, n_steps=n_steps, x_init=sol_x[0, :], u=sol_u, render=True)

    sim_fig = sim.plot_simulation(title="Simulation with optimized control")
    sim_fig.savefig(fname=model_name + "simulation" + "_T{}_N{}".format(T, N), format="png")

    sim.render_simulation(model_name + "_T{}_N{}".format(T, N))

    print("Done!")
