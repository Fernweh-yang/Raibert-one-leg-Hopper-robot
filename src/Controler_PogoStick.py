import time
import casadi
import mujoco_py as mp
import os
from util import MuJoCo2CasADi,set_state
import numpy as np

repo_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    # Set Model
    model_name = "PogoStick"
    model_xml_path = os.path.join(repo_path, "model/" + model_name + ".xml")
    model = mp.load_model_from_path(model_xml_path)
    sim = mp.MjSim(model)
    viewer = mp.MjViewer(sim)

    # Configure Controller
    T = 0.2
    N = 5
    muj_dt = sim.model.opt.timestep
    n_steps = int(T/N/muj_dt)
    opts = {"enable_fd": True, "fd_method": "forward"}
    f = MuJoCo2CasADi(name="f", mjsim=sim, n_steps=n_steps, opts=opts)
    opti = casadi.Opti()
    x = opti.variable(f.states_n, N+1)
    u = opti.variable(f.inputs_n, N)
    # ##Start Position
    opti.subject_to(x[:, 0] == casadi.MX([0]*f.states_n))
    # ##End Position
    opti.subject_to(x[0, -1] == casadi.MX(0.0))
    # ##Transitions
    for k in range(1, N+1):
        opti.subject_to(x[:, k] == f(x[:, k-1], u[:, k-1]))
    # ##Limit Control
    opti.subject_to(casadi.vec(u) < 1000)
    opti.subject_to(casadi.vec(u) > -1000)
    # ##Calculate
    opti.solver('ipopt')
    sol = opti.solve()
    u = sol.value(u).T
    if len(u.shape) == 1:
        u = np.reshape(u, (u.shape[0], -1))

    # Control
    for i in range(u.shape[0]):
        sim_state = {"ctrl": u[i, :]}
        set_state(sim,sim_state)
        for step in range(n_steps):
            sim.step()
            viewer.render()
            time.sleep(0.02)

