import casadi
from mujoco_py import load_model_from_path, MjSim
import os
from util import MuJoCo2CasADiSingleShooting
from casadi import MX, Function, jacobian, vertcat, vec

repo_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    model_name = "TwoMassSpringResonator"
    # model_name = "DoDo"
    model_xml_path = os.path.join(repo_path, "model/" + model_name + ".xml")
    model = load_model_from_path(model_xml_path)

    sim = MjSim(model)

    T = 1.0
    N = 50
    muj_dt = sim.model.opt.timestep
    n_steps = int(T/N/muj_dt)

    opts = {"enable_fd": True, "fd_method": "forward"}

    f = MuJoCo2CasADiSingleShooting(name="f", mjsim=sim, cntrl_int=N, n_steps=n_steps, opts=opts)

    opti = casadi.Opti()

    x = opti.parameter(f.states_n, 1)
    u = opti.variable(f.inputs_n, N)

    opti.subject_to(f(x, u)[0:2] == x[0:2])

    opti.subject_to(vec(u) < 100)
    opti.subject_to(vec(u) > -100)

    opti.solver('ipopt')

    opti.set_value(x, [0.0]*f.states_n)
    sol = opti.solve()

    print("Done!")
