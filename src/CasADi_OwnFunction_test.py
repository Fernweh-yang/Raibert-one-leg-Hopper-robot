import casadi
from mujoco_py import load_model_from_path, MjSim
import os
from util import MuJoCo2CasADi
from casadi import MX, Function, jacobian, vertcat
from time import time
import numpy as np

repo_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    model_xml_path = os.path.join(repo_path, "model/TwoMassSpringResonator.xml")
    model = load_model_from_path(model_xml_path)

    sim = MjSim(model)

    f = MuJoCo2CasADi("f", sim, 10, {"enable_fd": True, "fd_method": "forward"})
    f_own = MuJoCo2CasADi("f_own", sim, 10)

    print(f([0.1, 0.1, 0.1, 0.1], [0, 0]))
    print(f_own([0.1, 0.1, 0.1, 0.1], [0, 0]))

    x = MX.sym("x", 4)
    u = MX.sym("u", 2)

    j = Function("j", [x, u], [jacobian(f(x, u), vertcat(x, u))])
    j_own = Function("j_own", [x, u], [jacobian(f_own(x, u), vertcat(x, u))])

    # Initialize
    jac = j([0, 0, 0, 0], [0, 0])
    jac_own = j_own([0, 0, 0, 0], [0, 0])

    # Measure time
    t1 = time()
    for i in range(100):
        jac_own = j_own([0, 0, 0, 0], [0, 0])
    t2 = time()
    print(f"100x Own Jacobian executed in {(t2 - t1):.4f}s")

    t1 = time()
    for i in range(100):
        jac = j([0, 0, 0, 0], [0, 0])
    t2 = time()
    print(f"100x CasADi Jacobian executed in {(t2 - t1):.4f}s")

    print(jac)
    print(jac_own)

    print("Max Diff: {}".format(np.abs(jac - jac_own).max()))

    print("Done")
