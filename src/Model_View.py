import casadi
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from util import set_state
import numpy as np
from casadi import MX, vec, sumsqr

repo_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    model_name = "SLIP"
    model_xml_path = os.path.join(repo_path, "model/" + model_name + ".xml")
    model = load_model_from_path(model_xml_path)
    sim = MjSim(model)

    viewer = MjViewer(sim)

    z_min = -(1.45 - 1.45 * np.cos(np.deg2rad(-45)))

    set_state(sim, {"qpos": [-1.5, 0.0, np.deg2rad(0), 0.0]})
    sim.step()

    while True:
        viewer.render()
