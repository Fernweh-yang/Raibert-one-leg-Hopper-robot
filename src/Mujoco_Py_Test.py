from mujoco_py import load_model_from_path, MjSim
import os
from util import step_sim, sim_jacobian

repo_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    model_xml_path = os.path.join(repo_path, "model/DoDo.xml")
    model = load_model_from_path(model_xml_path)

    sim = MjSim(model)

    # Initialize Model
    step_sim(sim, 1000)

    jac = sim_jacobian(sim, n_steps=10000, eps=1e-6)

    print("Done")
