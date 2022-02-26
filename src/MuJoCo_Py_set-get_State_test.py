from mujoco_py import load_model_from_path, MjSim
import os
from util import step_sim, sim_jacobian, plot_states, get_state, set_state

repo_path = os.path.dirname(os.path.dirname(__file__))


if __name__ == "__main__":
    model_xml_path = os.path.join(repo_path, "model/DoDo.xml")
    model = load_model_from_path(model_xml_path)

    sim = MjSim(model)

    step_sim(sim, 100)

    sim2 = MjSim(model)

    set_state(sim2, get_state(sim))

    states = step_sim(sim, 1000, return_states=True)
    states2 = step_sim(sim2, 1000, return_states=True)

    diff = states - states2
    print(diff.max())

    fig = plot_states(states, show=False)
    plot_states(states, fig)

    print("Done")
