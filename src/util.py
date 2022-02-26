from mujoco_py import MjSim, load_model_from_path
import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from casadi import Callback, Sparsity, vertsplit, DM, horzsplit
from joblib import Parallel, delayed
from multiprocessing import Pool
import cv2
from time import time
import os

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


class MjSimCasADi:
    def __init__(self, model_xml_path):
        model = load_model_from_path(model_xml_path)
        self.sim = MjSim(model)
        self.n_pos = self.sim.model.nq
        self.n_vel = self.sim.model.nv
        self.n_states = self.n_pos + self.n_vel
        self.n_ctrl = self.sim.model.nu
        self.n_waypoints = None
        self.n_steps = None
        self.fps = None
        self.sim_states = None
        self.sim_frames = None
        self.f = None

    def get_function(self, name: str, n_steps: int = 1, opts: dict = None):
        if opts is None:
            opts = {}

        self.f = MuJoCo2CasADi(name=name, mjsim=self.sim, n_steps=n_steps, opts=opts)

        return self.f

    def step_n(self, n_steps: int = 1, return_states: bool = False):
        if return_states:
            states = np.zeros((n_steps + 1 , self.n_states))

        for n in range(n_steps):

            if return_states:
                new_state = self.get_state()
                states[n, :self.n_pos] = new_state["qpos"]
                states[n, self.n_pos:] = new_state["qvel"]

            self.sim.step()

        if return_states:
            new_state = self.get_state()
            states[n_steps, :self.n_pos] = new_state["qpos"]
            states[n_steps, self.n_pos:] = new_state["qvel"]

        if return_states:
            return states

    def get_state(self):
        return get_state(self.sim)

    def set_state(self, state: dict = None):
        set_state(self.sim, state)

    def reset_ctrl(self):
        self.set_state({"ctrl": np.zeros(self.n_ctrl)})

    def simulate(self, n_waypoints: int, n_steps: int, u: np.array = None, x_init: np.array = None,
                 render: bool = False, width: int = 1280, height: int = 720, fps: int = 100):

        self.n_waypoints = n_waypoints
        self.n_steps = n_steps
        self.fps = fps

        # Initialize state of simulation
        if x_init is None:
            self.sim.reset()
        else:
            assert len(x_init.shape) == 1
            assert len(x_init) == self.n_states
            self.set_state({"qpos": x_init[:self.n_pos], "qvel": x_init[self.n_pos:]})

        if u is None:
            self.reset_ctrl()
        else:
            if len(u.shape) == 1:
                u = np.reshape(u, (u.shape[0], -1))
            assert u.shape[1] == self.n_ctrl
            assert u.shape[0] == n_waypoints

        states = np.zeros((n_waypoints * n_steps + 1, self.n_states))

        if render:
            frames_no = int(fps * n_waypoints * n_steps * self.sim.model.opt.timestep)
            frames = np.zeros((frames_no, height, width, 3), dtype="uint8")
            frame = 0

        steps_per_frame = int(1 / fps / self.sim.model.opt.timestep)

        for waypoint in range(n_waypoints):
            if u is not None:
                self.set_state({"ctrl": u[waypoint, :]})

            for step in range(n_steps):

                if render:
                    if (waypoint * n_steps + step) % steps_per_frame == 0:
                        frames[frame] = np.flip(self.sim.render(width=width, height=height, camera_name="render"),
                                                axis=0)
                        frame += 1

                curr_state = self.get_state()
                states[waypoint * n_steps + step, :self.n_pos] = curr_state["qpos"]
                states[waypoint * n_steps + step, self.n_pos:] = curr_state["qvel"]

                self.sim.step()

        curr_state = self.get_state()
        states[n_waypoints * n_steps, :self.n_pos] = curr_state["qpos"]
        states[n_waypoints * n_steps, self.n_pos:] = curr_state["qvel"]

        self.sim_states = states
        if render:
            self.sim_frames = frames

    def plot_simulation(self, title):
        fig, axes = plt.subplots(int(self.n_states / 2), 2, figsize=(13, 7))

        dt = self.sim.model.opt.timestep
        sim_time_len = (self.sim_states.shape[0] - 1) * dt

        axes = axes.T.flatten()

        for i in range(self.n_states):
            ax = axes[i]
            if i < self.f.states_n / 2:
                state_name = self.sim.model.joint_names[i]
            else:
                state_name = self.sim.model.joint_names[i - int(self.f.states_n / 2)] + "_p"

            ax.set_title(state_name)

            ax.plot(np.arange(start=0, stop=sim_time_len + dt, step=dt),
                    self.sim_states[:, i])

            ax.grid()

        fig.suptitle(title)

        return fig

    def render_simulation(self, video_path):

        if not os.path.isabs(video_path):
            video_path = os.path.join(repo_path, "Renderings", video_path)

        if not os.path.isdir(os.path.dirname(video_path)):
            os.makedirs(os.path.dirname(video_path))

        if not os.path.splitext(video_path)[1] == ".mp4":
            video_path = video_path + ".mp4"

        width = self.sim_frames.shape[2]
        height = self.sim_frames.shape[1]
        frames_no = self.sim_frames.shape[0]

        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))

        for i in range(frames_no):
            out.write(self.sim_frames[i])

        out.release()

        print("Video rendered to\n " + video_path)

    def plot_solution(self, sol_x: np.array, sol_u: np.array):
        dt = self.sim.model.opt.timestep
        n_steps = self.f.n_steps
        n = sol_u.shape[0]

        t = n * n_steps * dt

        figs = []
        fig, axes = plt.subplots(int(self.f.states_n / 2), 2, figsize=(13, 7))

        axes = axes.T.flatten()

        for i in range(self.f.states_n):
            ax = axes[i]

            if i < self.f.states_n / 2:
                state_name = self.sim.model.joint_names[i]
            else:
                state_name = self.sim.model.joint_names[i - int(self.f.states_n / 2)] + "_p"

            ax.set_title(state_name)
            ax.scatter(x=np.arange(start=0, stop=t + n_steps * dt, step=n_steps * dt),
                       y=sol_x[:, i],
                       marker="+",
                       label="Solution")
            for j in range(sol_x.shape[0] - 1):
                ax.scatter(x=n_steps * dt * (j + 1),
                           y=np.array(self.f(sol_x[j], sol_u[j]))[i, 0],
                           c="r",
                           marker="x",
                           alpha=0.5,
                           label="Solution Result")

            ax.grid()

        fig.suptitle("Solution | States")

        figs.append(fig)

        fig, axes = plt.subplots(self.f.inputs_n, 1, figsize=(13, 7))

        if len(sol_u.shape) == 1:
            sol_u = np.reshape(sol_u, (sol_u.shape[0], -1))

        for i in range(self.f.inputs_n):
            if self.f.inputs_n == 1:
                ax = axes
            else:
                ax = axes[i]

            ax.set_title("Control {}".format(i))
            for j in range(sol_u.shape[0] - 1):
                ax.plot([j * dt * n_steps, (j + 1) * dt * n_steps], [sol_u[j, i], sol_u[j, i]], "r")

            ax.grid()

        fig.suptitle("Solution | Control")

        figs.append(fig)

        return figs


class MuJoCo2CasADi(Callback):
    """CasADi Function Wrapper for MuJoCo

    """
    def __init__(self, name, mjsim, n_steps=1, opts: dict = None):
        if opts is None:
            opts = {}
        Callback.__init__(self)
        self.mjsim = mjsim
        self.states_n = self.mjsim.model.nq + self.mjsim.model.nv
        self.inputs_n = self.mjsim.model.nu
        self.n_steps = n_steps
        self.construct(name, opts)

    def get_n_in(self):
        return 2

    def get_name_in(self, i):
        if i == 0:
            return "x_0"
        if i == 1:
            return "u"

    def get_name_out(self, i):
        return "x_{}".format(self.n_steps)

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return Sparsity.dense(self.states_n, 1)
        if i == 1:
            return Sparsity.dense(self.inputs_n, 1)

    def get_sparsity_out(self, i):
        return Sparsity.dense(self.states_n)

    def get_n_states(self):
        return self.mjsim.model.nq + self.mjsim.model.nv

    def get_n_inputs(self):
        return self.mjsim.model.nu

    def eval(self, arg):
        """ Set the state and the actuator values of the simulation and step the simulation

        :param arg: arg[0] should contain the position, velocities and actuator inputs
        :return: State of the simulation after step
        """
        # Decode States and Inputs
        states = vertsplit(arg[0])
        inputs = vertsplit(arg[1])

        nq = self.mjsim.model.nq

        # Prepare dict defining the initial simulation state
        sim_state = {
            "qpos": states[:nq],
            "qvel": states[nq:],
            "ctrl": inputs
        }

        # Set the state of the simulation
        set_state(self.mjsim, sim_state)

        # Step the simulation
        step_sim(self.mjsim, self.n_steps)

        # Get the resulting end state
        new_state = get_state(self.mjsim)

        output = []

        output.extend(new_state["qpos"])
        output.extend(new_state["qvel"])

        # Encode Output
        output = DM(output)

        return [output]


def plot_states(states: np.array, fig=None, show=True):
    n_states = states.shape[1]

    if fig is None:
        n_cols = 4
        n_rows = 5
        fig, _ = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    axs = fig.axes

    for n in range(n_states):
        axs[n].plot(states[:, n])

    if show:
        fig.show()

    return fig


def get_state(sim: MjSim):
    state = dict()

    state["qpos"] = copy(sim.data.qpos)
    state["qvel"] = copy(sim.data.qvel)
    state["qacc"] = copy(sim.data.qacc)
    state["qacc_warmstart"] = copy(sim.data.qacc_warmstart)
    state["qfrc_applied"] = copy(sim.data.qfrc_applied)
    state["xfrc_applied"] = copy(sim.data.xfrc_applied)
    state["ctrl"] = copy(sim.data.ctrl)

    return state


def set_state(sim: MjSim, state: dict = None):
    if state is not None:
        for par_name, values in state.items():
            for i, value in enumerate(values):
                getattr(sim.data, par_name)[i] = copy(value)


def step_sim(sim: MjSim, n_steps: int = 1, return_states=False):
    if return_states:
        pos_states_n = len(get_state(sim)["qpos"])
        vel_states_n = len(get_state(sim)["qvel"])
        states = np.zeros((n_steps, pos_states_n + vel_states_n))

    for n in range(n_steps):
        sim.step()

        if return_states:
            states[n, :pos_states_n] = get_state(sim)["qpos"]
            states[n, pos_states_n:] = get_state(sim)["qvel"]

    if return_states:
        return states




