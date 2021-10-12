import time
import ray
import argparse
import matplotlib.pyplot as plt

from wheeledSim.envs.pybullet_sim import WheeledSimEnv

@ray.remote
class Process:
    def __init__(self, envspec):
        self.env = WheeledSimEnv(envspec, T=100, render=False)
        print('initialized process on physics id {}'.format(self.env.env.physicsClientId))

    def get_traj(self):
        self.env.reset()
        t = False
        traj = []
        while not t:
            o, r, t, i = self.env.step([1., 0.])
            traj.append(o)
        return traj

def get_timing(N):
    _t = time.time()

    processes = [Process.remote(args.config) for i in range(N)]
    trajs = [process.get_traj.remote() for process in processes]
    trajs = ray.get(trajs)

    _et = time.time() - _t
    nsteps = 100*N
    return _et, nsteps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    ray.init()

    times = []
    steps = []
    for i in range(args.N):
        t, s = get_timing(i)
        times.append(t)
        steps.append(s)

    physics_steps = [x * 50 for x in steps]
    steps_per_second = [x/(10*t) for x,t in zip(steps, times)]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(range(1, args.N+1), steps, marker='.')
    axs[1].plot(range(1, args.N+1), steps_per_second, marker='.')
    axs[2].plot(range(1, args.N+1), [2.5*x for x in range(args.N)], marker='.')

    axs[0].set_title('Total steps')
    axs[1].set_title('Sim seconds per second')
    axs[2].set_title('CPU RAM (ESTIMATE)')

    for ax in axs:
        ax.set_xlabel('Number sims')

    axs[0].set_ylabel('Steps')
    axs[1].set_ylabel('Steps/Sec')
    axs[2].set_ylabel('GB')

    plt.show()
