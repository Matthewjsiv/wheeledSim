import gym
import numpy as np
import torch
import pybullet
import yaml
import argparse
import matplotlib.pyplot as plt
import pybullet as p

from wheeledSim.pybullet_sim import WheeledSimEnv

def generate_step_functions(T=50, throttle_n=3, steer_n=3):
    """
    Build the action sequences for which we will get step functions.
    """
    seqs = torch.zeros(2*throttle_n + 2*steer_n, T, 2) #[throttle x steer x time x actdim]
    throttles = torch.cat([
        torch.linspace(-1, 0, throttle_n+1)[:-1],
        torch.linspace(0, 1, throttle_n+1)[1:]
    ])

    steers = torch.cat([
        torch.linspace(-1, 0, steer_n+1)[:-1],
        torch.linspace(0, 1, steer_n+1)[1:]
    ])

    for i, th in enumerate(throttles):
        seqs[i, int(T//2):, 0] = th

    for i, st in enumerate(steers):
        seqs[i + len(throttles), :, 0] = 1.
        seqs[i + len(throttles), int(T//2):, 1] = st

    return seqs

def get_labels(robot):
    """
    Extract linear velocity and steering angle from the simulator.
    Linear velocity: project onto yaw
    Steer angle: diff the wheel and body position.
    """
    v = robot.getBaseVelocity_body()[0]
    delta = p.getJointState(robot.cliffordID, robot.jointNameToID['axle2frwheel'])[0]
    return torch.tensor([v, delta])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_as', type=str, required=True, help="What to save the data file as")
    parser.add_argument('--T', type=int, required=True, help="Number of timesteps per step func (the step happens at the halfway point)")
    parser.add_argument('--throttle_n', type=int, required=False, default=5, help="Number of different throttle excitations to try (2x this, for positive and negative.)")
    parser.add_argument('--steer_n', type=int, required=False, default=5, help="Number of different steer positions to try (for now, once at max throttle)")

    args = parser.parse_args()

    """load environment"""
    config_file = "../configurations/sysidEnvParams.yaml"
    env = WheeledSimEnv(config_file, T=50, render=False)

    step_funcs = generate_step_functions(args.T, args.throttle_n, args.steer_n)

    data_buf = {
        'observation':[],
        'action':[],
        'sysid_labels':[], #[v, delta]
    }

    """collect step responses"""
    for e, acts in enumerate(step_funcs):
        print("Step func {}/{}".format(e + 1, len(step_funcs)))
        states = [torch.tensor(env.reset())]
        sysid_labels = []
        
        for act in acts:
            no, r, t, i = env.step(act)
            labels = get_labels(env.robot)
            states.append(torch.tensor(no['state']))
            sysid_labels.append(labels)

        states = torch.stack(states, dim=0)
        sysid_labels = torch.stack(sysid_labels, dim=0)
        data_buf['observation'].append(states)
        data_buf['action'].append(acts)
        data_buf['sysid_labels'].append(sysid_labels)

    import pdb;pdb.set_trace()
    data_buf['observation'] = torch.stack(data_buf['observation'], dim=0).float()
    data_buf['action'] = torch.stack(data_buf['action'], dim=0).float()
    data_buf['sysid_labels'] = torch.stack(data_buf['sysid_labels'], dim=0).float()
    data_buf['car_params'] = env.robot.params
    torch.save(data_buf, args.save_as)
