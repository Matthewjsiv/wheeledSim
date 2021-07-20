import gym
import numpy as np
import pybullet
import yaml

from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.simController import simController
from wheeledSim.sensors.front_camera_sensor import FrontCameraSensor
from wheeledSim.sensors.lidar_sensor import LidarSensor
from wheeledSim.sensors.local_heightmap_sensor import LocalHeightmapSensor
from wheeledSim.sensors.shock_travel_sensor import ShockTravelSensor

from wheeledSim.envs.pybullet_sim import WheeledSimEnv

from sysid.kinematic_bicycle_model import KBMKinematics
from sysid.sysid_eval import ARXTransferFunction

import time
import torch
import os

import cProfile

PLOTLOCK = False #whether or not to show plot (showing plot slows down)
PLOTRATE = 1 #how often to plot (wait n steps each time)
EXP_LENGTH = 400 #how many steps to run simulation
BUFF = False #whether or not to show only recent steps in plotting

ITA = 100
BEHAVIOR = 'RWM'

sensor_str_to_obj = {
    'FrontCameraSensor':FrontCameraSensor,
    'LidarSensor':LidarSensor,
    'LocalHeightmapSensor':LocalHeightmapSensor,
    'ShockTravelSensor':ShockTravelSensor
}



class TFexpert():
    def __init__(self, fpath, dt= .1):
        self.model = KBMKinematics()
        self.throttle = torch.load(fpath + '_throttle.pt')
        self.steer = torch.load(fpath + '_steer.pt')
        self.dt = dt

        self.throttle.set_grad(False)
        self.steer.set_grad(False)



    #DOUBLECHECK THIS IS ONLY CALLED ONCE PER ITERATION
    #Otherwise will mess up buffer
    #not sure if -1 goes to theta or to -d
    def forward(self, obsState, ctrl):
        x = obsState[0]
        y = obsState[1]
        q = obsState[3:7]
        thetan = np.arctan2(2.0 * (q[2]*q[3] + q[0]*q[1]), -1.0 + 2.0*(q[3]*q[3] + q[0]*q[0]))#*180/np.pi - 90
        theta = 1*pybullet.getJointState(env.robot.cliffordID, env.robot.jointNameToID['axle2frwheel'])[0]
        theta = torch.tensor(theta)

        state = torch.tensor([x,y,thetan])
        vel = self.throttle.forward(torch.tensor(np.linalg.norm(obsState[7:10])),ctrl[0])
        d = self.steer.forward(theta,ctrl[1])
        tfstate = self.model.forward_dynamics(state,torch.tensor([vel,-d]),self.dt)
        return tfstate

    def future(self, obsState, cmd, nsteps = 5):
        x = obsState[0]
        y = obsState[1]
        q = obsState[3:7]
        thetan = np.arctan2(2.0 * (q[2]*q[3] + q[0]*q[1]), -1.0 + 2.0*(q[3]*q[3] + q[0]*q[0]))#*180/np.pi - 90
        theta = 1*pybullet.getJointState(env.robot.cliffordID, env.robot.jointNameToID['axle2frwheel'])[0]
        theta = torch.tensor(theta)

        state = torch.tensor([x,y,thetan])
        vels = self.throttle.future(torch.tensor(np.linalg.norm(obsState[7:10])),cmd[0], nsteps = nsteps)
        ds = self.steer.future(theta,cmd[1], nsteps = nsteps)

        tfstates = np.empty([0,3])
        for i in range(len(vels)):
            state = self.model.forward_dynamics(state, torch.tensor([vels[i],-ds[i]]), self.dt)
            tfstates = np.vstack([tfstates, state])

        return tfstates



class EnsembleLearner():

    def __init__(self, behavior='RWM', ita=.2):
        self.experts = []

        self.total_learner_loss = 0
        self.total_avg_loss = 0
        self.learner_loss_plt = []
        self.experts_loss_plt = []
        self.avg_loss_plt = []

        self.behavior = behavior
        self.ita = ita
        self.t = 0

    def reset(self):
        self.total_learner_loss = 0
        self.total_avg_loss = 0
        self.learner_loss_plt = []
        self.experts_loss_plt = []
        self.avg_loss_plt = []

    def buildTfEnsemble(self,fpath, dt = .1, dn = 1):
        # FNAMES = os.listdir(fpath)
        #
        # for i in range(len(FNAMES)):
        #     #FNAMES[i] = FNAMES[i][0:4]
        #     FNAMES[i] = FNAMES[i].replace('_throttle.pt','')
        #     FNAMES[i] = FNAMES[i].replace('_steer.pt','')
        # FNAMES = list(set(FNAMES))
        # FNAMES = FNAMES[::dn]

        FNAMES = []
        frict_vals = np.linspace(.5,2.5,10)
        # slope_vals = np.linspace(-.5,.5,10)
        slope_vals = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
        for frict in frict_vals:
            for slope in slope_vals:
                FNAMES.append('s' + str(slope) + 'f' + str(frict))
        FNAMES = FNAMES[::dn]
        # print(FNAMES)
        #might be useful later
        FNAME_MAP = {}
        for i,name in enumerate(FNAMES):
            if name == 'hide':
                continue
            FNAME_MAP[i] = name
            self.experts.append(TFexpert(fpath + '/' + name,dt=dt))
        # print(FNAME_MAP)
        self.idmap = FNAME_MAP
        print(len(self.experts))
        self.weights = [1]*len(self.idmap)

    #https://github.com/mohakbhardwaj/reinforcement-learning/blob/master/expert_advice.py
    def predict(self, obsState, ctrl):

        preds = np.empty((0,3))
        for i,expert in enumerate(self.experts):
            tfstate = expert.forward(obsState, ctrl)
            preds = np.vstack([preds, tfstate])

        if self.behavior == 'RWM':
            idx = self.RWM_decision()
        elif self.behavior == 'GWM':
            idx = self.GWM_decision()
        elif self.behavior == 'EXP3':
            idx = self.EXP3_decision()

        # print(self.idmap[idx])

        avgpred = (self.weights/np.sum(self.weights)) @ preds

        return idx,preds, avgpred

    def all_future(self, obsState, ctrl, nsteps = 5):
        ensemble_states = np.empty([0,nsteps,3])
        for i,expert in enumerate(self.experts):
            tfstates = expert.future(obsState, ctrl, nsteps = nsteps)
            # print(tfstates.shape)
            ensemble_states = np.vstack([ensemble_states, np.expand_dims(tfstates, axis=0)])
        # print(ensemble_states.shape)
        return ensemble_states


    def loss(self,idx,preds,gtstate, avgpred = None):
        # print(preds-gtstate)
        expert_losses = np.linalg.norm(preds-gtstate,axis=1)
        # print(expert_losses)
        learner_loss = expert_losses[idx]

        self.total_learner_loss += learner_loss
        if avgpred is not None:
            avg_loss = np.linalg.norm(avgpred - gtstate)
            self.total_avg_loss += avg_loss
            self.avg_loss_plt.append(avg_loss)

        if self.behavior == 'RWM':
            self.RWM_update_weights(expert_losses)
        elif self.behavior == 'GWM':
            self.GWM_update_weights(expert_losses[idx],idx)
        elif self.behavior == 'EXP3':
            self.EXP3_update_weights(expert_losses[idx],idx)

        self.learner_loss_plt.append(learner_loss)
        self.experts_loss_plt.append(expert_losses)

    #can maybe manage weights better? currently they always decrease
    def RWM_update_weights(self, expert_losses):
        # expert_losses = expert_losses/np.linalg.norm(expert_losses)
        # expert_losses = expert_losses/np.sum(expert_losses)
        new_weights = [0]*len(self.weights)

        # for i,weight in enumerate(self.weights):
        #     # decrease = 1 - (self.ita*expert_losses[i])
        #     decrease = 1 - (self.ita*expert_losses[i])
        #     new_weights[i] = weight*decrease
        for i,weight in enumerate(self.weights):
            # print(expert_losses[i])
            decrease = np.exp(-self.ita*expert_losses[i])
            new_weights[i] = weight*decrease
        # new_weights = new_weights/np.linalg.norm(new_weights)
        #in case negative
        #normalizing here decreases perforamnce here though?
        #new_weights += np.min(new_weights)
        new_weights = new_weights/np.sum(new_weights)
        # print(new_weights)
        #wmin = np.min(new_weights)
        #if wmin < .000000001:
        #    new_weights += wmin

        self.weights = new_weights

    def RWM_decision(self):
        # sum_weights = np.sum(self.weights + np.min(self.weights))
        # print(np.divide(self.weights + np.min(self.weights), sum_weights))
        # idx = np.argmax(np.random.multinomial(1, np.divide(self.weights + np.min(self.weights), sum_weights)))

        sum_weights = np.sum(self.weights + np.min(self.weights))
        # print(np.divide(self.weights + np.min(self.weights), sum_weights))
        # print((self.weights + np.min(self.weights))/sum_weights)
        # print(np.sum(np.divide(self.weights + np.min(self.weights), sum_weights)))

        #print('---------')
        #print(sum_weights)
        #print(self.weights + np.min(self.weights))
        #print(np.divide(self.weights + np.min(self.weights), sum_weights))
        idx = np.argmax(np.random.multinomial(1, np.divide(self.weights + np.min(self.weights), sum_weights)))

        return idx

    def GWM_update_weights(self, loss, idx):

        eta = np.sqrt(np.log(len(self.experts))/self.t)

        self.weights[idx] *= np.exp(-eta * loss)

    def GWM_decision(self):
        idx = np.random.choice(np.arange(len(self.weights)), p=self.weights/np.sum(self.weights))
        self.t += 1
        return idx

    def EXP3_update_weights(self, loss, idx):
        aprob = self.weights[idx]/np.sum(self.weights)
        loss /= aprob
        eta = np.sqrt(np.log(len(self.experts))/ (self.t * len(self.experts)))

        self.weights[idx] *= np.exp(-eta * loss)

    def EXP3_decision(self):
        idx = np.random.choice(np.arange(len(self.weights)), p=self.weights/np.sum(self.weights))
        self.t += 1
        return idx

class BehaviorTester():
    def __init__(self, behavior_list, ita, dt = .1, dn = 10):
        self.behavior_list = behavior_list
        self.ensembles = []

        for behavior in behavior_list:
            ensemble = EnsembleLearner(behavior = behavior, ita = ita)
            ensemble.buildTfEnsemble('../sysid/systematic_data/tfs', dt = dt, dn = dn)
            self.ensembles.append(ensemble)
    def get_preds(self, obsState, ctrl):
        pred_dict = {}
        for ensemble in self.ensembles:
            pred_dict[ensemble.behavior] = ensemble.predict(obsState, ctrl)
        return pred_dict

    def losses(self,pred_dict, gtstate):
        for ensemble in self.ensembles:
            pred_id,preds,avgpred = pred_dict[ensemble.behavior]
            ensemble.loss(pred_id, preds, gtstate)



def calculate_cumulative_rewards(total_prediction_rounds, learner_loss, experts_loss, avg_loss):
    experts_loss = np.asarray(experts_loss)
    #print experts_loss
    cumulative_learner_loss = []
    cumulative_experts_loss = []
    cum_avg_loss = []

    cl_experts = [0]*len(experts_loss[0])
    for i in range(len(learner_loss)):
        cumulative_learner_loss.append(sum(learner_loss[0:i+1]))
        cum_avg_loss.append(sum(avg_loss[0:i+1]))
        for k in range(len(experts_loss[0])):
            cl_experts[k] = sum(experts_loss[0:i+1,k])
        cumulative_experts_loss.append(list(cl_experts))
    return cumulative_learner_loss, cumulative_experts_loss, cum_avg_loss

def calculate_average_regret(total_prediction_rounds, learner_loss, experts_loss):
	#First calculate cumulative losses
	experts_loss = np.asarray(experts_loss)
	#print experts_loss
	cumulative_learner_loss = []
	cumulative_experts_loss = []
	cl_experts = [0]*len(experts_loss[0])
	for i in range(total_prediction_rounds):
		cumulative_learner_loss.append(sum(learner_loss[0:i+1]))
		for k in range(len(experts_loss[0])):
			cl_experts[k] = sum(experts_loss[0:i+1,k])
		cumulative_experts_loss.append(list(cl_experts))
	#Calculate the regrets
	episode_regrets = []
	#print cumulative_learner_loss

	for i in range(total_prediction_rounds):
		best_expert_loss = min(cumulative_experts_loss[i])
		#print best_expert_loss, cumulative_learner_loss[i]
		episode_regrets.append((cumulative_learner_loss[i] - best_expert_loss)/(i+1.))
	return episode_regrets

"""load environment"""
config_file = "../configurations/ensembleParams.yaml"
env = WheeledSimEnv(config_file, T=EXP_LENGTH, render=True)

def main():
    import matplotlib.pyplot as plt




    test = env.observation_space
    # print('Observation:', test)


    if not PLOTLOCK:
        fig, axs = plt.subplots(2, 4, figsize=(12, 12))
        plt.show(block=False)
        # fig2, axs2 = plt.subplots(2, 2, figsize=(12, 12))
        # plt.show(block=False)

    model = KBMKinematics(hyperparams = {'L':.9})
    throttle = torch.load('../sysid/tfs/f1p0_throttle.pt')
    steer = torch.load('../sysid/tfs/f1p0_steer.pt')


    # vel = torch.tensor(0)
    # d = torch.tensor(0)
    # nstate = torch.tensor([0.0,0.0,0.0])
    # tfstate = torch.tensor([0.0,0.0,0.0])
    # preds = np.zeros([len(ensemble.experts),3])
    #
    # # gtvals = np.array([])
    # # mvals = np.array([])
    # # mtfvals = np.array([])
    # velvals = np.array([])
    # pvelvals = np.array([])
    # svals = np.array([])
    # psvals = np.array([])
    #
    # gtvals = np.empty([0,3])
    # mvals = np.empty([0,3])
    # mtfvals = np.empty([0,3])
    # ensemble_states = np.empty([0,len(ensemble.experts),3])

    tfexpert = TFexpert('../sysid/systematic_data/tfs/s0f1.6111111111111112',dt=.1)
    ensemble = EnsembleLearner(behavior = BEHAVIOR, ita = ITA)
    ensemble.buildTfEnsemble('../sysid/systematic_data/tfs', dt = .1, dn = 10)
    btester = BehaviorTester(behavior_list = ['RWM','GWM','EXP3'], ita = ITA, dt = .1, dn = 10)
    blosses = np.empty([0,3])
    # tfexpert = ensemble.experts[0]

    # run simulation 5 times
    for _ in range(1):
        terminal = False
        t = 0
        vel = torch.tensor(0)
        d = torch.tensor(0)
        nstate = torch.tensor([0.0,0.0,0.0])
        tfstate = torch.tensor([0.0,0.0,0.0])
        preds = np.zeros([len(ensemble.experts),3])

        # gtvals = np.array([])
        # mvals = np.array([])
        # mtfvals = np.array([])
        velvals = np.array([])
        pvelvals = np.array([])
        svals = np.array([])
        psvals = np.array([])
        topexp = np.array([])

        gtvals = np.empty([0,3])
        mvals = np.empty([0,3])
        mtfvals = np.empty([0,3])
        ensemble_states = np.empty([0,len(ensemble.experts),3])
        now = time.perf_counter()
        while not terminal:
            t += 1
            if t % 50 == 1:
                a = env.action_space.sample()
            a = [.7, 0.3]
            # now = time.perf_counter()
            obs, reward, terminal, i = env.step(a)
            # print('STATE = {}, ACTION = {}, t = {}'.format(obs, a, t))

            # print((state[0],obs['state'][0]))


            # print(obs['state'])
            x = obs['state'][0]
            # print(x)
            y = obs['state'][1]
            # print(y)
            q = obs['state'][3:7]
            # print(q)
            # dt = time.perf_counter() - now
            # print(dt)
            # now = time.perf_counter()
            thetan = np.arctan2(2.0 * (q[2]*q[3] + q[0]*q[1]), -1.0 + 2.0*(q[3]*q[3] + q[0]*q[0]))
            theta = -1*pybullet.getJointState(env.robot.cliffordID, env.robot.jointNameToID['axle2frwheel'])[0]
            # print(theta)
            # print(theta)

            temp_obs = np.array((obs['state'][0],obs['state'][1],thetan))
            # print(temp_obs)
            gtvals = np.vstack([gtvals, temp_obs])
            mvals = np.vstack([mvals, nstate])
            mtfvals = np.vstack([mtfvals, tfstate])
            ensemble_states = np.vstack([ensemble_states, np.expand_dims(preds,axis=0)])

            velvals = np.hstack([velvals, np.linalg.norm(obs['state'][7:10])])
            pvelvals = np.hstack([pvelvals, vel])
            svals = np.hstack([svals, thetan])
            psvals = np.hstack([psvals, d])



            if not PLOTLOCK and (t % PLOTRATE == 0):
                for ax in axs.flatten():
                    ax.cla()
                # for ax in axs2.flatten():
                #     ax.cla()

            # axs[0,3].plot(svals,'-r', label = 'gt heading')
            # axs[0,3].plot(mtfvals[:,2],'b-', label = 'heading from kbm')
            # axs[0,3].legend()
            buf_size = 50
            if t != 1:
                ensemble.loss(pred_id, preds, temp_obs, avgpred = avgpred)
                btester.losses(pred_dict, temp_obs)
                tloss = np.zeros([1,3])


                #loss
                if t > buf_size:
                    cumulative_learner_loss, cumulative_experts_loss, cum_avg_loss = calculate_cumulative_rewards(buf_size, ensemble.learner_loss_plt[t-buf_size:], ensemble.experts_loss_plt[t-buf_size:], ensemble.avg_loss_plt[t-buf_size:])
                else:
                    cumulative_learner_loss, cumulative_experts_loss, cum_avg_loss = calculate_cumulative_rewards(t-1, ensemble.learner_loss_plt, ensemble.experts_loss_plt, ensemble.avg_loss_plt)
                # plot_cumulative_losses(t-1, cumulative_learner_loss, cumulative_experts_loss)
                experts_loss = np.asarray(cumulative_experts_loss)
                #print(np.argmin(experts_loss[-1,:]))
                topexp = np.hstack([topexp, np.argmin(experts_loss[-1,:])])
                #print(ensemble.weights)

                #behavior tester plotting
                if not PLOTLOCK and (t % PLOTRATE == 0):
                    if t > buf_size:
                        for i,tensemble in enumerate(btester.ensembles):
                            axs[1,3].plot(tensemble.learner_loss_plt[t-buf_size:], label = tensemble.behavior)
                        axs[1,3].plot(ensemble.avg_loss_plt[t-buf_size:], 'r--', linewidth=3, label="Weighted avg")
                        axs[1,3].legend()
                    else:
                        for i,tensemble in enumerate(btester.ensembles):
                            axs[1,3].plot(tensemble.learner_loss_plt, label = tensemble.behavior)
                        axs[1,3].plot(ensemble.avg_loss_plt, 'r--', linewidth=3, label="Weighted avg")
                        axs[1,3].legend()

                #other
                if not PLOTLOCK and (t % PLOTRATE == 0):
                    episode_regrets = calculate_average_regret(t-1, ensemble.learner_loss_plt, ensemble.experts_loss_plt)
                    axs[0,3].plot(topexp)
                    axs[0,2].bar(np.arange(len(ensemble.weights)), ensemble.weights)
                    axs[1,2].bar(np.arange(len(ensemble.weights)), ensemble.experts_loss_plt[-1])
                    tround = np.linspace(0, t-1, t)
                    for i in range(len(experts_loss[0])):
                        axs[0,0].plot(experts_loss[:, [i]])#, label = "Cumulative loss of expert {}".format(i+1))
                    axs[0,0].plot(cumulative_learner_loss, 'k--', linewidth=3, label="Cumulative loss of learner")
                    axs[0,0].plot(cum_avg_loss, 'r--', linewidth=3, label="Cumulative loss of weighted avg")
                    axs[0,0].legend()
                    # axs[0,0].draw()
                    #regret
                    axs[0,1].plot(episode_regrets, 'k')

            state = torch.tensor([x,y,theta])
            ctrl = torch.tensor(a)
            # nstate = model.forward_dynamics(nstate,ctrl,.1)
            vel = throttle.forward(torch.tensor(np.linalg.norm(obs['state'][7:10])),torch.tensor(a[0]))
            d = steer.forward(torch.tensor(theta),torch.tensor(a[1]))
            # vel = throttle.forward(vel,torch.tensor(a[0]))
            # d = steer.forward(d,torch.tensor(a[1]))
            # nstate = model.forward_dynamics(torch.tensor(temp_obs),torch.tensor([vel,d]),.1)

            future = ensemble.all_future(obs['state'], ctrl, nsteps = 10)

            nstate = tfexpert.forward(obs['state'],ctrl)
            pred_id,preds,avgpred = ensemble.predict(obs['state'],ctrl)
            pred_dict = btester.get_preds(obs['state'], ctrl)
            # print(pred_id)
            tfstate = preds[pred_id,:]
            # nstate = avgpred
            # print(preds.shape)
            # print(future[:,0,:] - preds)
            #
            # for i in range(future.shape[0]):
            #     axs[1,0].plot(future[i,:,0],future[i,:,1],'k.')
            #
            # for i in range(preds.shape[0]):
            #     axs[1,0].plot(preds[i,0],preds[i,1],'b.')



            if not PLOTLOCK and (t % PLOTRATE == 0):
                axs[0, 0].set_title('Expert Losses')
                axs[0, 1].set_title('Regret')
                axs[1, 0].set_title('Expert Predictions')
                axs[1, 1].set_title('Learner Predictions vs GT')
                axs[0,2].set_title('Weights')
                axs[1,2].set_title('Expert Loss')
                axs[0,3].set_title('Expert with lowest recent loss')
                axs[1,3].set_title('Performance of various policies')

                # axs[0, 0].scatter(obs['lidar'][:, 0], obs['lidar'][:, 1], s=1.,c=obs['lidar'][:,2],cmap=plt.get_cmap('viridis'))
                # axs[0, 1].imshow(obs['heightmap'][0,:,:])
                # axs[1, 0].imshow(obs['front_camera'][:, :, :3])
                # fc = obs['front_camera']

                if t > buf_size and BUFF:
                    xminax = min(np.min(gtvals[t-buf_size:,0]),np.min(mvals[t-buf_size:,0]), np.min(mtfvals[t-buf_size:,0]))-.2
                    xmaxax = max(np.max(gtvals[t-buf_size:,0]),np.max(mvals[t-buf_size:,0]), np.max(mtfvals[t-buf_size:,0]))+.2
                    yminax = min(np.min(gtvals[t-buf_size:,1]),np.min(mvals[t-buf_size:,1]), np.min(mtfvals[t-buf_size:,1]))-.2
                    ymaxax = max(np.max(gtvals[t-buf_size:,1]),np.max(mvals[t-buf_size:,1]), np.max(mtfvals[t-buf_size:,1]))+.2

                    for i in range(ensemble_states.shape[1]):
                        axs[1,0].plot(ensemble_states[t-buf_size:,i,0],ensemble_states[t-buf_size:,i,1])
                    axs[1,0].set_ylim(yminax,ymaxax)
                    axs[1,0].set_xlim(xminax,xmaxax)
                else:
                    # minax = min(np.min(gtvals),np.min(mvals), np.min(mtfvals))-.2
                    # maxax = max(np.max(gtvals),np.max(mvals), np.max(mtfvals))+.2
                    xminax = min(np.min(gtvals[:,0]),np.min(mvals[:,0]), np.min(mtfvals[:,0]))-.2
                    xmaxax = max(np.max(gtvals[:,0]),np.max(mvals[:,0]), np.max(mtfvals[:,0]))+.2
                    yminax = min(np.min(gtvals[:,1]),np.min(mvals[:,1]), np.min(mtfvals[:,1]))-.2
                    ymaxax = max(np.max(gtvals[:,1]),np.max(mvals[:,1]), np.max(mtfvals[:,1]))+.2
                    for i in range(ensemble_states.shape[1]):
                        axs[1,0].plot(ensemble_states[:,i,0],ensemble_states[:,i,1])
                    # axs[1,0].set_ylim(yminax,ymaxax)
                    # axs[1,0].set_xlim(xminax,xmaxax)

                # minax = min(np.min(gtvals),np.min(mvals), np.min(mtfvals))-.2
                # maxax = max(np.max(gtvals),np.max(mvals), np.max(mtfvals))+.2
                # for i in range(ensemble_states.shape[1]):
                #     axs[1,0].plot(ensemble_states[:,i,0],ensemble_states[:,i,1])
                # axs[1,0].set_ylim(minax,maxax)
                # axs[1,0].set_xlim(minax,maxax)

                # axs[1,0].plot(velvals,'-g')
                # axs[1,0].plot(pvelvals,'-r')
                # axs[1,0].plot(svals,'-k')
                # axs[1,0].plot(psvals,'-b')
                # axs[1, 0].imshow(np.transpose(fc[:, :, :],(1,2,0)))
                # for i, l in zip(range(4), ['fl', 'fr', 'bl', 'br']):
                #     axs[1, 1].plot(obs['shock_travel'][:, i], label='{}_travel'.format(l))
                # axs[1, 1].legend()
                axs[1,1].plot(gtvals[:,0],gtvals[:,1],'-g')
                axs[1,1].plot(mvals[:,0],mvals[:,1],'-r')
                axs[1,1].plot(mtfvals[:,0],mtfvals[:,1],'-b')
                axs[1,1].set_ylim(yminax,ymaxax)
                axs[1,1].set_xlim(xminax,xmaxax)

                for i in range(future.shape[0]):
                    axs[1,0].plot(future[i,:,0],future[i,:,1],'.')


                plt.pause(1e-2)
        ensemble.reset()
        env.reset()
        print('done')


if __name__ == '__main__':
    main()
