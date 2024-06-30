import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import random
from math import sqrt
from collections import deque
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from env_cfg import Config, TestDemand, Agent
from util import *
import torch
import os

EPSILON = 1e-1
gamma = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_init_len(init):
    """
    Calculate total number of elements in a 1D array or list of lists.
    :type init: iterable, list or (list of lists)
    :rtype: int
    """
    is_init_array = all([isinstance(x, (float, int, np.int64)) for x in init])
    if is_init_array:
        init_len = len(init)
    else:
        init_len = len(list(itertools.chain.from_iterable(init)))
    return init_len



class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, n_agents=4, n_turns_per_game=100,test_mode=False):
        super().__init__()
        c = Config()
        self.orders_dic = {
            'Retailers':0,
            'Wholesalers':0,
            'Distributors':0,
            'Manufacturers':0
            
        }
        
        
        config, unparsed = c.get_config()
        self.config = config
        self.test_mode = test_mode
        
        if self.test_mode:
            self.test_demand_pool = TestDemand()

        self.curGame = 1 # The number associated with the current game (counter of the game)
        self.curTime = 0
        self.m = 10             #window size
        self.totIterPlayed = 0  # total iterations of the game, played so far in this and previous games
        self.players = self.createAgent()  # create the agents
        self.T = 0
        self.demand = []
        self.orders = []
        self.shipments = []
        self.rewards = []
        self.cur_demand = 0

        self.ifOptimalSolExist = self.config.ifOptimalSolExist
        self.getOptimalSol()

        self.totRew = 0    # it is reward of all players obtained for the current player.
        self.totalReward = 0
        self.n_agents = n_agents

        self.n_turns = n_turns_per_game
        seed  = random.randint(0,1000000)
        self.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.totalTotal = 0

        # Agent 0 has 5 (-2, ..., 2) + AO
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5)]))

        ob_spaces = {}
        for i in range(self.m):
            ob_spaces[f'current_stock_minus{i}'] = spaces.Discrete(5)
            ob_spaces[f'current_stock_plus{i}'] = spaces.Discrete(5)
            ob_spaces[f'OO{i}'] = spaces.Discrete(5)
            ob_spaces[f'AS{i}'] = spaces.Discrete(5)
            ob_spaces[f'AO{i}'] = spaces.Discrete(5)

        # Define the observation space, x holds the size of each part of the state
        x = [750, 750, 170, 45, 45]
        oob = []
        for _ in range(self.m):
          for ii in range(len(x)):
            oob.append(x[ii])
        self.observation_space = gym.spaces.Tuple(tuple([spaces.MultiDiscrete(oob)] * 4))

        # print("Observation space:")
        # print(self.observation_space)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def createAgent(self):
      agentTypes = self.config.agentTypes
      return [Agent(i,self.config.ILInit[i], self.config.AOInit, self.config.ASInit[i],
                                self.config.c_h[i], self.config.c_p[i], self.config.eta[i],
                                agentTypes[i],self.config) for i in range(self.config.NoAgent)]

    def resetGame(self, demand, ):
        
        self.demand = demand
        self.playType='test'
        self.curTime = 0
        self.curGame += 1
        self.totIterPlayed += self.T
        self.T = self.planHorizon()         #now fixed
        self.totalReward = 0

        self.deques = []
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        # reset the required information of player for each episode
        for k in range(0,self.config.NoAgent):
            self.players[k].resetPlayer(self.T)

        # update OO when there are initial IL,AO,AS
        self.update_OO()

    def reset(self, focus_agent=None):
        if self.test_mode:
            demand = self.test_demand_pool.next()
            if not self.test_demand_pool:           #if run out of testing data
                self.test_demand_pool = TestDemand()
        else:
            demand = [random.randint(0,2) for _ in range(102)]

        self.resetGame(demand)
        observations = [None] * self.n_agents

        self.deques = []
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        # prepend current observation
        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[], [], [], []]
        for i in range(self.n_agents):
            spaces = {}
            for j in range(self.m):
                obs[i].append(self.deques[i]['current_stock_minus'][j])
                obs[i].append(self.deques[i]['current_stock_plus'][j])
                obs[i].append(self.deques[i]['OO'][j])
                obs[i].append(self.deques[i]['AS'][j])
                obs[i].append(self.deques[i]['AO'][j])
                # spaces[f'current_stock_minus{j}'] = self.deques[i]['current_stock_minus'][j]
                # spaces[f'current_stock_plus{j}'] = self.deques[i]['current_stock_plus'][j]
                # spaces[f'OO{j}'] = self.deques[i]['OO'][j]
                # spaces[f'AS{j}'] = self.deques[i]['AS'][j]
                # spaces[f'AO{j}'] = self.deques[i]['AO'][j]

            # observations[i] = spaces
        if focus_agent is None:
            obs_agent = [self.players[k].getCurState(self.curTime) for k in [0, 1, 2, 3]]
           
           
        else:
            obs_agent = self.players[focus_agent].getCurState(self.curTime)
        return obs_agent
        
        
    def step(self, action:list, focus_agent=None):
        if get_init_len(action) != self.n_agents:
            raise error.InvalidAction(f'Length of action array must be same as n_agents({self.n_agents})')
        if any(np.array(action) < 0):
            raise error.InvalidAction(f"You can't order negative amount. You agents actions are: {action}")

        self.handleAction(action)
        self.next()

        self.orders = action
         
        self.orders_dic['Retailers'] += self.orders[0]
        self.orders_dic['Wholesalers'] += self.orders[1]
        self.orders_dic['Distributors'] += self.orders[2]
        self.orders_dic['Manufacturers'] += self.orders[3]
        
        for i in range(self.n_agents):
            self.players[i].getReward()
        self.rewards = [1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]

        if self.curTime == self.T+1:
            self.done = [True] * 4
        else:
            self.done = [False] * 4

        # print(f"Step {self.curTime} done. state of Retailor is {self.players[0].getCurState(self.curTime)}")
        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[],[],[],[]]
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            spaces = {}
            for j in range(self.m):
              obs[i].append(self.deques[i]['current_stock_minus'][j])
              obs[i].append(self.deques[i]['current_stock_plus'][j])
              obs[i].append(self.deques[i]['OO'][j])
              obs[i].append(self.deques[i]['AS'][j])
              obs[i].append(self.deques[i]['AO'][j])

        obs_array = np.array([np.array(row) for row in obs])
        state = obs_array #observations #self._get_observations()
        if focus_agent is None:
            obs_agent = [self.players[k].getCurState(self.curTime) for k in [0, 1, 2, 3]]
    
            return obs_agent, self.rewards, self.done, {}

        else:
            obs_agent = self.players[focus_agent].getCurState(self.curTime)
            return obs_agent, self.rewards[focus_agent], self.done, {}

    def handleAction(self, action):
        # get random lead time
        leadTime = random.randint(self.config.leadRecOrderLow[0], self.config.leadRecOrderUp[0])
        self.cur_demand = self.demand[self.curTime]
        # set AO
        BS = False
        self.players[0].AO[self.curTime] += self.demand[self.curTime]       #orders from customer, add directly to the retailer arriving order
        for k in range(0, self.config.NoAgent):
            if k >= 0:  #recording action
                self.players[k].action = np.zeros(5)        #one-hot transformation
                self.players[k].action[action[k]] = 1
                BS = False
            else:
                raise NotImplementedError
                self.getAction(k)
                BS = True

            # updates OO and AO at time t+1
            self.players[k].OO += self.players[k].actionValue(self.curTime, self.playType, BS = BS)     #open order level update
            leadTime = random.randint(self.config.leadRecOrderLow[k], self.config.leadRecOrderUp[k])        #order
            if self.players[k].agentNum < self.config.NoAgent-1:
                if k>=0:
                    self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime,
                                                                                                   self.playType,
                                                                                                   BS=False)  # TODO(yan): k+1 arrived order contains my own order and the order i received from k-1
                else:
                    raise NotImplementedError
                    self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime,
                                                                                                   self.playType,
                                                                                                   BS=True)  # open order level update

    def next(self):
        # get a random leadtime for shipment
        leadTimeIn = random.randint(self.config.leadRecItemLow[self.config.NoAgent - 1],
                                    self.config.leadRecItemUp[self.config.NoAgent - 1])

        # handle the most upstream recieved shipment
        self.players[self.config.NoAgent-1].AS[self.curTime + leadTimeIn] += self.players[self.config.NoAgent-1].actionValue(self.curTime, self.playType, BS=True)
                                                                #the manufacture gets its ordered beer after leadtime

        self.shipments = []
        for k in range(self.config.NoAgent-1,-1,-1): # [3,2,1,0]

            # get current IL and Backorder
            current_IL = max(0, self.players[k].IL)
            current_backorder = max(0, -self.players[k].IL)

            # increase IL and decrease OO based on the action, for the next period
            self.players[k].recieveItems(self.curTime)

            # observe the reward
            possible_shipment = min(current_IL + self.players[k].AS[self.curTime],
                                    current_backorder + self.players[k].AO[self.curTime])       #if positive IL, ship all beer or all they needs, if backorders, ship all k-1 needs
            self.shipments.append(possible_shipment)

            # plan arrivals of the items to the downstream agent
            if self.players[k].agentNum > 0:
                leadTimeIn = random.randint(self.config.leadRecItemLow[k-1], self.config.leadRecItemUp[k-1])
                self.players[k-1].AS[self.curTime + leadTimeIn] += possible_shipment

            # update IL
            self.players[k].IL -= self.players[k].AO[self.curTime]

            # observe the reward
            self.players[k].getReward()
            rewards = [-1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]

            # update next observation
            self.players[k].nextObservation = self.players[k].getCurState(self.curTime + 1)

        if self.config.ifUseTotalReward:  # default is false
            # correction on cost at time T
            if self.curTime == self.T:
                self.getTotRew()

        self.curTime += 1

    def getAction(self, k):
        self.players[k].action = np.zeros(self.config.actionListLenOpt)

        if self.config.demandDistribution == 2:
            if self.curTime and self.config.use_initial_BS <= 4:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                        max(0, (self.players[k].int_bslBaseStock - (
                                                                    self.players[k].IL + self.players[k].OO -
                                                                    self.players[k].AO[self.curTime])))))] = 1
            else:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                        max(0, (self.players[k].bsBaseStock - (
                                                                    self.players[k].IL + self.players[k].OO -
                                                                    self.players[k].AO[self.curTime])))))] = 1
        else:
            self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                    max(0, (self.players[k].bsBaseStock - (
                                                                self.players[k].IL + self.players[k].OO -
                                                                self.players[k].AO[self.curTime])))))] = 1

    def getTotRew(self):
      totRew = 0
      for i in range(self.config.NoAgent):
        # sum all rewards for the agents and make correction
        totRew += self.players[i].cumReward

      for i in range(self.config.NoAgent):
        self.players[i].curReward += self.players[i].eta*(totRew - self.players[i].cumReward) #/(self.T)

    def planHorizon(self):
      # TLow: minimum number for the planning horizon # TUp: maximum number for the planning horizon
      #output: The planning horizon which is chosen randomly.
      return random.randint(self.n_turns, self.n_turns)# self.config.TLow,self.config.TUp)

    def update_OO(self):
        for k in range(0,self.config.NoAgent):
            if k < self.config.NoAgent - 1:
                self.players[k].OO = sum(self.players[k+1].AO) + sum(self.players[k].AS)
            else:
                self.players[k].OO = sum(self.players[k].AS)

    def getOptimalSol(self):
        # if self.config.NoAgent !=1:
        if self.config.NoAgent != 1:
            # check the Shang and Song (2003) condition.
            for k in range(self.config.NoAgent - 1):
                if not (self.players[k].c_h == self.players[k + 1].c_h and self.players[k + 1].c_p == 0):
                    self.ifOptimalSolExist = False

            # if the Shang and Song (2003) condition satisfied, it runs the algorithm
            if self.ifOptimalSolExist == True:
                calculations = np.zeros((7, self.config.NoAgent))
                for k in range(self.config.NoAgent):
                    # DL_high
                    calculations[0][k] = ((self.config.leadRecItemLow[k] + self.config.leadRecItemUp[k] + 2) / 2 \
                                          + (self.config.leadRecOrderLow[k] + self.config.leadRecOrderUp[k] + 2) / 2) * \
                                         (self.config.demandUp - self.config.demandLow - 1)
                    if k > 0:
                        calculations[0][k] += calculations[0][k - 1]
                    # probability_high
                    nominator_ch = 0
                    low_denominator_ch = 0
                    for j in range(k, self.config.NoAgent):
                        if j < self.config.NoAgent - 1:
                            nominator_ch += self.players[j + 1].c_h
                        low_denominator_ch += self.players[j].c_h
                    if k == 0:
                        high_denominator_ch = low_denominator_ch
                    calculations[2][k] = (self.players[0].c_p + nominator_ch) / (
                                self.players[0].c_p + low_denominator_ch + 0.0)
                    # probability_low
                    calculations[3][k] = (self.players[0].c_p + nominator_ch) / (
                                self.players[0].c_p + high_denominator_ch + 0.0)
                # S_high
                calculations[4] = np.round(np.multiply(calculations[0], calculations[2]))
                # S_low
                calculations[5] = np.round(np.multiply(calculations[0], calculations[3]))
                # S_avg
                calculations[6] = np.round(np.mean(calculations[4:6], axis=0))
                # S', set the base stock values into each agent.
                for k in range(self.config.NoAgent):
                    if k == 0:
                        self.players[k].bsBaseStock = calculations[6][k]
                    else:
                        self.players[k].bsBaseStock = calculations[6][k] - calculations[6][k - 1]
                        if self.players[k].bsBaseStock < 0:
                            self.players[k].bsBaseStock = 0
        elif self.config.NoAgent == 1:
            if self.config.demandDistribution == 0:
                self.players[0].bsBaseStock = np.ceil(
                    self.config.c_h[0] / (self.config.c_h[0] + self.config.c_p[0] + 0.0)) * ((
                                                                                                         self.config.demandUp - self.config.demandLow - 1) / 2) * self.config.leadRecItemUp
        elif 1 == 1:
            f = self.config.f
            f_init = self.config.f_init
            for k in range(self.config.NoAgent):
                self.players[k].bsBaseStock = f[k]
                self.players[k].int_bslBaseStock = f_init[k]

    def render(self, mode='human', details=False):
        # if mode != 'human':
        #     raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        # print("")
        if details:
            print('\n' + '=' * 20)
            print('Turn:     ', self.curTime)
        stocks = [p.IL for p in self.players]
        if details:
            print('Stocks:   ', ", ".join([str(x) for x in stocks]))
            print('Orders:   ', self.orders)
            
            print('Shipments:', self.shipments)
            print('Rewards:', self.rewards)
            print('Customer demand: ', self.cur_demand)
       
  
        
        AO = [p.AO[self.curTime] for p in self.players]
        AS = [p.AS[self.curTime] for p in self.players]

        if details:
            print('Arrived Order: ', AO)
            print('Arrived Shipment: ', AS)

        OO = [p.OO for p in self.players]
        
        if details:
            print('Working Order: ', OO)

        # print('Last incoming orders:  ', self.next_incoming_orders)
        # print('Cum holding cost:  ', self.cum_stockout_cost)
        # print('Cum stockout cost: ', self.cum_holding_cost)
        # print('Last holding cost: ', self.holding_cost)
        # print('Last stockout cost:', self.stockout_cost)
        return self.rewards


def run_DQN(focus_agent = 0, k_sup = 5, priority = False):
    env = BeerGame(test_mode=True)
    buf = Buffer()
    pt = 0
    Q_net = QMLP().to(device)
    Q_target = QMLP().to(device)
      
    losses = []
    for episode in tqdm(range(40), desc='Running'):
        obs = env.reset(focus_agent=focus_agent)
        done = False
        rewards =[]
        steps = 0
        while not done:
            actions = [int(min(4, max(0, env.players[i].bsBaseStock - env.players[i].IL))) for i in (1, 2, 3)]
            rnd_action = (best_action(Q_net, obs, device=device)[0], actions[0], actions[1], actions[2])
            # 运行一个step
            next_obs, reward, done_list, _ = env.step(rnd_action, focus_agent=focus_agent)
            
            # 记录data
            data_item = [obs, rnd_action[focus_agent], reward, next_obs]

            # add st1, at1, rt, st, pt to buffer
            buf.buffer.append(data_item)
            buf.pts.append(pt+EPSILON)
            pt = buf.get_max()[0]
            
            # 训练网络
            for k in range(k_sup):
                data, idx = buf.sample(weighted=priority)
                r2 = best_action(Q_target, data[3], device=device)[1]
                label = data[2] + gamma*r2    
                Q_net, loss = train(Q_net, data, label)
                losses.append(loss)
                if priority:
                    buf.pts[idx] = sqrt(loss)
                # update priority
            
            # 检验游戏结束状态
            done = all(done_list)
            
            # 更新目标网络
            if steps % 10 == 0:
                Q_target = Q_net
            
            # 更新reward函数
            rew = env.render(details=False)
            rewards.append(rew)
        
            # 更新obs
            obs = next_obs
            steps += 1
            
        # print(f"At episode {episode}, the testing rewards is {np.mean(rewards, axis=0)}")
    # record losses
    pr = 'prTpro' if priority else 'prF'
    np.save(f'Qnet64_{k_sup}_losses_{pr}.npy', np.array(losses))
    
    # save model
    torch.save(Q_net.state_dict(), f'models/Q_net64_{k_sup}_{pr}.pth')


def valid_model(model_name='models/Q_net64_5_prF.pth', focus_agent=0, competitors='DQN'):
    env = BeerGame(test_mode=True)
    if model_name == 'random' or model_name =='bs':
        QNet = None
    else:
        QNet = QMLP(hidden_size=64).to(device)
        
        QNet.load_state_dict(torch.load(model_name))
        QNet.eval()
    rewards_list = []
    focus_agent = 0
    
    for episode in tqdm(range(50)): 
        obs = env.reset(focus_agent=focus_agent)
        done = False
        rewards =[]
        steps = 0
        if episode < 40:
            continue
        while not done:    
            if competitors == 'DQN':
                Q2 = QMLP().to(device)
                Q2.load_state_dict(torch.load('models/Q_net64_5_prF_2.pth'))
                Q2.eval()
                Q3 = QMLP().to(device)
                Q3.load_state_dict(torch.load('models/Q_net64_5_prF_3.pth'))
                Q3.eval()
                Q4 = QMLP().to(device)
                Q4.load_state_dict(torch.load('models/Q_net64_5_prF_4.pth'))
                Q4.eval()
                actions = [best_action(Q2, obs, device=device)[0], \
                            best_action(Q3, obs, device=device)[0], \
                            best_action(Q4, obs, device=device)[0]]
            elif competitors == 'bs':
                actions = [int(min(4, max(0, env.players[i].bsBaseStock - env.players[i].IL))) for i in (1, 2, 3)]
            elif competitors == 'random':
                actions = [random.randint(0, 4) for _ in range(3)]
            # print(f"the actions are {action2, action3, action4}")
            if model_name == 'random':
                action1 = random.randint(0, 4)
            elif model_name == 'bs':
                action1 = int(min(4, max(0, env.players[0].bsBaseStock - env.players[0].IL)))
            else:
                action1 = best_action(QNet, obs, device=device)[0]
                
            
            rnd_action = (action1, actions[0], actions[1], actions[2])
            # 运行一个step
            next_obs, reward, done_list, _ = env.step(rnd_action, focus_agent=focus_agent)
            # 检验游戏结束状态
            done = all(done_list)
            # 更新reward函数
            rew = env.render(details=False)
            rewards.append(rew[0])
        
            # 更新obs
            obs = next_obs
            steps += 1
        # print(f"At episode {episode}, the testing rewards is {np.mean(rewards, axis=0)}")
        if episode>=40:
            rewards_list.append(np.mean(rewards, axis=0))
    print(f"model {model_name} tested")
    
    return rewards_list, env.orders_dic
        
        
def run_valid(competitors='bs'):
    result_dic = {}
    

    for k in tqdm([5, 10, 15], total=3, desc="Validating..."):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))
        
        result = valid_model(model_name=f'Q_net64_{k}_prF.pth', competitors=competitors)
        result_dic[f'Q_net64_{k}_prF'] = np.mean(result, axis=0)
        plt.plot(result, label=f'Q_net64_{k}_prF')
        
        result = valid_model(model_name=f'Q_net64_{k}_prT.pth', competitors=competitors)
        result_dic[f'Q_net64_{k}_prTpro'] = np.mean(result, axis=0)
        plt.plot(result, label=f'Q_net64_{k}_prT')
        
        result = valid_model(model_name='bs', focus_agent=0, competitors=competitors)
        result_dic['bs'] = np.mean(result, axis=0)
        plt.plot(result, label='bs')
        
        result = valid_model(model_name='random', focus_agent=0, competitors=competitors)
        result_dic['random'] = np.mean(result, axis=0)
        plt.plot(result, label='random')

            
        # Add title and labels
        ax.set_title('Test rewards on 10 games', fontsize=16)
        ax.set_xlabel('Game', fontsize=14)
        ax.set_ylabel('Rewards', fontsize=14)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(fontsize=12)

        # Add fancy ticks and tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        # Show the plot
        plt.tight_layout()
        

        plt.legend()
        plt.savefig(f'figures/valid_results{k}.png')
        plt.clf()
        # print(result_dic)


# # used for report writting
# def draw_losses_DQN4(path='figures/'):
#     loss_DQN4 = np.load('Qnet64_5_losses_prF_1.npy')
#     loss = np.load('Qnet64_5_losses_prF.npy')
    
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     ax.plot(loss_DQN4, label='3DQN-Loss', color='green', linestyle='-', linewidth=2, marker='o', markersize=4)
#     ax.plot(loss, label='3BS-Loss', color='red', linestyle='-', linewidth=2, marker='o', markersize=4)
    
#     # Add title and labels
#     ax.set_title('Training Loss Curve', fontsize=16)
#     ax.set_xlabel('Epoch', fontsize=14)
#     ax.set_ylabel('Loss', fontsize=14)
    
#     # Add grid
#     ax.grid(True, linestyle='--', alpha=0.7)
    
#     # Add legend
#     ax.legend(fontsize=12)
    
#     # Add fancy ticks and tick labels
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     ax.tick_params(axis='both', which='minor', labelsize=10)
    
#     # Show the plot
#     plt.tight_layout()
    
#     plt.legend()
#     plt.savefig(f'{path}losses_DQN4.png')

# def draw_losses(path = 'figures/'):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     result_dic = {}
#     for k in [5, 10, 15]:
#         lossesF = np.load(f'Qnet64_{k}_losses_prF.npy')
#         lossesTpro = np.load(f'Qnet64_{k}_losses_prTpro.npy')
        
#         # Create a figure and axis
#         fig, ax = plt.subplots(figsize=(10, 6))

#         ax.plot(lossesTpro[-3000:], label=f'k={k}, prTpro', color='red', linestyle='-', linewidth=2, marker='o', markersize=4)

#         ax.plot(lossesF[-3000:], label=f'k={k}, prF', color='green', linestyle='-', linewidth=2, marker='o', markersize=4)
    
#         # Add title and labels
#         ax.set_title('Training Loss Curve', fontsize=16)
#         ax.set_xlabel('Epoch', fontsize=14)
#         ax.set_ylabel('Loss', fontsize=14)

#         # Add grid
#         ax.grid(True, linestyle='--', alpha=0.7)

#         # Add legend
#         ax.legend(fontsize=12)

#         # Add fancy ticks and tick labels
#         ax.tick_params(axis='both', which='major', labelsize=12)
#         ax.tick_params(axis='both', which='minor', labelsize=10)

#         # Show the plot
#         plt.tight_layout()
        
#         plt.legend()
#         plt.savefig(f'{path}losses_{k}.png')
#         plt.clf()
#         result_dic['Qnet64_'+str(k)+'_prF'] = np.mean(lossesF)
#         result_dic['Qnet64_'+str(k)+'_prTpro'] = np.mean(lossesTpro)
#     print(result_dic)

# def run_DQN4(focus_agent = 0, k_sup = 5, priority = False):
#     env = BeerGame(test_mode=True)
#     buf1 = Buffer()
#     buf2 = Buffer()
#     buf3 = Buffer()
#     buf4 = Buffer()
    
#     pt1 = pt2 = pt3 = pt4 = 0
#     Q_net1 = QMLP().to(device)
#     Q_target1 = QMLP().to(device)
      
#     Q_net2 = QMLP().to(device)
#     Q_target2 = QMLP().to(device)
    
#     Q_net3 = QMLP().to(device)
#     Q_target3 = QMLP().to(device)
    
#     Q_net4 = QMLP().to(device)
#     Q_target4 = QMLP().to(device)
      
#     losses1 = []
#     losses2 = []
#     losses3 = []
#     losses4 = []
#     for episode in tqdm(range(40), desc='Running'):
#         obs = env.reset(focus_agent=None)
#         done = False
#         rewards =[]
#         steps = 0
#         while not done:
#             # print(f"the obs is {obs}")
#             action1 = best_action(Q_net1, obs[0], device=device)[0]
#             action2 = best_action(Q_net2, obs[1], device=device)[0]
#             action3 = best_action(Q_net3, obs[2], device=device)[0]
#             action4 = best_action(Q_net4, obs[3], device=device)[0]
            
#             rnd_action = (action1, action2, action3, action4)
#             # 运行一个step
#             next_obs, reward, done_list, _ = env.step(rnd_action, focus_agent=None)
            
#             # 记录data
#             data_item1 = [obs[0], rnd_action[0], reward[0], next_obs[0]]
#             data_item2 = [obs[1], rnd_action[1], reward[1], next_obs[1]]
#             data_item3 = [obs[2], rnd_action[2], reward[2], next_obs[2]]
#             data_item4 = [obs[3], rnd_action[3], reward[3], next_obs[3]]
            
#             buf1.buffer.append(data_item1)
#             buf1.pts.append(pt1+EPSILON)
#             pt1 = buf1.get_max()[0]
            
#             buf2.buffer.append(data_item2)
#             buf2.pts.append(pt2+EPSILON)
#             pt2 = buf2.get_max()[0]
            
#             buf3.buffer.append(data_item3)
#             buf3.pts.append(pt3+EPSILON)
#             pt3 = buf3.get_max()[0]
            
#             buf4.buffer.append(data_item4)
#             buf4.pts.append(pt4+EPSILON)
#             pt4 = buf4.get_max()[0]
        
 
#             # 训练网络
#             for k in range(k_sup):
#                 data1, idx1 = buf1.sample(weighted=priority)
#                 r21 = best_action(Q_target1, data1[3], device=device)[1]
#                 label1 = data1[2] + gamma*r21    
#                 Q_net1, loss1 = train(Q_net1, data1, label1)
#                 losses1.append(loss1)
         
#                 data2, idx2 = buf2.sample(weighted=priority)
#                 r22 = best_action(Q_target2, data2[3], device=device)[1] 
#                 label2 = data2[2] + gamma*r22
#                 Q_net2, loss2 = train(Q_net2, data2, label2)
#                 losses2.append(loss2)
                
#                 data3, idx3 = buf3.sample(weighted=priority)
#                 r23 = best_action(Q_target3, data3[3], device=device)[1]
#                 label3 = data3[2] + gamma*r23
#                 Q_net3, loss3 = train(Q_net3, data3, label3)
#                 losses3.append(loss3)
                
#                 data4, idx4 = buf4.sample(weighted=priority)
#                 r24 = best_action(Q_target4, data4[3], device=device)[1]
#                 label4 = data4[2] + gamma*r24
#                 Q_net4, loss4 = train(Q_net4, data4, label4)
#                 losses4.append(loss4)
                
#                 if priority:
#                     buf1.pts[idx1] = sqrt(loss1)
#                     buf2.pts[idx2] = sqrt(loss2)
#                     buf3.pts[idx3] = sqrt(loss3)
#                     buf4.pts[idx4] = sqrt(loss4)
                
                
            
#             # 检验游戏结束状态
#             done = all(done_list)
            
#             # 更新目标网络
#             if steps % 10 == 0:
#                 Q_target1 = Q_net1
#                 Q_target2 = Q_net2
#                 Q_target3 = Q_net3
#                 Q_target4 = Q_net4
            
#             # 更新reward函数
#             rew = env.render(details=False)
#             rewards.append(rew)
        
#             # 更新obs
#             obs = next_obs
#             steps += 1
            
#         # print(f"At episode {episode}, the testing rewards is {np.mean(rewards, axis=0)}")
#     # record losses
#     pr = 'prTpro' if priority else 'prF'
#     np.save(f'Qnet64_{k_sup}_losses_{pr}_1.npy', np.array(losses1))
#     np.save(f'Qnet64_{k_sup}_losses_{pr}_2.npy', np.array(losses2))
#     np.save(f'Qnet64_{k_sup}_losses_{pr}_3.npy', np.array(losses3))
#     np.save(f'Qnet64_{k_sup}_losses_{pr}_4.npy', np.array(losses4))
    
    
#     # save model
#     torch.save(Q_net1.state_dict(), f'Q_net64_{k_sup}_{pr}_1.pth')
#     torch.save(Q_net2.state_dict(), f'Q_net64_{k_sup}_{pr}_2.pth')
#     torch.save(Q_net3.state_dict(), f'Q_net64_{k_sup}_{pr}_3.pth')
#     torch.save(Q_net4.state_dict(), f'Q_net64_{k_sup}_{pr}_4.pth')
    
# def valid_DQN4(compete = 'DQN'):
#     result_dic = {}
#     k = 5
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(6, 6))
#     # set random seed
#     np.random.seed(0)
#     result = valid_model(model_name=f'Q_net64_{k}_prF_1.pth', competitors = compete)
#     result_dic[f'Q_net64_{k}_prF'] = np.mean(result, axis=0)
#     plt.plot(result, label=f'DQN4DQN_{compete}')
#     np.random.seed(2)
#     result = valid_model(model_name='bs', competitors=compete)
#     result_dic[f'Q_net64_{k}_prTpro'] = np.mean(result, axis=0)
#     plt.plot(result, label=f'bs_{compete}')
#     np.random.seed(4)
#     result = valid_model(model_name='random', competitors=compete)
#     result_dic['bs'] = np.mean(result, axis=0)
#     plt.plot(result, label=f'random_{compete}')
#     np.random.seed(6)
#     result = valid_model(model_name=f'Q_net64_{k}_prF.pth', competitors=compete)
#     result_dic['random'] = np.mean(result, axis=0)
#     plt.plot(result, label=f'DQN4bs_{compete}')

#     # Add title and labels
#     ax.set_title('Test rewards on 10 games', fontsize=16)
#     ax.set_xlabel('Game', fontsize=14)
#     ax.set_ylabel('Rewards', fontsize=14)

#     # Add grid
#     ax.grid(True, linestyle='--', alpha=0.7)

#     # Add legend
#     ax.legend(fontsize=12)

#     # Add fancy ticks and tick labels
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     ax.tick_params(axis='both', which='minor', labelsize=10)

#     # Show the plot
#     plt.tight_layout()
    
#     plt.legend()
#     plt.savefig(f'figures/DQN4_models{compete}.png')
#     plt.clf()
#     # print(result_dic)

# def valid_DQN42():
#     result_dic = {}

#     k = 5
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(6, 6))
    
#     result = valid_model(model_name=f'Q_net64_{k}_prF_1.pth', competitors = 'DQN')
#     result_dic[f'Q_net64_{k}_prF'] = np.mean(result, axis=0)
#     plt.plot(result, label=f'DQN4DQN_DQN')
    
#     result = valid_model(model_name=f'Q_net64_{k}_prF.pth', competitors='bs')
#     result_dic[f'Q_net64_{k}_prTpro'] = np.mean(result, axis=0)
#     plt.plot(result, label=f'DQN4bs_bs')
    
#     result = valid_model(model_name=f'Q_net64_{k}_prF_1.pth', competitors='random')
#     result_dic['bs'] = np.mean(result, axis=0)
#     plt.plot(result, label='DQN4DQN_random')
    
#     result = valid_model(model_name=f'Q_net64_{k}_prF.pth', competitors='random')
#     result_dic['bs'] = np.mean(result, axis=0)
#     plt.plot(result, label='DQN4bs_random')
    
    
#     # Add title and labels
#     ax.set_title('Test rewards on 10 games', fontsize=16)
#     ax.set_xlabel('Game', fontsize=14)
#     ax.set_ylabel('Rewards', fontsize=14)

#     # Add grid
#     ax.grid(True, linestyle='--', alpha=0.7)

#     # Add legend
#     ax.legend(fontsize=12)

#     # Add fancy ticks and tick labels
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     ax.tick_params(axis='both', which='minor', labelsize=10)

#     # Show the plot
#     plt.tight_layout()
    
#     plt.legend()
#     plt.savefig(f'figures/DQN4_competitors2.png')
#     plt.clf()
#     # print(result_dic)

# def draw_4_losses():
#     loss1 = np.load('Qnet64_5_losses_prF_1.npy')
#     loss2 = np.load('Qnet64_5_losses_prF_2.npy')
#     loss3 = np.load('Qnet64_5_losses_prF_3.npy')
#     loss4 = np.load('Qnet64_5_losses_prF_4.npy')
    
    
#     plt.plot(loss3[:1000], label='Distributor')
#     plt.plot(loss4[:1000], label='Manufacturer')

#     plt.legend()
#     plt.savefig('figures/4_losses_dis+manu.png')
#     plt.clf()
    
#     plt.plot(loss1[:1000], label='Retailer',color='green')
#     plt.plot(loss2[:1000], label='WholeSaler',color='red')

#     plt.legend()
#     plt.savefig('figures/4_losses_re+whole.png')
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--model', type=str, default='Q_net64_5_prF.pth', help='model name')
    parser.add_argument('--k', type=int, default=5, help='supervised learning steps')
    parser.add_argument('--priority', type=bool, default=False, help='if use priority')
    parser.add_argument('--competitors', type=str, default='DQN', help='competitors')
    
    
    args = parser.parse_args()
    
    model_folder = 'models/'
    if args.mode == 'train':
        run_DQN(focus_agent=0, k_sup=args.k, priority=args.priority)
        
    elif args.mode == 'test':
        print(f"validating model {args.model} with other player using {args.competitors}")
        model = model_folder + args.model
        _, orders_dic = valid_model(model, focus_agent=0, competitors=args.competitors)
        print(f"the average rewards of retailer on 10 valid tasks are {_}")
        # plot the rewards
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.plot(_, label='Retailer')
        plt.xlabel('Game id')
        plt.ylabel('Rewards')
        plt.legend()
        plt.show()