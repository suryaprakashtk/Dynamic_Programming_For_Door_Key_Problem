from utils import *
import numpy as np

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door
inf = 1000000

class MDP_a(object):
    def __init__(self, env, info):
        """Initialize a the MDP Problem.

        Parameters:
            env: Env parameters
        """
        self.env = env  # Initialize in Object of States
        self.key_pos = info["key_pos"]
        self.door_pos = info["door_pos"]
        self.goal_pos = info["goal_pos"]
        self.no_of_states = 5
        self.no_of_inputs = 5
        self.no_of_directions = 4
        self.time_horizon = self.get_total_time_horizon()
        self.no_of_all_states = self.get_total_time_horizon()

        dir_code = self.encode_dir(env.dir_vec)
        door = env.grid.get(info["door_pos"][0], info["door_pos"][1])
        if(door.is_open):
            self.prior = np.array([env.agent_pos[0],env.agent_pos[1],dir_code,0,1])
        else:
            self.prior = np.array([env.agent_pos[0],env.agent_pos[1],dir_code,0,0])
    
    
    def init_fuction(self):
        self.all_states = self.create_all_states_matrix()

        self.q_function = np.zeros(shape=(self.no_of_all_states,self.no_of_inputs),dtype=int)
        self.q_function[:,:] = inf
        self.value_function = np.zeros(shape=(self.no_of_all_states,self.time_horizon),dtype=int)
        self.value_function[:,:] = inf
        self.policy = np.zeros(shape=(self.no_of_all_states,self.time_horizon),dtype=int)
        self.policy[:,:] = inf
        return
    
    def dp(self):
        Ind_Valid_T = np.logical_and(self.all_states[:,0]==self.goal_pos[0],self.all_states[:,1]==self.goal_pos[1])
        self.value_function[Ind_Valid_T,-1] = 0
        # plot_env(self.env)
        for i in range(self.time_horizon-2, -1, -1):
            for j in range(self.all_states.shape[0]):
                for k in range(self.no_of_inputs):
                    # if(1<=self.all_states[j][0]<2 and 2<=self.all_states[j][1]<3 and 0<=self.all_states[j][2]<1):
                    next_state,flag = self.motion_model(self.all_states[j],k)
                    if(flag):
                        match_state = np.all(self.all_states == next_state, axis=1)
                        self.q_function[j,k] = 1 + self.value_function[match_state,i+1]
                    else:
                        self.q_function[j,k] = inf
                        
            self.value_function[:,i] = np.amin(self.q_function, axis=1)
            self.policy[:,i] = np.argmin(self.q_function, axis=1)
            init_state = np.all(self.all_states == self.prior, axis=1)
            if(self.value_function[init_state,i]<=inf/2):
                return i,self.value_function[init_state,i]
    
    def get_sequence(self,time:int):
        seq_list = []
        start_state = self.prior.copy()
        for i in range(time,self.time_horizon-1):
            state_state_mask = np.all(self.all_states == start_state, axis=1)
            seq_list.append(self.policy[state_state_mask,i][0])
            start_state_next = self.motion_model(start_state,self.policy[state_state_mask,i])
            start_state = start_state_next[0]
        return seq_list

    def motion_model(self,state:np.ndarray,action:int):
        if(self.check_valid_state(state)):
            next = state.copy()
            agent_dir = self.get_dir(state[2])
            if(action == MF):
                next[0:2] = state[0:2] + agent_dir
                if(0<=next[0]<self.env.width and 0<=next[1]<self.env.height):
                    if(self.check_valid_state(next)):
                        return next,True
                    else:
                        return state,False
                else:
                    return state,False
            elif(action == TL):
                if(state[2] == 0):
                    next[2] = 1
                elif(state[2] == 1):
                    next[2] = 2
                elif(state[2] == 2):
                    next[2] = 3
                elif(state[2] == 3):
                    next[2] = 0
                return next,True
            elif(action==TR):
                if(state[2] == 0):
                    next[2] = 3
                elif(state[2] == 1):
                    next[2] = 0
                elif(state[2] == 2):
                    next[2] = 1
                elif(state[2] == 3):
                    next[2] = 2
                return next,True
            elif(action == PK):
                next[0:2] = state[0:2] + agent_dir
                if(next[0]==self.key_pos[0] and next[1]==self.key_pos[1]):
                    # Already picked keys
                    if(state[3] == 1):
                        return state,False
                    next = state.copy()
                    next[3] = 1
                    return next,True
                else:
                    return state,False
            elif(action == UD):
                next[0:2] = state[0:2] + agent_dir
                if(next[0]==self.door_pos[0] and next[1]==self.door_pos[1]):
                    # Door already opened
                    if(state[4]==1):
                        return state,False
                    # if I dont have the key
                    if(state[3]==0):
                        return state,False
                    next = state.copy()
                    next[4] = 1
                    return next,True
                else:
                    return state,False
        else:
            return state,False

    def check_valid_state(self,state:np.ndarray):
        current = self.env.grid.get(state[0], state[1])
        if(current==None):
            return True
        if(current.type == 'key'):
            return True
        if(current.type == 'goal'):
            return True
        if(current.type == 'wall'):
            return False
        # If I am in door location and door is open. No problem
        if(current.type == 'door' and state[4]==1):
            return True
        elif(current.type == 'door'):
            return False
        return False

    
    def get_dir(self,direction_code:int):
        result = np.zeros(shape=(2),dtype=int)
        # Right
        if (direction_code == 0):
            result[0] = 1
            result[1] = 0
        # UP
        elif (direction_code == 1):
            result[0] = 0
            result[1] = -1
        # Left
        elif (direction_code == 2):
            result[0] = -1
            result[1] = 0
        # Down
        elif (direction_code == 3):
            result[0] = 0
            result[1] = 1
        return result
    
    def encode_dir(self,direction_vec:np.ndarray):
        # Right
        if (direction_vec[0]==1 and direction_vec[1]==0):
            result = 0
        # UP
        elif (direction_vec[0]==0 and direction_vec[1]==-1):
            result = 1
        # Left
        elif (direction_vec[0]==-1 and direction_vec[1]==0):
            result = 2
        # Down
        elif (direction_vec[0]==0 and direction_vec[1]==1):
            result = 3
        return result


    def get_total_time_horizon(self):
        result = self.env.height*self.env.width*self.no_of_directions*2*2
        return result
    
    def create_all_states_matrix(self):
        result = np.zeros(shape=(self.no_of_all_states,self.no_of_states),dtype=int)
        temp = 0
        for i in range(self.env.height):
            for j in range(self.env.width):
                for k in range(self.no_of_directions):
                    result[temp,0] = i
                    result[temp,1] = j
                    result[temp,2] = k
                    temp = temp+1

        result[temp:2*temp,:] = result[0:temp,:]
        result[temp:2*temp,3] = 1
        result[2*temp:4*temp,:] = result[0:2*temp,:]
        result[2*temp:4*temp,4] = 1
        return result
    
