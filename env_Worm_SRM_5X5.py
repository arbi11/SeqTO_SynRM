import numpy as np
#import femm
import imageio
import warnings
from  skimage import img_as_float, img_as_uint
import matplotlib.pyplot as plt
import re
import os
import constants_SRM as con
from shutil import copyfile
import matlab.engine

FEMM_PATH = con.FEMM_PATH + '\\'
MODEL_PATH = con.MODEL_PATH + '\\'
TRAIN_PATH = con.TRAIN_PATH + '\\'

class WormSRMEnv:

    def __init__(self, env_dim = [6, 6], startR=0 , startC= 3, max_steps= con.max_steps):
        
        self.env_dim = env_dim
        self.past_rewards = []
        self.past_states = []
        self.posR = startR
        self.posC = startC
        self.done = 0.0           
        self.issue = 0
        self.max_steps = max_steps
        self.R = 0.0
        self.count = 0
        self.worm_step_size = 1
        self.step_count = 0
        self.netT = 0.0
        self.frame_idx = -1
        self.eng = matlab.engine.start_matlab()
                
        self.eng.addpath(FEMM_PATH);
        
    def reset(self):
        
        self.past_rewards = []
        self.past_states = []
        self.netT = 0.0
        self.step_count = 0
        self.posR = 1
        self.posC = 4
        self.issue = 0
        self.done = 0
        self.frame_idx += 1

        ob = np.ones(self.env_dim)
        ob[self.posR -1, self.posC-1] = 0
        ob_mat = matlab.double(ob.tolist())

        T= self.eng.setGeo_py_dual_PS(ob_mat)
        
        # The actual environemnt is 5X5
        # But a padding of 1 is provided around the env for movement of controller
        ob = np.ones([x+2 for x in self.env_dim])*3
        # Coz of the padding the start posn is also padded with 1
        ob[1:1+self.env_dim[0], 1:1+self.env_dim[1]] = 1
        
        ob[self.posR, self.posC] = 0
        self.past_states.append(np.copy(ob))
        
#        self.issue = error
        self.past_rewards.append(T*100)                          
#        if self.issue == 0:
            
        return np.copy(ob[1:1+self.env_dim[0], 1:1+self.env_dim[1]].flatten())
    
    def reward(self, state):
        
        # Selecting only the active env space and removing the padding
        state = np.copy(state[1:1+self.env_dim[0], 1:1+self.env_dim[1]])
        state[state==5] = 0

#        print(state.shape)
        old_state = np.copy(np.squeeze([self.past_states[-1][1:1+self.env_dim[0], 1:1+self.env_dim[1]]], axis=0))
        old_state[old_state == 5] = 0
#        print(old_state.shape)

        con_state = np.dstack((state, old_state))
#        print('Going for reward with', state)
        
        state_mat = matlab.double(con_state.tolist())
        T = self.eng.setGeo_py_dual_PS(state_mat)      
        self.netT = T  
        return T*100
    
    def clean_up(self, mat_state):
        mat_state[mat_state == 0] = 5
        mat_state[0, :] = 3
        mat_state[-1, :] = 3
        mat_state[:, 0] = 3
        mat_state[:, -1] = 3    
        mat_state[self.posR, self.posC] = 0              
        return mat_state 

    def move(self, mat_state, w):
        mat_state[mat_state == 0] = 5   
        mat_state[self.posR, self.posC] = 0
        return(mat_state)    
    
    def step(self, action):
        state = np.copy(self.past_states[-1])
        self.step_count +=1
        self.R = 0.0
        self.count = 0
        
        #### ACTIONS ####
        # 0     : Right #     
        # 1     : Left  #
        # 2     : Up    #
        # 3     : Down  #
        #################

        if (action) == 0:
#            print('before', self.posR, self.posC)
            self.posC += self.worm_step_size
#            print('after', self.posR, self.posC)
            if state[self.posR, self.posC] == 1:    
                state2 = self.move(state, self.worm_step_size)
                self.issue = 0
            elif state[self.posR, self.posC] == 5:
                self.issue = 1
                state2 = self.move(state, self.worm_step_size)
            else:
                self.issue = 2
                self.posC -= self.worm_step_size
                        
        elif (action) == 1:
            self.posC -= self.worm_step_size
            
            if state[self.posR, self.posC] == 1:
                state2 = self.move(state, self.worm_step_size)
                self.issue = 0
            elif state[self.posR, self.posC] == 5:
                self.issue = 1
                state2 = self.move(state, self.worm_step_size)
            else:
                self.posC += self.worm_step_size
                self.issue = 2
        
        elif (action) == 2:
            self.posR -= self.worm_step_size
            
            if state[self.posR, self.posC] == 1:    
                state2 = self.move(state, self.worm_step_size)
                self.issue = 0
            elif state[self.posR, self.posC] == 5:
                self.issue = 1
                state2 = self.move(state, self.worm_step_size)
            else:
                self.posR += self.worm_step_size
                self.issue = 2
        
        elif (action) == 3:
#            print('before', self.posR, self.posC)               
            self.posR += self.worm_step_size
#            print('after', self.posR, self.posC)                        
            
            if state[self.posR, self.posC] == 1:    
                state2 = self.move(state, self.worm_step_size)
                self.issue = 0
            elif state[self.posR, self.posC] == 5:
                state2 = self.move(state, self.worm_step_size)
                self.issue = 1
            else:
                self.posR -= self.worm_step_size
                self.issue = 2
                
        if self.issue == 0:    
            #print('Going for reward with a shape of:', state2.shape)
            new_state = self.clean_up(state2)
            F = self.reward(state2)
            self.R = (F - self.past_rewards[-1])
#            R = F
            self.past_rewards.append(F)
           
        elif self.issue == 1:
            new_state = self.clean_up(state2)
            self.R = con.penalty+1          

        elif self.issue == 2:            
            new_state = self.clean_up(state)
            self.R = con.penalty            
    
        unique, counts = np.unique(new_state, return_counts=True)
        index, = np.where(unique == 5)
        try:
            self.count = counts[index[0]] + 1
        except:
            self.count = 0        
        
        if self.count > con.max_iron or self.step_count == self.max_steps:
            self.done = 1.0
        
        if new_state[self.posR, self.posC] != 0:
            raise('Problem in code. Worm outside the geometry')
        
        self.past_states.append(np.copy(new_state))  
        
        return(np.copy(new_state[1:1+self.env_dim[0], 1:1+self.env_dim[1]].flatten()), self.R, self.done, {})

#np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#mat_a = matlab.double(np_a.tolist())
#
#eng.sum(mat_a == 1.0)
#
#eng.size(mat_a)

    def render(self, train= False):

        if train == True:    
            eps_dir = FEMM_PATH + TRAIN_PATH + 'Eps_' + str(self.frame_idx) + '\\'
        else:
            eps_dir = FEMM_PATH + MODEL_PATH + 'Eps_' + str(self.frame_idx) + '\\'   
        
        if not (os.path.exists(FEMM_PATH + MODEL_PATH)):
            os.mkdir(FEMM_PATH + MODEL_PATH)

        if not (os.path.exists(eps_dir)):
            os.mkdir(eps_dir)

        toP = np.copy(self.past_states[-1])
        toP = toP[1:1+self.env_dim[0], 1:1+self.env_dim[1]]
        
#        toP[toP == 1] = 5
        toP = toP/np.max(toP)
        imageio.imwrite(eps_dir + str(self.frame_idx) + '_Step-' + str(self.step_count) + '_Issue-' + str(self.issue) + '_reward-' + str(self.R) + '.png', img_as_uint(toP))
        

def save_file(src, fileName):
    dst = FEMM_PATH + MODEL_PATH + fileName + '.py'    
    copyfile(src, dst)

def get_file(path_file):
    path_components = path_file.split('\\')
    for component in path_components:
        file_sre = re.search('(.*).py', component)
#        file_name = file_sre.group(1)
    return file_sre     

def get_file_back_slash(path_file):
    path_components = path_file.split('/')
    for component in path_components:
        file_sre = re.search('(.*).py', component)
#        file_name = file_sre.group(1)
    return file_sre     

def create_txt_file(txt):
    with open(FEMM_PATH + MODEL_PATH + "history.txt", "w") as myfile:
        myfile.write(txt)
        
def append_txt_file(txt):
    with open(FEMM_PATH + MODEL_PATH + "history.txt", "a") as myfile:
            myfile.write(txt)
        
