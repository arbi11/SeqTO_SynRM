import numpy as np
import random
#import pickle
import sys, os

import constants_SRM
from env_Worm_SRM_5X5 import WormSRMEnv, get_file_back_slash, get_file, save_file, create_txt_file, append_txt_file
    
def normalize(qstate):
    unique = np.unique(qstate)
    if unique.size == 1:
        return (np.ones([1, 4])*.25).ravel()
    else:
        arr2 = (qstate - qstate.min()) / (qstate.max() - qstate.min())
        arr2 = arr2/arr2.sum()
        return(arr2.ravel())
       
action_size            = 4
state_size             = constants_SRM.env_dim[0]*constants_SRM.env_dim[1]


total_episodes         = 100                                # Total episodes
learning_rate          = 0.8                                   # Learning rate
max_steps                 = constants_SRM.max_steps          # Max steps per episode
gamma                     = 0.95                             # Discounting rate

# Exploration parameters
epsilon                 = 1.0                 # Exploration rate
max_epsilon             = 1.0                  # Exploration probability at start
min_epsilon             = 0.1                # Minimum exploration probability 
decay_rate                = 0.05               # Exponential decay rate for exploration prob initial=0.005
play_time                 = 5

# List of rewards
rewards = []

qstates = np.empty((1, state_size))
qdict = dict()
qcount = dict()

q_file_path = sys.argv[0]
#q_file_path = os.path.realpath(__file__)  
env_file_path = os.path.abspath(sys.modules[WormSRMEnv.__module__].__file__)
constant_file_path = constants_SRM.__file__
#updater_file_path = os.path.abspath(sys.modules[GameRunner.__module__].__file__)

q_fileName = get_file_back_slash(q_file_path).group(1)
q_fileName = constants_SRM.MODEL_PATH
#q_fileName = get_file(q_file_path).group(1)
save_file(q_file_path, q_fileName)

env_fileName = get_file(env_file_path).group(1)
save_file(env_file_path, env_fileName)

constant_fileName = get_file(constant_file_path).group(1)
save_file(constant_file_path, constant_fileName)

#updater_fileName = get_file(updater_file_path).group(1)
#save_file(updater_file_path, updater_fileName)

txt_content = 'File Name:' + q_fileName + '\t Env Name:' + \
                env_fileName + '\t Constants:' + constant_fileName + '\n'
                
create_txt_file(txt_content)

env = WormSRMEnv(env_dim = constants_SRM.env_dim, startR=1 , startC= 1, max_steps = max_steps)
state = env.reset()

txt_writer = append_txt_file   

qstates[0] = state
qdict[str(0)] = np.zeros([4])
qcount[str(0)] = np.zeros([4])
#np.where((qstates==state[:,None]).all(-1))[1]

#dims = qstates.max(0)+1
#X1D = np.ravel_multi_index(qstates.T,dims)

#out = np.where(np.in1d(np.ravel_multi_index(qstates.T,dims),\
#                    np.ravel_multi_index(state.T,dims)))[0]

# 2 For life or until learning is stopped
for env.frame_idx in range(total_episodes):
    # Reset the environment
    state = env.reset()         
    step = 0
    done = False
    total_rewards = 0
    net_force = 0.0
    exploration = 0
    
    for step in range(max_steps):
        search_state = np.expand_dims(state, axis= 0)            
        a = np.where((qstates==search_state[:,None]).all(-1))[1]
        if len(a) == 1:
            index = a[0]
            s_index = str(index)
            
        elif len(a) == 0:
            qstates = np.append(qstates, search_state, axis=0)
            a = np.where((qstates==search_state[:,None]).all(-1))[1]
            s_index = str(a[0])
            qdict[s_index] = np.zeros([4])
            qcount[s_index] = np.zeros([4])
#            print('Adding new states')
                
        else:
            print('Error in the system')
            			            
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state) Greedy
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qdict[s_index][:])

        # Else doing a random choice --> exploration based on probablity of the q-values of the actions in a state
        else:
#            action = env.action_space.sample()
            action = np.random.choice(np.arange(0,4), p=normalize(qdict[s_index][:]))
#            print('action:', action)
            exploration += 1

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, _ = env.step(action)

        # Updating Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        search_new_state = np.expand_dims(new_state, axis= 0)            
        a_new = np.where((qstates == search_new_state[:,None]).all(-1))[1]
        if len(a_new) == 1:
            index_new = a_new[0]
            s_index_new = str(index_new)

        elif len(a_new) == 0:
            qstates = np.append(qstates, search_new_state, axis=0)
            a_new = np.where((qstates==search_new_state[:,None]).all(-1))[1]
            s_index_new = str(a_new[0])
            qdict[s_index_new] = np.zeros([4])
            qcount[s_index_new] = np.zeros([4])
#            print('Adding new states')    
            
        else:
            print('Error in the system')
        
        qdict[s_index][action] = qdict[s_index][action]*qcount[s_index][action] + learning_rate * (reward + gamma * np.max(qdict[s_index_new][:]) - qdict[s_index][action])       

#qtable[state, action] = qtable[state, action]*qcount[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        qcount[s_index][action] += 1
        qdict[s_index][action] = qdict[s_index][action]/qcount[s_index][action]
        
        total_rewards += reward
        
        # Our new state is state
        state = new_state
        env.render(train= True)

        # If done (iron count = 200 or steps = max_steps)
        if done == True:
            net_force = env.netT
            break
        
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*env.frame_idx)
    
    txt_content = ('Epoch {:d}, Reward {:.3f}, iron_c: {:d}, Exp {:d}, steps: {}, Net Torque: {:.3f}'
				            .format(env.frame_idx, total_rewards, env.count, exploration, env.step_count, net_force))
    txt_writer('\n' + txt_content)        
    print(txt_content)
   
    rewards.append(total_rewards)
    if env.frame_idx % play_time == 0:
        print('Playing the game after the {:2.0f}th epoch:--->'.format((env.frame_idx)))
        
#        env.save_checkpoints(qstates, '\\qstates')         
#        env.save_checkpoints(qdict, '\\qdict')    
#        env.save_checkpoints(qcount, '\\qcount')         
        done = False
         
        state = env.reset()
        episodic_reward = 0.0
        net_force = 0.0
        while not done:
            
            env.render(train= False)
            search_state = np.expand_dims(state, axis= 0)
            a = np.where((qstates==search_state[:,None]).all(-1))[1]
            if len(a) == 1:      # If the state has already been visited
                index = a[0]
                s_index = str(index)
                
            elif len(a) == 0:    # First visit to a state
                qstates = np.append(qstates, search_state, axis=0)
                a = np.where((qstates==search_state[:,None]).all(-1))[1]
                s_index = str(a[0])
                qdict[s_index] = np.zeros([4])
                qcount[s_index] = np.zeros([4])               
                
            else:     # probably a state has been stored twice
                print('Error in the system')                   
                   
            action = np.argmax(qdict[s_index][:])
#            action = np.argmax(actions, axis= 1)
            new_state, reward, done, _ = env.step(action)
            state = new_state
            episodic_reward += reward
            if done == True:
                net_force = env.netT
        
        test_txt_content = ('\t Episodic Reward: {:2.4f}, iron_c: {:d}, Net Torque: {:2.4f}'\
                     .format(episodic_reward, env.count, net_force))

        txt_writer('\n' + test_txt_content)                      
        print(test_txt_content)                                        

           