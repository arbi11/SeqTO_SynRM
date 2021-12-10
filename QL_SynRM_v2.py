"""


"""

import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import datetime
import socket
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
plt.ioff()

class QL_Agent():
    def __init__(self, params, episode_length):

        """
        
        """
        
        self.params = params
        self.num_episodes = self.params['num_episodes']      
        if episode_length == None:
            self.episode_length = int(self.params['design_domain_size']*params['episode_length_multiplier'])
        else:
            self.episode_length = episode_length

        # Exploration
        self.min_epsilon = 0.1
        self.max_epsilon = 0.99
        self.epsilon = self.max_epsilon
        self.epsilon_decay = self.params['epsilon_decay']
        self.test = False

        # Episode experience data
        self.buffer = []
        self.Q_Table = {}
        
        self.episode_buffer = {'observations': [],
                               'actions': [],
                               'rewards': [],
                               'dones': []
                               }
        
        self.buffer = {'observations': [],
                       'actions': [],
                       'TD_update': []
                       }

        self.buffer_length = int(self.params['buffer_episodes']*self.episode_length)
        
        self.eps_step = 0
        self.start = 0
        self.exploration_count = 0

        # Instantiate Resume Training if True
        
        data_directory = params['file_path'].parents[0] / 'Saves'
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        
        if not (os.path.exists(str(data_directory))):
            os.mkdir(data_directory)
            
        current_save_dir = data_directory / current_time
        if not (os.path.exists(str(current_save_dir))):
            os.mkdir(current_save_dir)
                    
        # train_logFile_location = self.params['saves_main_folder'] / 'session_files_log.csv'

        # self.info_filename = self.params['save_directory'] / 'info_RLv5.txt'

        # json.dump(str(train_logFile_location), fp=open(self.info_filename, 'w+'))
        # json.dump(str(train_logFile_location.resolve()), indent=4, fp=open(self.info_filename, 'a+'))

        # print(train_logFile_location.resolve())

        # if self.params['agent_RL_resume_training'] == True:
        #     if not os.path.isfile(str(train_logFile_location)):
        #         raise ('No session file')
        #     else:
        #         train_logFile_df = pd.read_csv(str(train_logFile_location))
        #         print(train_logFile_df.shape)
        #         print(train_logFile_df['Session_no'].max())

        #         train_logFile_df.set_index('Session_no', inplace=True)
        #         if 'Unnamed: 0' in train_logFile_df.columns:
        #             train_logFile_df = train_logFile_df.drop('Unnamed: 0', axis=1)

        #         if params['RL_session_no'] == None:
        #             Session_no = int(train_logFile_df.index.max())
        #             self.data_directory = train_logFile_df.loc[Session_no, 'Session_folder_location']
        #             self.resume_epoch = train_logFile_df.loc[Session_no, 'Resume_epoch']
        #         else:
        #             self.data_directory = train_logFile_df.loc[params['RL_session_no'], 'Session_folder_location']
        #             self.resume_epoch = train_logFile_df.loc[params['RL_session_no'], 'Resume_epoch']

        #         self.data_directory = self.params['saves_main_folder'].resolve() / self.data_directory
        #         print('self.data_directory', self.data_directory)
        #         resume_Q_table_file_loc = str(self.resume_epoch.item()) + '_Q_table'
        #         print('resume_Q_table_file_loc', resume_Q_table_file_loc)
        #         checkpoint_Q_csv = os.path.join(str(self.data_directory.resolve()), 'checkpoint',
        #                                         resume_Q_table_file_loc)
        #         print(checkpoint_Q_csv)

        #         infile = open(str(checkpoint_Q_csv), 'rb')
        #         self.Q_Table = pickle.load(infile)

        #         print('len(agent.Q_Table)', len(self.Q_Table))

        #         self.start = self.resume_epoch.item()

        # json.dump({'In_fresh_start': self.start}, indent=4, fp=open(self.info_filename, 'a+'))

        

        log_dir = os.path.join(
            current_save_dir,
            'runs',
            current_time + '_' + socket.gethostname()
        )

        # self.checkpoint_directory = os.path.join(current_save_dir,
        #                                          'checkpoint')
        # if not (os.path.exists(self.checkpoint_directory)):
        #     os.mkdir(self.checkpoint_directory)

        self.writer = tf.summary.create_file_writer(log_dir)

    def normalize(self, qstate):
        unique_ele = len(set(qstate))

        if unique_ele == 1:
            return ([1/self.params['num_actions'] for i in range(self.params['num_actions'])])
        else:
            arr2 = [(qs - min(qstate)) / (max(qstate) - min(qstate)) for qs in qstate]
            arr3 = [qs/sum(arr2) for qs in arr2]
            return(arr3)
       
    def compute_action(self, raw_state):
        """
        :param state: pandas.DataFrame
            


        :return: action
            
        """
        state = np.array2string(raw_state.flatten(), separator='')

        try:
            max_value = max(self.Q_Table[state])
            act = self.Q_Table[state].index(max_value)

        except:
            act = np.random.choice(self.params['num_actions'])
            self.Q_Table[state] = [0, 0, 0, 0]

            if self.test == False:
                p = np.random.uniform(0, 1)
                if p < self.epsilon:
                    act = np.arange(0,self.params['num_actions'])
                    p = self.normalize(self.Q_Table[state])
                    self.exploration_count += 1
                    try:
                        act = np.random.choice(np.arange(0,self.params['num_actions']), p=p)
                    except:
                        print(p)
                        if state in self.Q_Table:
                            print('Key exists')
                            print(self.Q_Table[state])
                        raise('p has an error')
                else:
                    act = act

        return act

    def train(self, env):
        """
        Performs following functions:
            - Interacts Building env with the agent

            - Memory buffer to store transactional experience.
                storage helpers for a single batch of data

            - Performs agent training based on PPO algorithm
                training loop: collect samples, send to optimizer, repeat updates times

        :param env: Instance of seqTO_SynRM

        :return: Nothing
        """
        
        for eps_id in range(self.start, self.params['num_episodes']):
            self.episode_buffer = {'observations': [],
                                   'actions': [],
                                   'rewards': [],
                                   'dones': []
                                   }   
            self.exploration_count = 0
            
            state = env.reset()
            print(np.array2string(state.flatten(), separator=''))
            print(type(np.array2string(state.flatten(), separator='')))
            for frame_id in range(self.episode_length):
                self.episode_buffer['observations'].append(np.array2string(state.flatten(), separator=''))
                action = self.compute_action(state)
                next_state, R, done = env.step(action)
                
                self.episode_buffer['actions'].append(action)
                self.episode_buffer['rewards'].append(R)
                self.episode_buffer['dones'].append(done)
                
                if done == True:
                    break
                else:
                    state = next_state
                                        
            print(f'Eps: {eps_id}, Avg Torque: {env.history["past_avg_torques"][-1]}, Void count: {env.void_count},\
                  Exploration: {self.exploration_count}, Reward: {sum(self.episode_buffer["rewards"])}')
        
            self.buffer['observations'] += self.episode_buffer['observations'][:-1]
            self.buffer['actions'] += self.episode_buffer['actions'][:-1]
            for td_inx in range(len(self.episode_buffer['observations'])-1):
                self.buffer['TD_update'].append(self.episode_buffer['rewards'][td_inx] + \
                                                    (self.params['gamma'] * \
                                                     max(self.Q_Table[self.episode_buffer['observations'][td_inx+1]])))
                
            
            if len(self.buffer['TD_update']) >= self.buffer_length:
                assert (len(self.buffer['observations']) >= self.buffer_length)
                assert (len(self.buffer['actions']) >= self.buffer_length)

                self.buffer_full = True
                self.buffer['observations'] = self.buffer['observations'][-self.buffer_length:]
                self.buffer['actions'] = self.buffer['actions'][-self.buffer_length:]
                self.buffer['TD_update'] = self.buffer['TD_update'][-self.buffer_length:]
                
                for i in range(10):
                    indexes = np.random.choice(self.buffer_length, size=int(self.buffer_length/2))
    
                    for inx in indexes:
                        try:
                            obs = self.buffer['observations'][inx]
                            act = self.buffer['actions'][inx]
                        except:                        
                            print(len(self.buffer['actions']), len(self.buffer['observations']), self.buffer_length)
                            print('Error at: ', inx, end='\t')
                            raise('Error in updating')
    
                        td_update = self.buffer['TD_update'][inx]
                        self.Q_Table[obs][act] += self.params['lr']* (td_update - self.Q_Table[obs][act])
            
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * eps_id)

            with self.writer.as_default():
                tf.summary.scalar("Train/Avg_torque",
                                  env.history["past_avg_torques"][-1], step=eps_id)                
                tf.summary.scalar("Train/Sum_of_reward", sum(self.episode_buffer['rewards']), step=eps_id)                
                tf.summary.scalar("Exploration/Count", self.exploration_count, step=eps_id)
                tf.summary.scalar("Exploration/Epsilon", self.epsilon, step=eps_id)

            tf.summary.flush(self.writer)

            if eps_id % self.params['test_frequency'] == 0:
                
                self.test = True
                self.episode_buffer = {'observations': [],
                                       'actions': [],
                                       'rewards': [],
                                       'dones': []
                                       }    
                
                state = env.reset()
                for frame_id in range(self.episode_length):
                    self.episode_buffer['observations'].append(state.flatten('F'))
                    action = self.compute_action(state)
                    next_state, R, done = env.step(action)
                    
                    self.episode_buffer['actions'].append(action)
                    self.episode_buffer['rewards'].append(R)
                    self.episode_buffer['dones'].append(done)
                    
                    if done == True:
                        break
                    else:
                        state = next_state
            
                with self.writer.as_default():
                    tf.summary.scalar("Test/Avg_torque", env.history["past_avg_torques"][-1], step=eps_id)                    
                    tf.summary.scalar("Test/Sum_of_reward", sum(self.episode_buffer['rewards']), step=eps_id)     

                self.test = False

            # save the dictionary, episodes, buffer, lr, 
            # if self.frame_id % self.params['checkpoint_save_frequency'] == 0:

            #     outfile = open(os.path.join(self.checkpoint_directory, str(self.frame_id) + '_Q_table'), 'wb')
            #     pickle.dump(self.Q_Table, outfile)
            #     outfile.close()

            #     run_params = {'lr': self.params['lr'],
            #                   'gamma': self.params['gamma'],
            #                   'epsilon': self.epsilon
            #                   }

            #     outfile = open(os.path.join(self.checkpoint_directory, str(self.frame_id) + '_run_params'), 'wb')
            #     pickle.dump(run_params, outfile)
            #     outfile.close()