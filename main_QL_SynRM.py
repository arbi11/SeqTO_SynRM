import argparse
import os
import datetime
import json
import pandas as pd
from pathlib import Path
import numpy as np

from env_seqTO_SynRM import seqTO_SynRM 
from QL_SynRM_v2 import QL_Agent

def main():
    
    print(params['file_path'])
    
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    saves_main_folder = params['file_path'].parents[0] / 'Saves'
    saves_current_folder = params['file_path'].parents[0] / 'Saves' / current_time
    
    # print(saves_current_folder, '\n', saves_main_folder)
    
    env = seqTO_SynRM(main_folder_loc = params['file_path'].parents[0],
                      episode_length = int(params['design_domain_size']*params['episode_length_multiplier']),
                      dd_size = params['design_domain_size'],
                      params=params
                      )    
    
    ############
    # 0: Right #
    # 1: Left  #
    # 2: Up    #
    # 3: Down  #
    ############
    episode_length = int(params['design_domain_size']*params['episode_length_multiplier'])
    
    agent = QL_Agent(params = params, episode_length = episode_length)
    agent.train(env)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path',
                        default= Path(__file__).resolve())

    parser.add_argument('--design_domain_size', default= 9) # cosim dependent
    parser.add_argument('--action_dim', default=5) # Number of actions for each actor

    parser.add_argument('--buffer_episodes', default=3)  # cosim dependent
    parser.add_argument('--num_actions', default=4)
    parser.add_argument('--episode_length_multiplier', default=3)
    
    parser.add_argument('--verbose', default=False)    
    parser.add_argument('--num_episodes', default=2000) # Number of episodes to run

    parser.add_argument('--gamma', default= 0.99) # discount_factor
    parser.add_argument('--lr', default=0.01) # RL specific
    # parser.add_argument('--epsilon_decay', default=1e-3)
    parser.add_argument('--epsilon_decay', default=1e-4)

    parser.add_argument('--test_frequency', default=25)  # RL specific    

    parser.add_argument('--save_static_plots', default=True) # Save plots for the episode
    parser.add_argument('--eps_data_save_frequency', default=5) # save every 5 episodes
    parser.add_argument('--checkpoint_save_frequency', default=50) # Save NN weights every 5 eps

    parser.add_argument('--RL_session_no', default=None) # Only needed if resume training - iF none, most recent run is resumed

    params = vars(parser.parse_args())

    main()
