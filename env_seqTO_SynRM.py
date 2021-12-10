# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:15:00 2021

@author: akhan147
"""

import numpy as np
import pandas as pd
import win32com.client as win32

class seqTO_SynRM():
    
    def __init__(self, main_folder_loc = None, episode_length = None, dd_size = None, params = None):
        
        if params == None:
            raise('params missing')
        else:
            self.params = params
            
        if main_folder_loc == None:
            raise('Folder location missing')
        else:
            self.main_folder_loc = main_folder_loc
        
        if dd_size == None:
            raise('Design Domain dimensions missing')
        else:
            self.dd_size = dd_size
            
        if episode_length == None:
            self.episode_length = int(dd_size*self.params['episode_length_multiplier'])
        else:
            self.episode_length = episode_length            
            
        # Initializinhg self decralred variables
        self.frame_id = 0
        self.episode_id = 0
        self.T_avg_blank = 0
        self.adv_angle = 38
        
        self.state = np.zeros((self.dd_size, self.dd_size))
        self.posR = 0
        self.posC = 3
        self.void_count = 0
        done = False
        
        # History
        self.history = {'past_states' : [],
                        'past_actions' : [],
                        'past_rewards' : [],
                        'past_avg_torques' : []
                        }
        
        # 
        reset_file_name = 'reset_' + str(self.dd_size) + 'X' + str(self.dd_size) + '.mn'
        # print(reset_file_name)
        self.magnet_model_loc_reset = self.main_folder_loc / reset_file_name
        
        # 
        set_file_name = 'set_' + str(self.dd_size) + 'X' + str(self.dd_size) + '.mn'
        # print(set_file_name)
        self.magnet_model_loc_set = self.main_folder_loc / set_file_name
        
        self.mn = win32.dynamic.Dispatch('MagNet.Application')
        self.mn.Visible = False
        
    def check_geometry(self, create_geo=False):
        
        if create_geo == True:
            self.state[1:-1,1:-1] = 1
        
        # Convert previous pointer location to air
        self.state[self.state == 5] = 0
        
        # Setting the path around the dd
        self.state[1,1:-1] = 3
        self.state[-2,1:-1] = 3
        self.state[1:-1,-2] = 3
        
        # Checking if 
        if self.state[self.posR, self.posC] == -1:
            raise('Controller out of design domain')
            
        # Set the current location of pointer
        self.state[self.posR, self.posC] = 5        
    
    def reset(self):
        
        doc = self.mn.openDocument(str(self.magnet_model_loc_reset))
        
        if self.episode_id == 0:
            self.T_avg_blank = self.performance_evaluation(self.adv_angle)
                    
        self.state = np.ones((self.dd_size+4, self.dd_size+3)) *(-1)
        self.posR = 1
        self.posC = 4
        
        self.check_geometry(create_geo=True)
        self.history['past_states'].append(np.copy(self.state))
        self.history['past_avg_torques'].append(self.T_avg_blank)
        self.frame_id = 0
        self.void_count = 0
        done = False
        
        return (self.state)
    
            ############
            # 0: Right #
            # 1: Left  #
            # 2: Up    #
            # 3: Down  #
            ############
    
    def step(self, action):
        self.frame_id += 1
        if action == 0:
            self.posC += 1
            
            if self.state[self.posR, self.posC] == -1:
                self.posC -= 1
                issue = 3
            elif self.state[self.posR, self.posC] == 3:
                issue = 2
            elif self.state[self.posR, self.posC] == 0:
                issue = 1
            else:
                issue = 0
                
        elif action == 1:
            self.posC -= 1
            
            if self.state[self.posR, self.posC] == -1:
                self.posC += 1
                issue = 3
            elif self.state[self.posR, self.posC] == 3:
                issue = 2
            elif self.state[self.posR, self.posC] == 0:
                issue = 1
            else:
                issue = 0
                
        elif action == 2:
            self.posR -= 1
            
            if self.state[self.posR, self.posC] == -1:
                self.posR += 1
                issue = 3
            elif self.state[self.posR, self.posC] == 3:
                issue = 2
            elif self.state[self.posR, self.posC] == 0:
                issue = 1
            else:
                issue = 0
        
        elif action == 3:
            self.posR += 1
            
            if self.state[self.posR, self.posC] == -1:
                self.posR -= 1
                issue = 3
            elif self.state[self.posR, self.posC] == 3:
                issue = 2
            elif self.state[self.posR, self.posC] == 0:
                issue = 1
            else:
                issue = 0
        
        if (issue == 1) or (issue == 2):
            R = 0
            self.check_geometry()
            # print('Issue 1 or 2')
        elif issue == 3:
            R = -10
            # print('Issue 3')
        elif issue == 0:
            # print('Issue 0')
            self.check_geometry()
            self.modify_geometry()
            T_avg = self.performance_evaluation()            
            R = T_avg - self.history['past_avg_torques'][-1]
            R *= 5
            self.history['past_avg_torques'].append(T_avg)
            
        self.history['past_states'].append(np.copy(self.state))

        if (self.void_count >= self.episode_length) or (self.frame_id >= self.episode_length):
            done = True
        else:
            done = False
            
        return (self.state, R, done)
            
    def modify_geometry(self):
        previous_state = np.copy(self.history['past_states'][-1])
        previous_state[previous_state == 5] = 0
        current_geo = np.copy(self.state[2:-2, 1:-2])
        current_geo[current_geo == 5] = 0
        diff_state = current_geo - previous_state[2:-2, 1:-2]
        self.void_count = (current_geo == 0).sum()
        
        # print('diff_state:', diff_state)
        
        diff = np.where(diff_state != 0)
        diff_row = diff[0][0]
        diff_col = diff[1][0]
        
        diff_row_list = diff_row.tolist()
        diff_col_list = diff_col.tolist()
        diff_row_list2 = []
        diff_col_list2 = []
        if not isinstance(diff_row_list, list):
            # print(diff_row_list)
            diff_row_list2.append(diff_row_list)
            diff_col_list2.append(diff_col_list)
            # print('After:', diff_row_list2)
        else:
            diff_row_list2 = diff_row_list
            diff_col_list2 = diff_col_list
                        
        # print('Row & col:', diff_row_list2)
        # print(diff_col_list2)
        air = 'Virtual Air';
        
        for inx, _ in enumerate(diff_row_list2):
            # print(inx)
            # print(diff_col_list2[inx])
            comp_name = 'Component#' + str(diff_col_list2[inx] + 1) + '_' + str(self.dd_size - diff_row_list2[inx])
            
            # print(comp_name)
            command = 'Call getDocument().setParameter("'+comp_name+'","Material", "'+air+'",infoStringParameter)'
            self.mn.processCommand(command)
            
            comp_name = 'Component#' + str(diff_col_list2[inx] + 1) + '_' + str(self.dd_size + diff_row_list2[inx]+1)
            # print(comp_name)
            command = 'Call getDocument().setParameter("'+comp_name+'","Material", "'+air+'",infoStringParameter)'
            self.mn.processCommand(command)            
    
    def performance_evaluation(self, adv_angle=38):
        self.mn.processCommand('Call getDocument().setParameter("","CurrentRated","%CurrentRatedMax",infoNumberParameter)')
        self.mn.processCommand('Call getDocument().setParameter("","xCurrentAdvanceAngle","' + str(adv_angle) + '",infoNumberParameter)')
        
        command = 'getDocument.solveTransient2dWithMotion()'
        self.mn.processCommand(command)
        
        command = 'CALL getDocument().getSolution().getGlobalSolutionTimeInstants(1,t)'
        self.mn.processCommand(command)
        
        command = 'Call setVariant(1,t)'
        self.mn.processCommand(command)
                
        time_instants = self.mn.getVariant(1)
        torque = []
        
        for t in time_instants:
            self.mn.processCommand('CALL getDocument().getSolution().getTorqueOnBody(Array(1,' + str(t) + '),2, Array(0,0,0),,, T_z)')
            self.mn.processCommand('Call setVariant(0,T_z)')
            torque.append(self.mn.getVariant(0))
            
        torque_series = pd.Series(data=torque, index=time_instants)
        avg_torque = 4*(sum(torque) / len(torque))
        # print('Torque Avg:', avg_torque)
                
        command = 'Call getDocument().save("'+ str(self.magnet_model_loc_set) + '", infoMinimalModel)'
        self.mn.processCommand(command)
        
        if self.frame_id == self.episode_length:
            command = 'Call getDocument().getView().close(False)'
            self.mn.processCommand(command)
            
            # Only when the whole training is over
                # command = 'Call close(False)'
                # self.mn.processCommand(command)
            
        return(avg_torque)
        
        
        
    