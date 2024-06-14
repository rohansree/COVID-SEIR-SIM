import numpy as np
import math
import particles
import pickle



class Simulator(): 
    def __init__(self):
        '''
        The Simulator object contains model parameters for the epidemic 
        simulation of the population of interest. Particles move in a 2D
        map represented by a square with x limits (-1, 1) and y limits (-1, 1).
        '''
        self.NUMBER_OF_PARTICLES = 5000 
        self.INITIAL_EXPOSED = 50
        self.AGE_GROUPS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.AGE_DISTRS = [500, 500, 500, 500, 500, 500, 500, 500, 1000]
        
        self.SIMULATION_LENGTH = 10
        self.INIT_V_MAX = 1
        self.KT = 20
        self.KA = 15
        self.KDT_FREQ = 10
        self.MORTALITY_RATE = 0.15
        self.MOR_IMM1 = 0.52
        self.MOR_IMM2 = 0.95
        
        self.TESTING_RATE = 0.8
        self.TRACING = 0
        self.NTEST = 0
        self.TEST_SN = 0.95
        self.TEST_SP = 0.99
        
        self.TRANSMISSION_RATE_EXPOSED = 0.7
        self.TRANSMISSION_RATE_SEVERE_INFECT = 0.3
        self.TRANSMISSION_RATE_QUA = 0.3
        self.SIR = [0, 0.1, 0.3, 0.8, 0.15, 0.6, 0.22, 0.51, 0.93]
        self.SER = 0.093

        self.SIR_DECAY = 0.95  #decay factor for regular immunization
        
        self.mean_dst = 1/math.sqrt(self.NUMBER_OF_PARTICLES)
        self.init_cont_threshold = self.mean_dst/self.KT
        self.speed_gain = self.INIT_V_MAX/self.KA
        self.delta_t = self.init_cont_threshold/self.INIT_V_MAX
        self.number_of_iter = math.ceil(self.SIMULATION_LENGTH/self.delta_t)    
        
        self.DAILY_VAC = 2
        self.VAC_FLOAT = self.DAILY_VAC*self.NUMBER_OF_PARTICLES/(1000*self.delta_t)
        self.VAC_FLOOR = math.floor(self.VAC_FLOAT)
        self.VAC_CEIL = math.ceil(self.VAC_FLOAT)
        self.RAND_VAC = 1 # 1 for vaccination; 0 for no vaccination
        self.VAC_AGE = 0 # 1 for random all, 0 for age-based vaccination
        
        self.T_IMM1 = math.ceil(12/self.delta_t)*self.delta_t
        self.T_IMM2 = math.ceil(28/self.delta_t)*self.delta_t
        self.T_INF = 4
        self.T_EXP = 2

        # Define vaccine profiles
        self.VACCINES = {
            # 'vaccine_A': {'doses': 1, 'effectiveness': [0.24], 'decay': 0.95}, #J&J
            'vaccine_C': {'doses': 2, 'effectiveness': [0.5, 0.1], 'decay': 0.95},#Novavax
            'vaccine_B': {'doses': 2, 'effectiveness': [0.5, 0.05], 'decay': 0.95},#Pfizer or Moderna
        }
         
    def susceptible_to_exposed(self, model, susceptible_contacted): 
        '''
        Class method to transition epidemic status of the particles from 
        susceptible to exposed.
        '''
        vac_status = model.vac_imm[susceptible_contacted]
        # print(vac_status)
        vac_type = model.vac_type[susceptible_contacted]

        effectiveness = 0
        if len(vac_status) > 0:
            if(vac_status[0] > 0):
                effectiveness = self.VACCINES[vac_type]['effectiveness'][vac_status[0] - 1]
            
        
        transmission_rate = self.TRANSMISSION_RATE_EXPOSED * (1 - effectiveness)
    
        if np.random.random() < transmission_rate:
            model.epidemic_state[susceptible_contacted] = 1
            model.time_cur_state[susceptible_contacted] = 0
    
    def decay_sir(self):
        '''
        Method to decay the SIR and SIR_VAC values by the decay factor.
        '''

        # one vaccine, one dose
        # self.SIR_VAC = [vac * self.SIR_VAC_DECAY for vac in self.SIR_VAC]
        self.SIR = [suspect * self.SIR_DECAY for suspect in self.SIR ]

        #one vaccine, two doses
        # self.SIR_VAC_FIRST_DOSE = [vac * self.SIR_VAC_DECAY for vac in self.SIR_VAC_FIRST_DOSE]
        # self.SIR_VAC_SECOND_DOSE = [vac * self.SIR_VAC_DECAY for vac in self.SIR_VAC_SECOND_DOSE]

        for vaccine in self.VACCINES.values():
            vaccine['effectiveness'] = [eff * vaccine['decay'] for eff in vaccine['effectiveness']]

    def apply_mortality_reduction(self, mortality_rate, vac_status, vac_type):
        '''
        Apply mortality reduction based on vaccination status.
        '''

        #mortality was not propagated properly!
        #this is a vestige of single vaccine
        # if vac_status == 1:
        #     return mortality_rate * (1 - self.MOR_IMM1)
        # elif vac_status == 2:
        #     return mortality_rate * (1 - self.MOR_IMM2)

        if vac_status >0:
            # transmission_rate = self.TRANSMISSION_RATE_EXPOSED * (1 -self.VACCINES[vac_type]['effectiveness'][vac_status-1])
            return mortality_rate * (1 - self.VACCINES[vac_type]['effectiveness'][vac_status-1])
        return mortality_rate

    def update_mortality(self, model):
        '''
        Update mortality status of individuals based on their vaccination status.
        '''
        for i in self.AGE_GROUPS:
            severe_inf = np.where((model.epidemic_state == 7) & (model.ages == i))
            for idx in severe_inf[0]:
                mortality_rate = self.MORTALITY_RATE
                vac_status = model.vac_imm[idx]
                vac_type = model.vac_type[idx]
                adjusted_mortality_rate = self.apply_mortality_reduction(mortality_rate, vac_status, vac_type)
                if np.random.random() < adjusted_mortality_rate * self.delta_t:
                    model.epidemic_state[idx] = 4
        
    def pos_to_trace(self, model, i, contact):
        '''
        Class method to transition epidemic status of the particles from 
        susceptible to exposed.
        '''
        
        ind_recent_inf = np.where(model.test_res>=(i-1)) 
        filtered_by_app = np.where(model.app[ind_recent_inf]==1)
        to_iter = np.intersect1d(contact, filtered_by_app)
                
        for k in range(len(to_iter)):
            if type(model.contact_cell[k, 0])!=list:
                continue
            else:
                m = len(model.contact_cell[k, 0])
            for j in range(m):
                if (model.contact_cell[k, 1][j]>=(self.i*self.delta_t-self.T_INF)):
                    if model.epidemic_state[model.contact_cell[[k, 0], j]]==0:
                        model.epidemic_state[model.contact_cell[[k, 0], j]] = 9
                        model.time_cur_state[model.contact_cell[[k, 1], j]] = 0
                    elif model.epidemic_state[model.contact_cell[[k, 0], j]]==1:
                        model.epidemic_state[model.contact_cell[[k, 0], j]] = 5
                        
                    elif model.epidemic_state[model.contact_cell[[k, 0], j]]==2:
                        model.epidemic_state[model.contact_cell[[k, 0], j]] = 6
    
    
    def exposed_to_infected(self, model): 
        '''
        Class method to transition epidemic status of the particles from 
        Exposed to Infected state. 
        '''
        
        to_inf = np.where((model.epidemic_state==1) & (model.time_cur_state >= self.T_EXP))
        model.epidemic_state[to_inf] = 2
        model.time_cur_state[to_inf] = 0
        
        
    def infected_to_recovered(self, model):
        '''
        Class method to transition epidemic status of the particles from 
        Infected to Recovered state. 
        '''

        infected_passed_t_inf = np.where((model.epidemic_state==2) & (model.time_cur_state >= self.T_INF))  
        model.epidemic_state[infected_passed_t_inf] = 3

    
    def infected_to_severe_infected(self, model, i): 
        '''
        Class method to transition epidemic status of the particles from 
        Infected to Severe Infected. 
        '''
        #modified based on vaccination status
        for i in self.AGE_GROUPS:
                    ind_sevinf = np.where((model.epidemic_state == 2) & (model.ages == i) & (model.vac_imm == 0))
                    if len(ind_sevinf[0]) > 0:
                        temp_ar = np.random.random((len(ind_sevinf[0]), 1))
                        fil = np.where(temp_ar < (self.SIR[i-1] * self.delta_t))
                        to_sev_inf = ind_sevinf[0][fil[0]]
                        model.epidemic_state[to_sev_inf] = 7

                    for vac_type, properties in self.VACCINES.items():
                        doses = properties['doses']
                        for dose in range(doses):
                            ind_sevinfvac = np.where((model.epidemic_state == 2) & (model.ages == i) & (model.vac_imm == dose + 1) & (model.vac_type == vac_type))
                            if len(ind_sevinfvac[0]) > 0:
                                temp_ar = np.random.random((len(ind_sevinfvac[0]), 1))
                                model.epidemic_state[ind_sevinfvac[temp_ar < (properties['effectiveness'][dose] * self.delta_t)]] = 7



    def severe_infected_to_dead_recovered(self, model, i):
        '''
        Class method to transition epidemic status of the particles from 
        Severe Infected to Dead/Recovered state. 
        '''
        
        temp = np.random.rand(self.NUMBER_OF_PARTICLES, 1)
        ind_end_severe_inf = np.where((model.time_cur_state >= self.T_INF) & (model.epidemic_state == 7) & (temp>self.MORTALITY_RATE))
        model.epidemic_state[ind_end_severe_inf] = 3 
        ind_severe_inf = np.where((model.time_cur_state >= self.T_INF) & (model.epidemic_state == 7) & (temp<self.MORTALITY_RATE))
        model.epidemic_state[ind_severe_inf] = 4
        
        
    
    def random_vac(self, model, i, vac_iter):
        '''
        Class method for random all and age-based vaccination. 
        '''
    
        #NOT in age order
        if self.RAND_VAC == 1:
            if self.VAC_AGE == 1:
                
                
                vac_type = np.random.choice(list(self.VACCINES.keys())) 


                not_vac_ind = np.where((model.vac == 0) & ((model.epidemic_state == 0) | 
                                                                                (model.epidemic_state == 1) | 
                                                                                (model.epidemic_state == 2) | 
                                                                                (model.epidemic_state == 3)) & 
                                                                                (model.ages == age), 1, 0)
                if sum(not_vac_ind) > 0:
                    vac_ind = np.where(not_vac_ind == 1)
                    order = np.random.permutation(len(vac_ind[0]))
                    vac_ind = vac_ind[0][order]
                    if len(vac_ind) > vac_iter:
                        vac_ind = vac_ind[:vac_iter]
                    
                    model.vac[vac_ind] = i * self.delta_t
                    model.vac_type[vac_ind] = vac_type
                    model.vac_dose[vac_ind] = 1
                    model.vac_effectiveness[vac_ind] = self.VACCINES[vac_type]['effectiveness'][0]
                    t_vac = i * self.delta_t - model.vac

                    vac_iter -= len(vac_ind)

            #IN age order
            else:
                
                vac_type = np.random.choice(list(self.VACCINES.keys())) 

                #*actually, reverse age order (old people first)
                for age in reversed(self.AGE_GROUPS):
                    not_vac_ind = np.where((model.vac == 0) & ((model.epidemic_state == 0) | 
                                                                (model.epidemic_state == 1) | 
                                                                (model.epidemic_state == 2) | 
                                                                (model.epidemic_state == 3)) & 
                                                                (model.ages == age), 1, 0)
                    

                    if sum(not_vac_ind) > 0:
                        vac_ind = np.where(not_vac_ind == 1)
                        order = np.random.permutation(len(vac_ind[0]))
                        vac_ind = vac_ind[0][order]
                        if len(vac_ind) > vac_iter:
                            vac_ind = vac_ind[:vac_iter]
                        
                        model.vac[vac_ind] = i * self.delta_t
                        model.vac_type[vac_ind] = vac_type
                        model.vac_dose[vac_ind] = 1
                        model.vac_effectiveness[vac_ind] = self.VACCINES[vac_type]['effectiveness'][0]
                        t_vac = i * self.delta_t - model.vac

                        vac_iter -= len(vac_ind)
                        if vac_iter <= 0:
                            break


         
            
    def tp_to_tqiso(self, model, i):
        '''
        Class method to transition True Positive tested particles to 
        True Quarantined or True Isolated states. 
        '''
        
        sir_ind = np.where(((model.epidemic_state==1)|(model.epidemic_state==2)))
        temp_ar = np.random.rand(self.NUMBER_OF_PARTICLES, 1)
        temp_test = np.zeros((self.NUMBER_OF_PARTICLES, 1))
        temp_test[sir_ind] = 1
        sir_ind_ts_tp = np.where((temp_test==1) & (temp_ar<(self.delta_t*self.TESTING_RATE*self.TEST_SN)))
        model.test_res[sir_ind_ts_tp] = 1
        int_qua = np.where((model.test_res!=0)&(model.epidemic_state==1))
        model.epidemic_state[int_qua] = 5
        int_iso = np.where((model.test_res==1)&(model.epidemic_state==2))
        model.epidemic_state[int_iso] = 6
        
    def fp_to_fiso(self, model, i):   
        '''
        Class method to transition False Positive tested particles to
        False Isolated state.
        '''  
        
        sir_ind = np.where(model.epidemic_state==0)
        temp_ar = np.random.rand(self.NUMBER_OF_PARTICLES, 1)
        tempp_test = np.zeros([self.NUMBER_OF_PARTICLES, 1])
        tempp_test[sir_ind] = 1
        sir_ind_ts_fp = np.where((tempp_test==1) & (temp_ar<self.TESTING_RATE*(1-self.TEST_SP)*self.delta_t))
        model.test_res[sir_ind_ts_fp] = 2
        int_iso = np.where((model.test_res!=0)&(model.epidemic_state==0))
        model.epidemic_state[int_iso] = 8
        model.time_cur_state[int_iso] = 0
        
    def isof_to_sus(self, model):
        '''
        Class method to transition epidemic status of the particles from 
        False Isolated to Susceptible super-state. 
        '''
        
        to_sus =np.where((model.epidemic_state==8) & (model.time_cur_state >= self.T_INF))
        model.epidemic_state[to_sus] = 0
        
        
    def quat_to_isot(self, model):
        '''
        Class method to transition epidemic status of the particles from 
        True Quarantined to True Isolated 
        '''
        
        to_isot =np.where((model.epidemic_state==5) & (model.time_cur_state >= self.T_EXP))
        model.epidemic_state[to_isot] = 6
        model.time_cur_state[to_isot] = 0 
        
    def quaf_to_sus(self, model):
        '''
        Class method to transition epidemic status of the particles from 
        False Quarantined. 
        '''
        
        to_sus = np.where(((model.epidemic_state==9) & (model.time_cur_state >= self.T_EXP)))
        model.epidemic_state[to_sus] = 0
        model.time_cur_state[to_sus] = 0 
        
        
    def isot_to_rec(self, model):
        '''
        Class method to transition epidemic status of the particles from 
        True Isolated to Recovered state. 
        '''
        
        to_rec = np.where((model.epidemic_state==6) & (model.time_cur_state >= self.T_INF))
        model.epidemic_state[to_rec] = 3
        
    def isot_to_sevinf(self, model, i):
        '''
        Class method to transition epidemic status of the particles from 
        True Isolated to Severe Infected sub-state.
        '''

        for i in self.AGE_GROUPS:
            ind_sevinf = np.where((model.epidemic_state==6)&(model.ages==i)&(model.vac_imm==0), 1, 0)

            if (len(ind_sevinf)>0):
                  temp_ar = np.random.random((len(ind_sevinf), 1)) 
                  temp_diff = temp_ar< self.SIR[i-1]*self.delta_t
                  model.epidemic_state[ind_sevinf[temp_diff]] = 7
            
            # ind_sevinfvac = np.where((model.epidemic_state==6)&(model.ages==i)&(model.vac_imm==1), 1, 0)
            # if (len(ind_sevinfvac)>0):
            #       temp_ar = np.random.random((len(ind_sevinfvac), 1)) 
            #       model.epidemic_state[ind_sevinfvac[temp_ar<(self.SIR_VAC[i-1]*self.delta_t)]] = 7
            
            ind_sevinfvac1 = np.where((model.epidemic_state == 2) & (model.ages == i) & (model.vac_imm == 1))
            if len(ind_sevinfvac1[0]) > 0:
                  temp_ar = np.random.random((len(ind_sevinfvac1[0]), 1)) 
                  model.epidemic_state[ind_sevinfvac1[temp_ar < (self.SIR_VAC_FIRST_DOSE[i-1] * self.delta_t)]] = 7

            ind_sevinfvac2 = np.where((model.epidemic_state == 2) & (model.ages == i) & (model.vac_imm == 2))
            if len(ind_sevinfvac2[0]) > 0:
                  temp_ar = np.random.random((len(ind_sevinfvac2[0]), 1)) 
                  model.epidemic_state[ind_sevinfvac2[temp_ar < (self.SIR_VAC_SECOND_DOSE[i-1] * self.delta_t)]] = 7
  
        
#A more efficient way to do this is to better utilize numpy arrays, vectorize the operations, and implement parallelization
#that implementation did not work properly (parallelization) and was not included in the final version

