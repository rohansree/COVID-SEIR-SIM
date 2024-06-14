import particles
import simulator
import csv
import matplotlib.pyplot as plt
import os
import numpy as np

path = ("plot")
CHECK_FOLDER = os.path.isdir(path)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(path)
    
# frequency of plotting the epidemic states and saving images
plot_freq = 1000

# the object of class simulator 
sim = simulator.Simulator()   

# the object of class Particles 
particles = particles.Particles(sim)


def print_statistics(step, sim, particles):
    total_susceptible = np.count_nonzero(particles.states['susceptible'])
    total_quarantined = np.count_nonzero(particles.states['true_qua']) + np.count_nonzero(particles.states['false_qua'])
    total_exposed = np.count_nonzero(particles.states['exposed'])
    total_infected = np.count_nonzero(particles.states['infected'])
    total_recovered = np.count_nonzero(particles.states['recovered'])
    total_dead = np.count_nonzero(particles.states['dead'])

    total_isolated = np.count_nonzero(particles.states['true_iso']) + np.count_nonzero(particles.states['false_iso'])
    total_severe_infected = np.count_nonzero(particles.states['severe_inf'])

    total_population = sum([total_susceptible, total_quarantined, total_exposed, total_infected, total_recovered, total_dead, total_isolated, total_severe_infected])
    print(total_population)


    susceptible_percentage = (total_susceptible / total_population) * 100
    exposed_percentage = (total_exposed / total_population) * 100
    infected_percentage = (total_infected / total_population) * 100
    recovered_percentage = (total_recovered / total_population) * 100
    dead_percentage = (total_dead / total_population) * 100
    quarantined_percentage = (total_quarantined / total_population) * 100
    isolated_percentage = (total_isolated / total_population) * 100
    severe_infected_percentage = (total_severe_infected / total_population) * 100

    print(f"Step {step}:")
    print(f"Total Susceptible: {total_susceptible} ({susceptible_percentage:.2f}%)")
    print(f"Total Exposed: {total_exposed} ({exposed_percentage:.2f}%)")
    print(f"Total Infected: {total_infected} ({infected_percentage:.2f}%)")
    print(f"Total Severe Infected: {total_severe_infected} ({severe_infected_percentage:.2f}%)")
    print(f"Total Recovered: {total_recovered} ({recovered_percentage:.2f}%)")
    print(f"Total Dead: {total_dead} ({dead_percentage:.2f}%)")
    print(f"Total Quarantined: {total_quarantined} ({quarantined_percentage:.2f}%)")
    print(f"Total Isolated: {total_isolated} ({isolated_percentage:.2f}%)")
    print()

for i in range(sim.number_of_iter):
    if i%10==0:
        print("Completed {}/{} iterations".format(i, sim.number_of_iter))
    
    # update the records on easch epidemic state 
    particles.update_states(i, sim)
    
    # update the velocities and coordinates of particles
    particles.update_velocities(i, sim)
    particles.update_coordinates(sim)
    
    # number of vaccines for current iteration
    vac_iter = particles.vac_per_iter(i, sim)
    # particles.apply_vaccination(i, sim, vac_iter)
    # print('Vaccinated: ', vac_iter)
    
    # contacts of contagious particle
    contact_sub = particles.get_contact(i, sim)
    
    # increment the timer for each state
    particles.time_cur_state = particles.time_cur_state + sim.delta_t  
    
    # Add calibration time from the main folder
    calibration_time = 1
    days = i*sim.delta_t + calibration_time

    # Susceptible particles that got exposed to the infection
    new_cases = particles.get_new_cases_ids(i, sim)

    #decay the effectiveness of the vaccine
    #decay the effectiveness of recovery
    sim.decay_sir()

    #make sure mortality is propagated correctly!
    #it would be nice if 0 deaths was the truth in real life also. 
    sim.update_mortality(particles)

    # Susceptible to Exposed transition
    sim.susceptible_to_exposed(particles, new_cases)
        
    # Trace contacts of the positive tested particles and send them to 
    # quarantined or isolated states
    sim.pos_to_trace(particles, i, contact_sub)

    # Exposed to Infected transition
    sim.exposed_to_infected(particles)
    
    # True Quarantined to True Isolated transition
    sim.quat_to_isot(particles)
    
    # False Quarantined to Susceptible transition
    sim.quaf_to_sus(particles)
    
    # Infected to Recovered transition
    sim.infected_to_recovered(particles)
    
    # Infected to Severe Infected transition
    sim.infected_to_severe_infected(particles, i)
    
    # False Isolated to Susceptible transition
    sim.isof_to_sus(particles)
    
    # True Isolated to Recovered transition
    sim.isot_to_rec(particles)
    
    # True Isolated to Severe Infected transition
    sim.isot_to_sevinf(particles, i)
    
    # True Positive to True Quarantined transition
    sim.tp_to_tqiso(particles, i)
    
    # False Positive to False Isolated transition
    sim.fp_to_fiso(particles, i)
    
    # Severe Infected to Dead/Recovered transntion
    sim.severe_infected_to_dead_recovered(particles, i)
    
    # Random vaccination 
    sim.random_vac(particles, i, vac_iter)
    
    # print_statistics(i, sim, particles)

    #plot the epidemic states
    if i>=plot_freq and i%plot_freq==0:
        print_statistics(i, sim, particles)#print BEFORE plotting
        plot = particles.plot(sim, i)
        


    

