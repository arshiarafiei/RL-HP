from CQLearning_hyprl import CQLearning
from plotting import plot_v2
from parsing import get_options, save_param
from CustomLogger import CustomLogger
import numpy as np


map_names = ['MIT']
# map_names = ['ISR', 'Pentagon', 'MIT', 'SUNY']
# map_names = ['ISR', 'Pentagon', 'MIT', 'SUNY']


agents, shielding, iterations, display, save, grid, fair, extra, collision_cost, alpha, discount, episodes , d_max,\
           t_thresh, c_thresh, c_max, start_c, delta, nsaved, noop = get_options()
steps_test = 100
ep_test = 10
conv = False
convs = dict.fromkeys(map_names, [])
# collision_cost = 30
# noop = True
print('Collision cost : ', collision_cost, ' - Shielding :', shielding, ' - noop : ', noop, '- grid: ', grid)

def format_data(steps, acc, coll, inter, ep):
    info = {}
    info['steps'] = steps
    info['acc_rewards'] = acc
    info['collisions'] = coll
    info['interferences'] = inter
    info['episodes'] = ep

    return info


# Loop over all maps
for m in map_names:

    cq = CQLearning(map_name=m, nagents=agents, grid=grid, alpha=alpha, disc=discount,
                    d_max=d_max, t_thresh=t_thresh, c_thresh=c_thresh, c_max=c_max,
                    start_c=start_c, d=delta, ns=nsaved)

    i_step_max, i_episode_max, step_max, episode_max = cq.get_recommended_training_vars()

    if episodes is not None:
        episode_max = episodes

    train_data = []
    test_data = []
    done = False
    i = 0
    print('map : ', m)
    # print("\n *************************** map : ",m," iteration ", i+1, "/", iterations, "**************************")
    cq.initialize_qvalues(step_max=i_step_max, episode_max=i_episode_max, c_cost=collision_cost, noop=noop)

    s, acc, coll, inter = cq.run(step_max=step_max, episode_max=episode_max, noop = noop,
                                    debug=False, shielding=shielding, grid=grid, fair=fair, c_cost=collision_cost)


    # train_data_i = format_data(s, acc, coll, inter, episode_max)
    # plot_v2(train_data_i, agents=agents, iteration=i + 1, map=m, test=False, shielding=shielding, save=save,
    #         display=False)

    s2, acc2, coll2, inter2 = cq.run(step_max=steps_test, episode_max=ep_test, noop = noop,
                                        testing=True, debug=False, shielding=shielding, grid=grid, fair=fair, c_cost=collision_cost)
    
    with open("test1.txt", "a") as file:
        file.write(f"Map: {m}\n")
        file.write(f"---- Step :{np.mean(s2)}\n")
        file.write(f"---- Coll :  {np.mean(coll2)}\n")

        
    print('  ---- Step : ', np.mean(s2))
    print('step', s2)
    print('  ---- Coll : ', np.mean(coll2))
    print(coll2)
    print('  ---- Inter : ', np.mean(inter2))
    print(inter2)

