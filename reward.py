import numpy as np
import math
from env0 import SimpleMDPEnv


def reward_env0(tr, tr1, env, env1, step):
    def calculate_phi(tr_a, tr_b, env_a, env_b, target_a, target_b):
        temp_values = []

        for i in range(max(len(tr_a), len(tr_b))):
            if i >= len(tr_b): 
                temp_values.append(1 - env_a.distance_to_target(tr_a[i], target_a))
                continue

            if i >= len(tr_a):  
                temp_values.append(1 - env_b.distance_to_target(tr_b[i], target_b))
                continue

            
            temp = min(
                1 - env_a.distance_to_target(tr_a[i], target_a),
                1 - env_b.distance_to_target(tr_b[i], target_b)
            )
            temp_values.append(temp)

        # print(temp_values)
        return max(temp_values)

    
    phi1 = calculate_phi(tr, tr1, env, env1, "b", "c")
    phi2 = calculate_phi(tr, tr1, env, env1, "c", "b")

    print(max(phi1, phi2))

    
    return max(phi1, phi2)


def reward_pcp(trajectory, trajectory1, domino, domino1, step):

    ##############Semimatch############## 

    ### phi1

    phi_1_list = list()

    domino_1_top = domino[0] + "#"

    domino_1_bottom = domino[1] + "#"


    for index in range(min(len(domino_1_top), len(domino_1_bottom))-1):

        if domino_1_top[index] ==  domino_1_bottom[index]:
            phi_1_list.append(1) ## 1 - dist (a,a) = 1
        else:
            phi_1_list.append(0)  ## 1 - dist (a,b) = 0

    phi_1 = min(phi_1_list)

    #######phi_2

    index = min(len(domino_1_top), len(domino_1_bottom)) - 1

    phi_2_list = [min(-1*int(domino_1_top[index]== "#"), int(domino_1_bottom[index]== "#")), min(int(domino_1_top[index]== "#"), -1 * int(domino_1_bottom[index]== "#"))]

    phi_2 = max(phi_2_list)

    semimatch = max(phi_1,phi_2)

    #################Match##################

    domino_2_top = domino1[0] + "#"

    domino_2_bottom = domino1[1] + "#"

    max_length = max(len(domino_2_top), len(domino_2_bottom))

    # Padding 

    # print(domino_2_top, domino_2_bottom)

    domino_2_top = domino_2_top.ljust(max_length, '#')
    domino_2_bottom = domino_2_bottom.ljust(max_length, '#')

    # print(domino_2_top,domino_2_bottom)

    match_list = list()

    for index in range(min(len(domino_2_top), len(domino_2_bottom))-1):

        if domino_2_top[index] ==  domino_2_bottom[index]:
            match_list.append(1) ## 1 - dist (a,a) = 1
        else:
            match_list.append(0)  ## 1 - dist (a,b) = 0

    match = min(match_list)

    ##################Extend##################

    # print(trajectory,trajectory1)
    extend_list= list()

    for index in range(min(len(trajectory), len(trajectory1))):

        if trajectory[index] ==  trajectory1[index]:
            extend_list.append(1) ## 1 - dist (a,a) = 1
        else:
            extend_list.append(0)  ## 1 - dist (a,b) = 0
    
    extend = min(extend_list)

   

    reward = max (-1*semimatch , min(extend, match))

    return reward

def reward_pcp_new(domino):

    ##############Semimatch############## 

    ### phi1

    phi_1_list = list()

    domino_1_top = domino[0] + "#"

    domino_1_bottom = domino[1] + "#"

    phi_1 = 1

    flag = 0


    for index in range(min(len(domino_1_top), len(domino_1_bottom))-1):

        if domino_1_top[index] ==  domino_1_bottom[index]:
            continue
        else:
            flag = 1
            phi_1 = phi_1 - 1 
    
    if flag ==0:
        phi_1 = 10


    # phi_1 = min(phi_1_list)

    #######phi_2

    # index = min(len(domino_1_top), len(domino_1_bottom)) - 1

    # phi_2_list = [min(-1*int(domino_1_top[index]== "#"), int(domino_1_bottom[index]== "#")), min(int(domino_1_top[index]== "#"), -1 * int(domino_1_bottom[index]== "#"))]

    # phi_2 = max(phi_2_list)

    # semimatch = max(phi_1,phi_2)

    semimatch = phi_1

    #################Match##################

    domino_2_top = domino[0] + "#"

    domino_2_bottom = domino[1] + "#"

    max_length = max(len(domino_2_top), len(domino_2_bottom))

    # Padding 

    # print(domino_2_top, domino_2_bottom)

    domino_2_top = domino_2_top.ljust(max_length, '#')
    domino_2_bottom = domino_2_bottom.ljust(max_length, '#')

    # print(domino_2_top,domino_2_bottom)

    # match_list = list()

    match = 1
    flag1 = 0

    for index in range(max_length):

        if domino_2_top[index] ==  domino_2_bottom[index]:
            continue
        else:
            flag1 = 1
            match = match -1 

    if flag1 == 0:
        match = 50

    reward = max(semimatch , match)


    return reward