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

        print(temp_values)
        return max(temp_values)

    
    phi1 = calculate_phi(tr, tr1, env, env1, "b", "c")
    phi2 = calculate_phi(tr, tr1, env, env1, "c", "b")

    
    return max(phi1, phi2)