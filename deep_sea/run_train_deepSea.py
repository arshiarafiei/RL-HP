import os
from multiprocessing import Process

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from DeapSea import DeepSeaTreasureEnv  # or wherever your env lives


def train_hypRL():
    models_dir = "models/PPO_hypRL"
    logdir     = "logs_hypRL"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir,     exist_ok=True)

    env = DeepSeaTreasureEnv(hypRL=True)
    env.reset()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        batch_size=16,
        gae_lambda=0.98,
        gamma=0.999,
        n_steps=25,
        tensorboard_log=logdir
    )

    TIMESTEPS = 500
    
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO_hypRL",
        log_interval=1
    )
    model.save(f"{models_dir}/{TIMESTEPS}_steps")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        deterministic=True,
        n_eval_episodes=1
    )
    print(f"[hypRL] Time Step {TIMESTEPS} -- mean_reward: {mean_reward:.2f} ± {std_reward:.2f}")


def train_original():
    models_dir = "models/PPO_original"
    logdir     = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir,     exist_ok=True)

    env = DeepSeaTreasureEnv()
    env.reset()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        batch_size=16,
        gae_lambda=0.98,
        gamma=0.999,
        n_steps=25,
        tensorboard_log=logdir
    )

    TIMESTEPS = 500
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO_1",
        log_interval=1
    )
    model.save(f"{models_dir}/{TIMESTEPS}_steps")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        deterministic=True,
        n_eval_episodes=1
    )
    print(f"[orig] Time Step {TIMESTEPS} -- mean_reward: {mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    p1 = Process(target=train_hypRL)
    p2 = Process(target=train_original)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
