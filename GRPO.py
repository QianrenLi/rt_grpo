import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

from torch.utils.tensorboard import SummaryWriter

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path

from collections import deque
from tqdm import tqdm

device = th.device("cuda" if th.cuda.is_available() else "cpu")

group_num = 1 # This should be 1 in default
trajectories_per_update = 50 # This is the group size
trajectory_len = 15 # This is the horizon length
beta = 0.0001

def collect_trajectory(env, policy, trajectory_len = 15, deterministic = False):
    obs = env.reset()
    log_probs = []
    observations = []
    chosen_actions = []
    for _ in range(trajectory_len):
        with th.no_grad():
            action_tensor, _, log_prob = policy(th.as_tensor(obs, device=device), deterministic= deterministic)
            
        action = action_tensor.cpu().numpy()  # 去除批次维度
        ## action less than -10 or greater than 10 be 10
        action = np.clip(action, -10, 10)
        obs, reward, done, infos = env.step(action)
        observations.append(obs)
        log_probs.append(log_prob.sum().item())
        chosen_actions.append(action)
        if done:
            break
    normalized_reward = np.mean([reward])
    return observations, log_probs, chosen_actions, normalized_reward

import numpy as np

def compute_advantages(trajectories, N):
    # Split the trajectories into N groups
    group_size = len(trajectories) // N
    split_trajectories = [trajectories[i * group_size: (i + 1) * group_size] for i in range(N)]

    # Compute advantages for each group
    advantages = []
    for group in split_trajectories:
        rewards = [r for o, l, a, r in group]
        mean_reward = sum(rewards) / len(rewards)
        std_reward = np.std(rewards) + 1e-8
        group_advantages = [(r - mean_reward) / std_reward for r in rewards]
        advantages.extend(group_advantages)

    return advantages

def grpo_update(trajectories, policy, eps=0.2, n_iterations=20, max_grad_norm = None, ref_policy = None, writer = None, epi = None):
    # rewards = [r for o, l, a, r in trajectories]
    # mean_reward = sum(rewards) / len(rewards)
    # std_reward = np.std(rewards) + 1e-8
    # advantages = [(r - mean_reward) / std_reward for r in rewards]
    advantages = compute_advantages(trajectories, group_num)
    
    policy.set_training_mode(True)
    
    for iter_num in range(n_iterations):
        loss = 0
        for traj, advantage in zip(trajectories, advantages):
            observations, old_log_probs, chosen_actions, _ = traj
            trajectory_loss = 0
            for t, _ in enumerate(observations):
                obs_tensor = th.as_tensor(observations[t], device=device)
                action_tensor = th.as_tensor(chosen_actions[t], device=device)
                _, new_log_prob, _ = policy.evaluate_actions(obs_tensor, action_tensor)
                ratio = th.exp(new_log_prob.sum() - old_log_probs[t])
                clipped_ratio = th.clamp(ratio, min=1 - eps, max=1 + eps)
                
                policy_loss_1 = advantage * ratio
                policy_loss_2 = advantage * clipped_ratio
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                trajectory_loss += policy_loss
                
                with th.no_grad():
                    _, ref_log_prob, _ = ref_policy.evaluate_actions(obs_tensor, action_tensor)
                log_ratio = ref_log_prob - new_log_prob
                log_ratio = th.clamp( log_ratio, max = 10 ) # Avoid inf div
                kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                trajectory_loss += beta * kl_div
                
            trajectory_loss /= len(observations)
            loss += trajectory_loss
            
        loss /= len(trajectories)
        
        policy.optimizer.zero_grad()
        loss.backward()
        
        if writer is not None and epi is not None:
            writer.add_scalar('Training/Loss', loss, epi * n_iterations + iter_num)
                
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy.optimizer.step()
        
def log_dir_gen(folder, alg):
    import re
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    existing_dirs = os.listdir(folder)
    pattern = re.compile(rf"^{alg}_(\d+)$")
    
    max_x = 0
    for d in existing_dirs:
        match = pattern.match(d)
        if match:
            max_x = max(max_x, int(match.group(1)))
    
    return os.path.join(folder, f"{alg}_{max_x + 1}")

def gen_env_model_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="if toggled, run evaluation instead of training",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    
    return model_path, env, env_name, args
    
def enjoy() -> None:  # noqa: C901
    model_path, env, env_name, args = gen_env_model_path()
    GRPO_model_path = model_path.replace('best_model', f'grpo_model_gs_{trajectories_per_update}_beta_{beta}')
    
    if args.eval:    
        model = ALGOS['ppo'].load(GRPO_model_path)
        policy = model.policy  # 提取完整的策略网络
        
        reward = collect_trajectory(env, policy, trajectory_len, deterministic=True)[-1]
        print("Eval Reward: ", reward)
        exit()
        
    model = ALGOS['ppo'].load(model_path)
    policy = model.policy  # 提取完整的策略网络
    maximum_norm = model.max_grad_norm
        
    ref_model = ALGOS['ppo'].load(model_path)
    ref_policy = ref_model.policy

    folder = f'./logs/{env_name.gym_id}'
    log_dir = log_dir_gen(folder, 'GRPO')
    writer = SummaryWriter(log_dir=log_dir)
    
    episode_reward_window = deque(maxlen=100)
    max_running_reward = 0
    for i_episode in range(5000):
        trajectories = []
        episode_rewards = []
        
        for _ in range(trajectories_per_update):
            obs, log_probs, actions, norm_reward = collect_trajectory(env, policy, trajectory_len = trajectory_len)
            trajectories.append((obs, log_probs, actions, norm_reward))
            episode_rewards.append(norm_reward)
            writer.add_scalar('Episode/Reward', norm_reward, i_episode * trajectories_per_update + _)

        avg_reward = np.mean(episode_rewards)
        writer.add_scalar('Training/Avg Reward', avg_reward, i_episode)
        writer.add_scalar('Training/Max Reward', max(episode_rewards), i_episode)
        
        if avg_reward > max_running_reward:
            avg_reward = max_running_reward
            model.save(GRPO_model_path)

        grpo_update(trajectories, policy, n_iterations=20, max_grad_norm=maximum_norm, ref_policy = ref_policy, writer=writer, epi= i_episode)

        episode_reward_window.extend(episode_rewards)

        print(f'Episode {i_episode}, Avg Reward: {np.mean(episode_rewards):.2f}')
        print(f'Episode {i_episode}, Max Reward: {np.max(episode_rewards):.2f}')
                
    writer.close()
    env.close()
    
if __name__ == "__main__":
    enjoy()
