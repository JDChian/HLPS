import gym
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import expm
from scipy.spatial.distance import cdist
import shutil
import torch

from algos.hlps import hlps_agent
from arguments.arguments_hlps import get_args_ant
from goal_env.mujoco import *


beta = 0.00005


def plot_subgoal_frame(
        start_pos, end_pos,
        subgoal_pos, variance,
        trajectory, current_state,
        visualization_subdir, num_subgoal
    ):

    plt.figure(figsize=(8, 8))
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    
    # 1. 起點 (Start point)
    plt.scatter(start_pos[0], start_pos[1], c='blue', s=100, label='Start Point', marker='s')
    
    # 2. 終點 (End point)
    plt.scatter(end_pos[0], end_pos[1], c='green', s=150, label='Target Goal', marker='*')
    
    # 3. Subgoal
    plt.scatter(subgoal_pos[0], subgoal_pos[1], c='red', s=100, label='Subgoal', marker='x')
    
    # 4. Trajectory
    traj_arr = np.array(trajectory)
    if traj_arr.ndim == 2 and len(traj_arr) > 0:
        plt.plot(traj_arr[:, 0], traj_arr[:, 1], c='gray', alpha=0.6, label='Trajectory')
    
    # 5. Current State (End of this subgoal's execution)
    plt.scatter(current_state[0], current_state[1], c='orange', s=80, label='Current State')
    
    plt.legend(loc='upper right')
    uncertainty = variance / (variance + beta)
    plt.title(f"Subgoal {num_subgoal} | Variance: {100*variance:.3f}  | Uncertainty: {uncertainty:.3f} | Distance: {np.linalg.norm(current_state - subgoal_pos):.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    
    plt.savefig(f"{visualization_subdir}/subgoal_{num_subgoal:03d}.png")
    plt.close()


def main():
    # Get arguments
    args = get_args_ant()
    args.eval = True
    args.resume = True
    args.resume_path = 'saved_models\AntMaze1-v1_HLPS_Seed_125_May07_04-37-50'
    args.resume_epoch = 0
    args.device = 'cpu'
    args.animate = False

    # Create environment
    env = gym.make(args.env_name, max_episode_steps=10000)
    test_env = gym.make(args.test, max_episode_steps=env._max_episode_steps)
    def get_env_params(env):
        obs = env.reset()
        params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
                'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
                'max_timesteps': env._max_episode_steps}
        return params
    env_params = get_env_params(env)
    env_params['max_test_timesteps'] = test_env._max_episode_steps

    # Create agent
    sac_trainer = hlps_agent(args, env, env_params, test_env, None, None)

    # GP model parameters
    L = np.exp(sac_trainer.gplayer.ell.data.cpu().numpy())                # (1,)
    GAMMA_SQUARE = np.exp(sac_trainer.gplayer.gamma2.data.cpu().numpy())  # (1,)
    SIGMA_SQUARE = np.exp(sac_trainer.gplayer.sigma2.data.cpu().numpy())  # (1,)

    # Kalman filter constants
    LAMBDA = np.sqrt(3) / L                                                   # (1,)
    A = np.array([[0, 1], [-LAMBDA ** 2, -2 * LAMBDA]])                       # (2, 2)
    SIGMA_0 = np.array([[GAMMA_SQUARE, 0], [0, GAMMA_SQUARE * LAMBDA ** 2]])  # (2, 2)
    H = np.array([[1], [0]])                                                  # (2, 1)

    # Kalman filter initialization
    MU = np.zeros((2, 2))  # (2, 2)
    SIGMA = SIGMA_0        # (2, 2)
    
    # ==================================================
    visualize_trajectory = False
    visualize_relationship = True
    visualization_dir = 'visualizations'
    num_episodes = 100
    # ==================================================
    
    # Create directory
    if visualize_trajectory or visualize_relationship:
        if os.path.exists(visualization_dir):
            shutil.rmtree(visualization_dir)
        os.makedirs(visualization_dir, exist_ok=True)

    num_success = 0
    all_subgoal_variances = []
    all_subgoal_distances = []

    for episode in range(1, num_episodes + 1):
        # Create sub-directory
        if visualize_trajectory:
            visualization_subdir = os.path.join(visualization_dir, f'episode_{episode:03d}')
            os.makedirs(visualization_subdir, exist_ok=True)
        
        num_subgoal = 0
        trajectory = []
        
        for step in range(env_params['max_test_timesteps']):
            """
            Step 3: s' ~ P ( · | s , a )
            """
            if step == 0:
                observation = test_env.reset()
                state = observation['observation']  # (29,)
                prev_state = state.copy()           # (29,)
                initial_state = state[:2].copy()    # (2,)
                goal = observation['desired_goal']  # (2,)
            else:
                observation, rew, terminated, truncated, info = test_env.step(action)
                prev_state = state.copy()           # (29,)
                state = observation['observation']  # (29,)

            trajectory.append(state[:2].copy())

            if step != 0:
                if terminated or truncated:
                    num_success += 1
                    all_subgoal_variances.append(variance)
                    all_subgoal_distances.append(np.linalg.norm(state[:2] - subgoal))
                    
                    if visualize_trajectory:
                        plot_subgoal_frame(
                            initial_state, goal,
                            subgoal, variance,
                            trajectory, state[:2],
                            visualization_subdir, num_subgoal
                        )
                    
                    if terminated:
                        print(f'✅ Episode {episode} SUCCESS at step {step}')
                    else:
                        print(f'❌ Episode {episode} FAIL at step {step}')
                    
                    break

            with torch.no_grad():
                """
                Step 1: g_sub ~ pi^h ( · | s , g )
                """
                # Kalman filter variables
                S = np.stack([prev_state, state], axis=0)                                                             # (2, 29)
                F = sac_trainer.representation(torch.Tensor(state).to(sac_trainer.device)).detach().cpu().numpy()[0]  # (2,)
                DELTA_S = cdist(S, S)                                                                                 # (2, 2)
                PSI = expm(A * DELTA_S)                                                                               # (2, 2)
                
                # Kalman filter update
                MU = PSI @ MU                                                  # (2, 2)
                SIGMA = PSI @ SIGMA @ PSI.T + SIGMA_0 - PSI @ SIGMA_0 @ PSI.T  # (2, 2)
                K = (SIGMA @ H) / (H.T @ SIGMA @ H + SIGMA_SQUARE)             # (2, 1)
                MU = MU + K @ (F.T - H.T @ MU)                                 # (2, 2)
                SIGMA = SIGMA - K @ H.T @ SIGMA                                # (2, 2)

                # Extract state in subgoal space representation
                Z = MU[0]  # (2,)

                # Sample a subgoal every c steps
                condition = step % sac_trainer.c == 0  # Default value of sac_trainer.c is 50
                # condition = step % 100 == 0
                # condition = step == 0 or np.linalg.norm(state[:2] - subgoal) <= 5
                if condition:
                    # Record old subgoal
                    if step != 0:
                        all_subgoal_variances.append(variance)
                        all_subgoal_distances.append(np.linalg.norm(state[:2] - subgoal))
                        
                        if visualize_trajectory:
                            plot_subgoal_frame(
                                initial_state, goal,
                                subgoal, variance,
                                trajectory, state[:2],
                                visualization_subdir, num_subgoal
                            )
                        num_subgoal += 1
                        trajectory = []

                    # High-level policy: generate relative subgoal
                    hi_agent_output = sac_trainer.hi_agent.select_action(np.concatenate((state, goal)), evaluate=True)  # (2,)
                    
                    # Compute new subgoal and its uncertainty
                    if args.old_sample:
                        absolute_subgoal = hi_agent_output
                        subgoal = absolute_subgoal                                                # (2,)
                    else:
                        relative_subgoal = hi_agent_output
                        subgoal = np.clip(Z + relative_subgoal, -args.abs_range, args.abs_range)  # (2,)
                    variance = float(SIGMA[0, 0])                                                 # (1,)

                """
                Step 2: a ~ pi^l ( · | s , g_sub )
                """
                # Low-level policy: generate action
                action = sac_trainer.test_policy(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(sac_trainer.device),
                    torch.tensor(subgoal, dtype=torch.float32).unsqueeze(0).to(sac_trainer.device)
                )
    
    print(f"Final Success Rate: {num_success}/{num_episodes} = {num_success/num_episodes:.2f}")

    # Plot variance and distance relationship
    if visualize_relationship:
        unc_arr = np.array(all_subgoal_variances) / (np.array(all_subgoal_variances) + beta)
        unc_arr = np.log(unc_arr + 1) / np.log(1.1)  # log base 2
        dist_arr = np.array(all_subgoal_distances)

        plt.figure(figsize=(12, 5))
        plt.xlim(0, 1)
        plt.ylim(0, 40)
        plt.scatter(unc_arr, dist_arr, alpha=0.7)
        plt.xlabel('Uncertainty')
        plt.ylabel('Distance')
        plt.title(f'Relationship between Subgoal Uncertainties and Distances')
        plt.savefig(f"{visualization_dir}/relationship.png")
        plt.close()


if __name__ == "__main__":
    main()
