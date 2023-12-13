from cmath import inf
import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        self.last_battle_end = 0
        self.episode_threshold = 50
        self.episodes_since_guide_window_reduction = 0
        self.jsrl_guide_windows = {}
        self.explore_policy_active = False
        self.eval_enemy_win_rate = None
        self.enemy_models = 4
        self.model = None
        self.total_accuracy = 0
        self.input_layer_size = 6834
        self.max_win_rate = 0.0
        self.arena_enemy = False
        self.arena_curr_player = 1
        self.arena_curr_enemy = 1
        self.arena_model_saved = [False,False,False]

        for i in range(self.n_rollout_threads):
            # [guide_window, last_battle]
            self.jsrl_guide_windows[i] = [inf, 0]

    def run(self):
        self.warmup()

        # self.eval(1)
        # self.eval_envs.envs[0].save_replay()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        
        self.guide_index = 0
        if self.multi_player and self.eval_enemy_win_rate == None:
            for i in range(self.enemy_models):
                if(i+1 != self.enemy_models): 
                    self.restore_enemy_policy(i+1)
                    self.enemy_trainer = TrainAlgo(self.all_args, self.enemy_policy, self.jump_start_policy, device = self.device)
                    self.eval_enemy()

        for episode in range(episodes):
            if self.all_args.jump_start_model_pool_dir:
                self.jump_start_policy = self.jump_start_policy_pool[self.guide_index]
                # cycle through guide windows.
                if self.guide_index < 4:
                    self.guide_index += 1
                else:
                    self.guide_index = 0

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            if self.multi_player:
                self.available_enemy_actions = np.ones((self.n_rollout_threads, self.num_enemies, 6+self.num_agents), dtype=np.float32)
                self.enemy_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                self.enemy_obs = np.zeros((self.n_rollout_threads, self.num_enemies, self.input_layer_size), dtype=np.float32)
                if episode%100 == 0 :
                    self.model = int(episode/100) % self.enemy_models
                    print('model ============', self.model)
                    if self.model+1 != self.enemy_models:
                        self.restore_enemy_policy(self.model+1)
                        self.enemy_trainer = TrainAlgo(self.all_args, self.enemy_policy, self.jump_start_policy, device = self.device)
            

            for step in range(self.episode_length):
                # Sample actions
                if self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect_guide(step)
                    explore_values, explore_actions, explore_action_log_probs, explore_rnn_states, explore_rnn_states_critic = self.collect(step)
                else:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                for i in range(self.n_rollout_threads):
                    if (self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir) and (step - self.jsrl_guide_windows[i][1] >= self.jsrl_guide_windows[i][0]):
                        values[i] = explore_values[i]
                        actions[i] = explore_actions[i]
                        action_log_probs[i] = explore_action_log_probs[i]
                        rnn_states[i] = explore_rnn_states[i]
                        rnn_states_critic[i] = explore_rnn_states_critic[i]

                # Obser reward and next obs
                enemy_actions = None
                if self.multi_player:
                    if self.model+1 == self.enemy_models:
                        obs, enemy_obs, share_obs, rewards, dones, infos, available_actions, available_enemy_actions = self.static_envs.step(actions)
                    else:
                        enemy_actions, enemy_rnn_states = self.collect_enemy(self.enemy_obs,self.enemy_rnn_states,self.buffer.masks[step],self.available_enemy_actions)
                        obs, enemy_obs, share_obs, rewards, dones, infos, available_actions, available_enemy_actions = self.envs.step(actions,enemy_actions)
                        self.available_enemy_actions = available_enemy_actions.copy()
                        self.enemy_rnn_states = enemy_rnn_states.copy()
                        self.enemy_obs = enemy_obs.copy()
                else:
                    obs, enemy_obs, share_obs, rewards, dones, infos, available_actions, available_enemy_actions = self.envs.step(actions)
                # obs, enemy_obs, share_obs, rewards, dones, infos, available_actions, available_enemy_actions = self.envs.step(actions,enemy_actions)  if self.multi_player else self.envs.step(actions)
                    

                # if the policy is swapped to explore, and the sum rewards are greater, then great, move on to the next guide_window, I think? This is one option.
                # if self.guide_window > 0 and self.explore_policy_active and rewards[0][0][0] > 10:
                #    self.guide_window = self.guide_window - 1
                #    self.episodes_since_guide_window_reduction = 0 # set the threshold to 0.
                #    self.guide_policy_last_rewards[0][0][0] = inf

                for index, done in enumerate(dones):
                    if done.all():
                        if self.all_args.reward_speed and infos[index][0]['won']:
                            max_reward = 1840 # Hardcoded from env
                            scale_rate = 20 # Hardcoded from env.
                            map_length = step - self.jsrl_guide_windows[index][1]
                            upper_map_length_limit = 150
                            norm_max = 30
                            z1 = (map_length / upper_map_length_limit) * norm_max
                            z1 = (z1 - norm_max) * -1 # invert reward to reward lower map_length
                            speed_reward = z1 / (max_reward / scale_rate)
                            rewards[index] += speed_reward

                        self.jsrl_guide_windows[index][1] = step

                        # If we haven't set the guide window yet, set it to the last frame of last attempt, that's our starting point.
                        if self.jsrl_guide_windows[index][0] is inf:
                            self.jsrl_guide_windows[index][0] = step


                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 


                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            # self.envs.envs[0].save_replay()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                        wandb.log({"guide_window": self.jsrl_guide_windows[0][0]}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # If we make it through some number of episodes, just adjust guide window anyway
            if (self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir) and self.episodes_since_guide_window_reduction >= self.episode_threshold:
                for key in self.jsrl_guide_windows.keys():
                    if self.jsrl_guide_windows[key][0] > 0:
                        self.jsrl_guide_windows[key][0] = self.jsrl_guide_windows[key][0] - 1

                self.episodes_since_guide_window_reduction = -1

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                #self.eval_envs.envs[0].save_replay()

            self.episodes_since_guide_window_reduction += 1

    def run_arena(self):

        # self.eval(1)
        # self.eval_envs.envs[0].save_replay()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        
        self.guide_index = 0
            
        env_cnt = 0
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            # if episode%1000 == 0:
            #     reload = False
            #     if episode < 2000 and env_cnt != 0:
            #         env_cnt = 0
            #         reload = True
            #     elif episode >= 2000 and episode < 4000 and env_cnt != 1:
            #         env_cnt = 1
            #         reload = True
            #     elif episode >= 4000 and episode < 8000 and env_cnt != 2:
            #         env_cnt = 2
            #         reload = True
            #     elif episode >= 8000 and env_cnt != 3:
            #         env_cnt = 3
            #         reload = True
            #     if reload or episode == 0:
            #         print('reloading the smac environment on episode :', episode)
            #         print('current env', env_cnt)
            #         self.envs = self.arena_envs[env_cnt]
            #         self.num_agents = self.num_agents_arena[env_cnt]
            #         self.num_enemies = self.num_enemies_arena[env_cnt]
                    
            #         share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
            #         self.policy = Policy(self.all_args,
            #                     self.envs.observation_space[0],
            #                     share_observation_space,
            #                     self.envs.action_space[0],
            #                     device = self.device,
            #                     policy_identifier="Exploration Policy")
            #         self.trainer = TrainAlgo(self.all_args, self.policy, self.jump_start_policy, device = self.device)
            #         if env_cnt > 0 :
            #             self.restore()

            #         self.buffer = SharedReplayBuffer(self.all_args,
            #                                     self.num_agents,
            #                                     self.num_enemies,
            #                                     self.envs.observation_space[0],
            #                                     share_observation_space,
            #                                     self.envs.action_space[0]) 

            #         self.warmup()

            if episode%2000 == 0:
                print('reloading the smac environment on episode : ====================================', episode)
                print('current env ===================================', env_cnt)
                if env_cnt >= 6:
                    self.arena_curr_player += 1
                    if self.arena_curr_player > 3:
                        self.arena_curr_player = 1
                    env_cnt = 0
                print('current player ===================================', self.arena_curr_player)
                if env_cnt == 4:
                    print('enemy 2')
                    self.arena_enemy = True
                    if self.arena_curr_player == 1:
                        self.arena_curr_enemy = 2
                    elif self.arena_curr_player == 2:
                        self.arena_curr_enemy = 3
                    elif self.arena_curr_player == 3:
                        self.arena_curr_enemy = 1

                    self.envs = self.enemy_envs
                    self.num_enemies = 8
                    self.num_agents = 8
                    self.buffer = SharedReplayBuffer(self.all_args,
                                                self.num_agents,
                                                self.num_enemies,
                                                self.envs.observation_space[0],
                                                share_observation_space,
                                                self.envs.action_space[0]) 

                    self.available_enemy_actions = np.ones((self.n_rollout_threads, self.num_enemies, 6+self.num_agents), dtype=np.float32)
                    self.enemy_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    self.enemy_obs = np.zeros((self.n_rollout_threads, self.num_enemies, self.input_layer_size), dtype=np.float32)
                    if self.arena_model_saved[self.arena_curr_enemy-1]:
                        self.restore_enemy_policy(self.arena_curr_enemy)
                        self.enemy_trainer = TrainAlgo(self.all_args, self.enemy_policy, self.jump_start_policy, device = self.device)
                elif env_cnt == 5:
                    print('enemy 3')
                    self.arena_enemy = True
                    if self.arena_curr_player == 1:
                        self.arena_curr_enemy = 3
                    elif self.arena_curr_player == 2:
                        self.arena_curr_enemy = 1
                    elif self.arena_curr_player == 3:
                        self.arena_curr_enemy = 2

                    self.envs = self.enemy_envs
                    self.num_enemies = 8
                    self.num_agents = 8
                    self.available_enemy_actions = np.ones((self.n_rollout_threads, self.num_enemies, 6+self.num_agents), dtype=np.float32)
                    self.enemy_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    self.enemy_obs = np.zeros((self.n_rollout_threads, self.num_enemies, self.input_layer_size), dtype=np.float32)
                    if self.arena_model_saved[self.arena_curr_enemy-1]:
                        self.restore_enemy_policy(self.arena_curr_enemy)
                        self.enemy_trainer = TrainAlgo(self.all_args, self.enemy_policy, self.jump_start_policy, device = self.device)
                else:
                    self.arena_enemy = False
                    self.envs = self.arena_envs[env_cnt]
                    self.num_agents = self.num_agents_arena[env_cnt]
                    self.num_enemies = self.num_enemies_arena[env_cnt]
                    print('enemys ==================================================================', self.num_enemies)
                    
                    share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
                    self.policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device = self.device,
                                policy_identifier="Exploration Policy")
                    self.trainer = TrainAlgo(self.all_args, self.policy, self.jump_start_policy, device = self.device)

                    if self.arena_model_saved[self.arena_curr_player-1]:
                        self.restore_player(self.arena_curr_player)

                    self.buffer = SharedReplayBuffer(self.all_args,
                                                self.num_agents,
                                                self.num_enemies,
                                                self.envs.observation_space[0],
                                                share_observation_space,
                                                self.envs.action_space[0]) 

                self.warmup()
                env_cnt += 1
                # env_cnt = env_cnt%6

            for step in range(self.episode_length):
                # Sample actions
                
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                enemy_actions = None
                if self.arena_enemy:
                    enemy_actions, enemy_rnn_states = self.collect_enemy(self.enemy_obs,self.enemy_rnn_states,self.buffer.masks[step],self.available_enemy_actions)
                    obs, enemy_obs, share_obs, rewards, dones, infos, available_actions, available_enemy_actions = self.envs.step(actions,enemy_actions)
                    self.available_enemy_actions = available_enemy_actions.copy()
                    self.enemy_rnn_states = enemy_rnn_states.copy()
                    self.enemy_obs = enemy_obs.copy()
                else:
                    obs, enemy_obs, share_obs, rewards, dones, infos, available_actions, available_enemy_actions = self.envs.step(actions)
                for index, done in enumerate(dones):
                    if done.all():
                        if self.all_args.reward_speed and infos[index][0]['won']:
                            max_reward = 1840 # Hardcoded from env
                            scale_rate = 20 # Hardcoded from env.
                            map_length = step - self.jsrl_guide_windows[index][1]
                            upper_map_length_limit = 150
                            norm_max = 30
                            z1 = (map_length / upper_map_length_limit) * norm_max
                            z1 = (z1 - norm_max) * -1 # invert reward to reward lower map_length
                            speed_reward = z1 / (max_reward / scale_rate)
                            rewards[index] += speed_reward

                        self.jsrl_guide_windows[index][1] = step

                        # If we haven't set the guide window yet, set it to the last frame of last attempt, that's our starting point.
                        if self.jsrl_guide_windows[index][0] is inf:
                            self.jsrl_guide_windows[index][0] = step


                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 


                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                self.save_curr_arena_player(self.arena_curr_player)
                self.arena_model_saved[self.arena_curr_player - 1] = True

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                        wandb.log({"guide_window": self.jsrl_guide_windows[0][0]}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # if episode == 1999 or episode == 19999 or episode == 39999 or episode == 59999:
            #     print('saving a training replay ==================', incre_win_rate)
            #     self.envs.envs[0].save_replay()

            # If we make it through some number of episodes, just adjust guide window anyway
            if (self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir) and self.episodes_since_guide_window_reduction >= self.episode_threshold:
                for key in self.jsrl_guide_windows.keys():
                    if self.jsrl_guide_windows[key][0] > 0:
                        self.jsrl_guide_windows[key][0] = self.jsrl_guide_windows[key][0] - 1

                self.episodes_since_guide_window_reduction = -1

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                # self.eval(total_num_steps)
                self.eval_in_multi_map(total_num_steps)
                #self.eval_envs.envs[0].save_replay()

            self.episodes_since_guide_window_reduction += 1

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # if self.multi_player and self.static_envs is not None:
        #     self.static_envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def collect_guide(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.jump_start_policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    @torch.no_grad()
    def collect_enemy(self, obs, prev_rnn_states, masks, available_actions):
        self.enemy_trainer.prep_rollout()
        actions, rnn_states = \
            self.enemy_trainer.policy.act(np.concatenate(obs),
                                    np.concatenate(prev_rnn_states),
                                    np.concatenate(masks),
                                    np.concatenate(available_actions),
                                    deterministic=True)
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

        return actions, rnn_states

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        info_we_want_to_keep = ['average_step_rewards', 'dead_ratio']
        for k, v in train_infos.items():
            if k not in info_we_want_to_keep:
                continue # Skip info we don't care about.

            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    # def eval_in_multi_map(self, total_num_steps):
    #     maps = ['3m','8m','2s3z','3s5z']
    #     for player in range(4,9):
    #         print('player =================================== ', player)
    #         for env_cnt in range(4):
    #             self.envs = self.arena_envs[env_cnt]
    #             self.eval_envs = self.envs
    #             self.num_agents = self.num_agents_arena[env_cnt]
    #             self.num_enemies = self.num_enemies_arena[env_cnt]
                
    #             share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
    #             self.policy = Policy(self.all_args,
    #                         self.envs.observation_space[0],
    #                         share_observation_space,
    #                         self.envs.action_space[0],
    #                         device = self.device,
    #                         policy_identifier="Exploration Policy")
    #             self.trainer = TrainAlgo(self.all_args, self.policy, self.jump_start_policy, device = self.device)
    #             self.restore_player(player+1)
    #             self.eval_mp_arena(total_num_steps)
    #             print('map =================================== ', maps[env_cnt])
    
    @torch.no_grad()
    def eval_in_multi_map(self, total_num_steps):
        maps = ['3m','8m','2s3z','3s5z']
        self.avg_win_rate = 0
        self.avg_total_health_remaining = 0
        prev_trainer = self.trainer
        prev_agents = self.num_agents
        prev_enemy = self.num_enemies
        for env_cnt in range(4):
            self.eval_envs = self.arena_eval_envs[env_cnt]
            self.num_agents = self.num_agents_arena[env_cnt]
            self.num_enemies = self.num_enemies_arena[env_cnt]
            
            share_observation_space = self.eval_envs.share_observation_space[0] if self.use_centralized_V else self.eval_envs.observation_space[0]
            self.policy = Policy(self.all_args,
                        self.eval_envs.observation_space[0],
                        share_observation_space,
                        self.eval_envs.action_space[0],
                        device = self.device,
                        policy_identifier="Exploration Policy")
            self.trainer = TrainAlgo(self.all_args, self.policy, self.jump_start_policy, device = self.device)
            self.restore_player(self.arena_curr_player)
            self.eval_mp_arena(total_num_steps,maps[env_cnt])
            print('map =================================== ', maps[env_cnt])
        wandb.log({"avg_eval_win_rate_player_"+str(self.arena_curr_player): self.avg_win_rate/4}, step=total_num_steps)
        wandb.log({"avg_total_health_remaining_player_"+str(self.arena_curr_player): self.avg_total_health_remaining/4}, step=total_num_steps)
        self.trainer = prev_trainer
        self.num_agents = prev_agents
        self.num_enemies = prev_enemy
    
    @torch.no_grad()
    def eval_mp_arena(self, total_num_steps,map_name):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_healths = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_env_infos = {'total_health_remaining': 0, 'eval_average_episode_rewards': 0}

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            previous_state = eval_share_obs
            eval_obs, enemy_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions, eval_available_enemy_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            # Get relative health and shield values for units, this will only work with protoss?
            featureCount = 22

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1

                    total_relative_health = 0
                    total_relative_shield = 0
                    for agent in range(self.num_agents):
                        healthIdx = agent * featureCount
                        shieldIdx = healthIdx + 4
                        total_relative_health += previous_state[eval_i][agent][healthIdx]
                        total_relative_shield += previous_state[eval_i][agent][shieldIdx]

                    if eval_infos[eval_i][0]['won']:
                        eval_env_infos['total_health_remaining'] = (total_relative_shield + total_relative_health) / (self.num_agents * 2)
                    else:
                        eval_env_infos['total_health_remaining'] = 0 


                    eval_episode_healths.append(eval_env_infos['total_health_remaining'])
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_healths = np.array(eval_episode_healths)
                eval_env_infos['eval_average_episode_rewards'] = eval_episode_rewards
                eval_env_infos['total_health_remaining'] = eval_episode_healths
        
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                # if eval_win_rate >= 0.6 and eval_win_rate > self.max_win_rate:
                #     print('saving a replay ==================', eval_win_rate)
                #     self.eval_envs.envs[0].save_replay()
                #     self.max_win_rate = eval_win_rate

                self.avg_win_rate += eval_win_rate
                self.avg_total_health_remaining += eval_episode_healths.mean()
                if self.use_wandb:
                    wandb.log({"eval_win_rate_"+map_name+"_player_"+str(self.arena_curr_player): eval_win_rate}, step=total_num_steps)
                    # wandb.log({"total_health_remaining_"+map_name: eval_episode_healths.mean()}, step=total_num_steps)
                    # wandb.log({"eval_enemy_win_rate": self.eval_enemy_win_rate[0]}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_healths = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_env_infos = {'total_health_remaining': 0, 'eval_average_episode_rewards': 0}

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            previous_state = eval_share_obs
            eval_obs, enemy_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions, eval_available_enemy_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            # Get relative health and shield values for units, this will only work with protoss?
            featureCount = 22

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1

                    total_relative_health = 0
                    total_relative_shield = 0
                    for agent in range(self.num_agents):
                        healthIdx = agent * featureCount
                        shieldIdx = healthIdx + 4
                        total_relative_health += previous_state[eval_i][agent][healthIdx]
                        total_relative_shield += previous_state[eval_i][agent][shieldIdx]

                    if eval_infos[eval_i][0]['won']:
                        eval_env_infos['total_health_remaining'] = (total_relative_shield + total_relative_health) / (self.num_agents * 2)
                    else:
                        eval_env_infos['total_health_remaining'] = 0 


                    eval_episode_healths.append(eval_env_infos['total_health_remaining'])
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_healths = np.array(eval_episode_healths)
                eval_env_infos['eval_average_episode_rewards'] = eval_episode_rewards
                eval_env_infos['total_health_remaining'] = eval_episode_healths
        
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if eval_win_rate >= 0.6 and eval_win_rate > self.max_win_rate:
                    print('saving a replay ==================', eval_win_rate)
                    self.eval_envs.envs[0].save_replay()
                    self.max_win_rate = eval_win_rate

                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                    wandb.log({"total_health_remaining": eval_episode_healths.mean()}, step=total_num_steps)
                    # wandb.log({"eval_enemy_win_rate": self.eval_enemy_win_rate[0]}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
    
    @torch.no_grad()
    def eval_robustness(self):
        for i in range(self.enemy_models):
            if(i+1 != self.enemy_models):
                self.restore_enemy_policy(i+1)
                self.enemy_trainer = TrainAlgo(self.all_args, self.enemy_policy, self.jump_start_policy, device = self.device)
                self.eval_two_player(1)
                print('================= enemy model ==============', i+1)
                print('================= total accuracy ==============', self.total_accuracy)
            else:
                self.enemy_trainer = self.trainer
                self.eval_enemy()
                print('================= enemy model ==============', i+1)
                print('================= total accuracy ==============', self.total_accuracy)


    @torch.no_grad()
    def eval_two_player(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_healths = []

        eval_obs, eval_agent_state, eval_available_actions = self.envs.reset()

        opp_avail_actions = np.ones((self.n_rollout_threads, self.num_enemies, 6+self.num_agents), dtype=np.float32)
        opp_obs = np.zeros((self.n_rollout_threads, self.num_enemies, self.input_layer_size), dtype=np.float32)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        opp_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        opp_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_env_infos = {'total_health_remaining': 0, 'eval_average_episode_rewards': 0}

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            opp_actions, enemy_rnn_states = self.collect_enemy(opp_obs,opp_rnn_states,opp_masks,opp_avail_actions)
            
            # Obser reward and next obs
            previous_state = eval_agent_state
            eval_obs, enemy_obs, eval_agent_state, eval_rewards, eval_dones, eval_infos, eval_available_actions, available_enemy_actions = self.envs.step(eval_actions, opp_actions)
            opp_avail_actions = available_enemy_actions.copy()
            opp_rnn_states = enemy_rnn_states.copy()
            opp_obs = enemy_obs.copy()
            
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            # Get relative health and shield values for units, this will only work with protoss?
            featureCount = 22

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1

                    total_relative_health = 0
                    total_relative_shield = 0
                    for agent in range(self.num_agents):
                        healthIdx = agent * featureCount
                        shieldIdx = healthIdx + 4
                        total_relative_health += previous_state[eval_i][agent][healthIdx]
                        total_relative_shield += previous_state[eval_i][agent][shieldIdx]

                    if eval_infos[eval_i][0]['won']:
                        eval_env_infos['total_health_remaining'] = (total_relative_shield + total_relative_health) / (self.num_agents * 2)
                    else:
                        eval_env_infos['total_health_remaining'] = 0 


                    eval_episode_healths.append(eval_env_infos['total_health_remaining'])
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_healths = np.array(eval_episode_healths)
                eval_env_infos['eval_average_episode_rewards'] = eval_episode_rewards
                eval_env_infos['total_health_remaining'] = eval_episode_healths
        
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                self.total_accuracy+=eval_win_rate
                print("eval win rate is {}.".format(eval_win_rate))

                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                    wandb.log({"total_health_remaining": eval_episode_healths.mean()}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break

    @torch.no_grad()
    def eval_enemy(self):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_healths = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_env_infos = {'total_health_remaining': 0, 'eval_average_episode_rewards': 0}

        while True:
            self.enemy_trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.enemy_trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            previous_state = eval_share_obs
            eval_obs, enemy_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions, eval_available_enemy_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            # Get relative health and shield values for units, this will only work with protoss?
            featureCount = 22

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1

                    total_relative_health = 0
                    total_relative_shield = 0
                    for agent in range(self.num_agents):
                        healthIdx = agent * featureCount
                        shieldIdx = healthIdx + 4
                        total_relative_health += previous_state[eval_i][agent][healthIdx]
                        total_relative_shield += previous_state[eval_i][agent][shieldIdx]

                    if eval_infos[eval_i][0]['won']:
                        eval_env_infos['total_health_remaining'] = (total_relative_shield + total_relative_health) / (self.num_agents * 2)
                    else:
                        eval_env_infos['total_health_remaining'] = 0 


                    eval_episode_healths.append(eval_env_infos['total_health_remaining'])
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_healths = np.array(eval_episode_healths)
                eval_env_infos['eval_average_episode_rewards'] = eval_episode_rewards
                eval_env_infos['total_health_remaining'] = eval_episode_healths
        
                # self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                self.total_accuracy+=eval_win_rate
                print("eval enemy win rate is {}.".format(eval_win_rate))
                # if self.eval_eney_win_rate == None: 
                #     self.eval_enemy_win_rate = [eval_win_rate]
                # else:
                #     self.eval_enemy_win_rate.append(eval_win_rate)

                # if self.use_wandb:
                #     wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                #     wandb.log({"total_health_remaining": eval_episode_healths.mean()}, step=total_num_steps)
                # else:
                #     self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
