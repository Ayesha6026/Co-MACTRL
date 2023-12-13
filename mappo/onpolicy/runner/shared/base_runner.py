import wandb
import os
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.static_envs = config['static_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.num_enemies = config['num_enemies']
        self.num_agents_arena = config['num_agents_arena']
        self.num_enemies_arena = config['num_enemies_arena']
        self.arena_envs = config['arena_envs']
        self.arena_eval_envs = config['arena_eval_envs']
        self.enemy_envs = config['enemy_envs']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.multi_player = self.all_args.multi_player
        self.arena = self.all_args.arena

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.jump_start_policy = None
        self.jump_start_policy_pool = [None, None, None, None, None]

        # dir
        self.model_dir = self.all_args.model_dir
        self.jump_start_model_dir = self.all_args.jump_start_model_dir
        self.jump_start_model_pool_dir = self.all_args.jump_start_model_pool_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device,
                            policy_identifier="Exploration Policy")
        
        if self.multi_player:
            if self.arena:
                share_observation_space = self.enemy_envs.share_observation_space[0] if self.use_centralized_V else self.enemy_envs.observation_space[0]
                self.enemy_policy = Policy(self.all_args,
                            self.enemy_envs.observation_space[0],
                            share_observation_space,
                            self.enemy_envs.enemy_action_space[0],
                            device = self.device,
                            policy_identifier="Enemy Policy")
            else:
                self.enemy_policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.enemy_action_space[0],
                                device = self.device,
                                policy_identifier="Enemy Policy")
            
            # self.restore_enemy_policy()
            self.enemy_trainer = TrainAlgo(self.all_args, self.enemy_policy, self.jump_start_policy, device = self.device)
        

        if self.model_dir is not None:
            self.restore()
            self.trainer = TrainAlgo(self.all_args, self.policy, self.jump_start_policy, device = self.device)

        if self.jump_start_model_dir is not None:
            self.jump_start_policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device = self.device,
                                policy_identifier="Jump Start")

            self.restore_jump_start()

            self.trainer = TrainAlgo(self.all_args, self.policy, self.jump_start_policy, device = self.device)

        if self.model_dir is None and self.jump_start_model_dir is None:
            self.trainer = TrainAlgo(self.all_args, self.policy, self.jump_start_policy, device = self.device)

        if self.jump_start_model_pool_dir is not None:
            for i in range(0, 5):
                self.jump_start_policy_pool[i] = Policy(self.all_args,
                                    self.envs.observation_space[0],
                                    share_observation_space,
                                    self.envs.action_space[0],
                                    device = self.device,
                                    policy_identifier=f"Jump Start pool {i}")

            self.restore_jump_start_pool()

            self.trainer = TrainAlgo(self.all_args, self.policy, self.jump_start_policy_pool[0], device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.num_enemies,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0]) 

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    @torch.no_grad()
    def compute_guide(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.jump_start_policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def save_curr_arena_player(self, curr):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        save_dir = './results/'+str(self.all_args.experiment_name)
        if not Path(save_dir).exists():
            os.makedirs(save_dir)
        torch.save(policy_actor.state_dict(), './results/'+str(self.all_args.experiment_name)+'/actor_nn'+str(curr)+'.pt')

    def restore(self):
        """Restore policy's networks from a saved model."""
        # cur_dir = os.getcwd()
        # policy_actor_state_dict = torch.load(str(cur_dir) + '\\actor.pt', map_location=torch.device('cpu'))
        #policy_actor_state_dict = torch.load(str(self.model_dir) + '\\actor.pt', map_location=torch.device('cpu'))

        policy_actor_state_dict = torch.load('/home/ayesha/projects/researchSpring2022/mappo/onpolicy/scripts/results/StarCraft2/'+str(self.all_args.map_name)+'/mappo/'+str(self.all_args.experiment_name)+'/wandb/latest-run/files/actor.pt')

        # policy_actor_state_dict = torch.load('/home/ayesha/projects/TrainedModels/3s5z/singlePlayer/actor8.pt', map_location=torch.device('cpu'))
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        print('======================================= Model restored',)
        # if not self.all_args.use_render:
        #     #policy_critic_state_dict = torch.load(str(self.model_dir) + '\\critic.pt', map_location=torch.device('cpu'))
        #     #policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
        #     policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', map_location=torch.device('cpu'))
        #     self.policy.critic.load_state_dict(policy_critic_state_dict)
    def restore_player(self,model):

        policy_actor_state_dict = torch.load('./results/'+str(self.all_args.experiment_name)+'/actor_nn'+str(model)+'.pt')
        # policy_actor_state_dict = torch.load('/home/ayesha/projects/TrainedModels/3s5z/singlePlayer/actor'+str(model)+'.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        print('======================================= Model restored',)
    
    def restore_enemy_policy(self,model = 1):
        """Restore enemy policy's networks from a saved model."""
        # cur_dir = os.getcwd()
        # policy_actor_state_dict = torch.load(str(cur_dir) + '\\actor'+str(model)+'.pt', map_location=torch.device('cpu'))

        # policy_actor_state_dict = torch.load('/home/ayesha/projects/researchSpring2022/2s3z/arena/actor'+str(model)+'.pt')
        policy_actor_state_dict = torch.load('./results/'+str(self.all_args.experiment_name)+'/actor_nn'+str(model)+'.pt')
        self.enemy_policy.actor.load_state_dict(policy_actor_state_dict)
        print('======================================= Enemy Model restored',)
        # if not self.all_args.use_render:
        #     #policy_critic_state_dict = torch.load(str(self.model_dir) + '\\critic.pt', map_location=torch.device('cpu'))
        #     #policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
        #     policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', map_location=torch.device('cpu'))
        #     self.policy.critic.load_state_dict(policy_critic_state_dict)

    def restore_jump_start(self):
        """Restore the jump_start policy's networks from a saved model."""
        # cur_dir = os.getcwd()
        print('=======================================',self.jump_start_model_dir)
        #policy_actor_state_dict = torch.load(str(self.jump_start_model_dir) + '\\actor.pt', map_location=torch.device('cpu'))
        policy_actor_state_dict = torch.load(str(self.jump_start_model_dir) + '/actor.pt')
        #policy_actor_state_dict = torch.load(str(self.jump_start_model_dir) + '/actor.pt', map_location=torch.device('cpu'))
        # policy_actor_state_dict = torch.load(str(cur_dir) + '\\actor.pt', map_location=torch.device('cpu'))
        self.jump_start_policy.actor.load_state_dict(policy_actor_state_dict)
        # if not self.all_args.use_render:
            #policy_critic_state_dict = torch.load(str(self.jump_start_model_dir) + '\\critic.pt', map_location=torch.device('cpu'))
            #policy_critic_state_dict = torch.load(str(self.jump_start_model_dir) + '/critic.pt')
            # policy_critic_state_dict = torch.load(str(self.jump_start_model_dir) + '/critic.pt', map_location=torch.device('cpu'))
            # policy_critic_state_dict = torch.load(str(cur_dir) + '\\critic.pt', map_location=torch.device('cpu'))
            # self.jump_start_policy.critic.load_state_dict(policy_critic_state_dict)

    def restore_jump_start_pool(self):
        """Restore the jump_start policy's networks from a saved model."""
        #policy_actor_state_dict = torch.load(str(self.jump_start_model_dir) + '\\actor.pt', map_location=torch.device('cpu'))
        #policy_actor_state_dict = torch.load(str(self.jump_start_model_dir) + '/actor.pt')
        policy_actor_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/actor.pt', map_location=torch.device('cpu'))
        self.jump_start_policy_pool[0].actor.load_state_dict(policy_actor_state_dict)

        policy_actor_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/actor_1.pt', map_location=torch.device('cpu'))
        self.jump_start_policy_pool[1].actor.load_state_dict(policy_actor_state_dict)

        policy_actor_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/actor_2.pt', map_location=torch.device('cpu'))
        self.jump_start_policy_pool[2].actor.load_state_dict(policy_actor_state_dict)

        policy_actor_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/actor_3.pt', map_location=torch.device('cpu'))
        self.jump_start_policy_pool[3].actor.load_state_dict(policy_actor_state_dict)

        policy_actor_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/actor_4.pt', map_location=torch.device('cpu'))
        self.jump_start_policy_pool[4].actor.load_state_dict(policy_actor_state_dict)

        if not self.all_args.use_render:
            #policy_critic_state_dict = torch.load(str(self.jump_start_model_dir) + '\\critic.pt', map_location=torch.device('cpu'))
            #policy_critic_state_dict = torch.load(str(self.jump_start_model_dir) + '/critic.pt')
            policy_critic_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/critic.pt', map_location=torch.device('cpu'))
            self.jump_start_policy_pool[0].critic.load_state_dict(policy_critic_state_dict)

            policy_critic_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/critic_1.pt', map_location=torch.device('cpu'))
            self.jump_start_policy_pool[1].critic.load_state_dict(policy_critic_state_dict)

            policy_critic_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/critic_2.pt', map_location=torch.device('cpu'))
            self.jump_start_policy_pool[2].critic.load_state_dict(policy_critic_state_dict)

            policy_critic_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/critic_3.pt', map_location=torch.device('cpu'))
            self.jump_start_policy_pool[3].critic.load_state_dict(policy_critic_state_dict)

            policy_critic_state_dict = torch.load(str(self.jump_start_model_pool_dir) + '/critic_4.pt', map_location=torch.device('cpu'))
            self.jump_start_policy_pool[4].critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
