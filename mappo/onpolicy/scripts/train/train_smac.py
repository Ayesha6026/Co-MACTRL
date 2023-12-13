#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.starcraft2.StarCraft2_Env_mplayer import StarCraft2EnvMPlayer
from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from onpolicy.envs.starcraft2.smac_maps import get_map_params
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareDummyVecEnvMPlayer

"""Train script for SMAC."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2EnvMPlayer(all_args) if all_args.multi_player else StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnvMPlayer([get_env_fn(0)]) if all_args.multi_player else ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument('--eval_map_name', type=str, default='3m',
                        help="Which smac map to eval")
    parser.add_argument('--arena_maps', type=str, default='3m',
                        help="Which smac maps to choose, comma separated")
    parser.add_argument('--group_name', type=str, default='test_group',
                        help="Which wandb group to choose")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--use_influence_map", action='store_true', default=False)
    parser.add_argument("--use_influence_map_critic", action='store_true', default=False)
    parser.add_argument('--normalize_influence_map', action='store_true', default=False)
    parser.add_argument('--reward_only_positive', action='store_true', default=False)
    parser.add_argument('--reward_win', type=int, default=200)
    parser.add_argument('--reward_speed', action='store_true', default=False)
    parser.add_argument('--reward_death_value', type=int, default=10)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.group_name,
                        #  group="test",
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    # all_args.eval_map_name = '3m'
    envs = make_train_env(all_args)
    # all_args.eval_map_name = '8m'
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    static_envs = make_eval_env(all_args) if all_args.multi_player else None
    # static_envs = None
    num_agents = get_map_params(all_args.map_name)["n_agents"]
    num_enemies = get_map_params(all_args.map_name)["n_enemies"]
    arena_envs = None
    num_agents_arena = None
    num_enemies_arena = None
    enemy_envs = None

    if all_args.arena:
        print('arena maps ======', all_args.arena_maps)
        # maps = all_args.arena_maps.split(',')
        maps = ['3m','8m','2s3z','3s5z']
        print('maps ======', maps)
        arena_envs = []
        arena_eval_envs = []
        num_agents_arena = []
        num_enemies_arena = []
        for map in maps:
            all_args.eval_map_name = map
            agents = get_map_params(map)["n_agents"]
            num_agents_arena.append(agents)
            enemies = get_map_params(map)["n_enemies"]
            num_enemies_arena.append(enemies)
            arena_envs.append(make_eval_env(all_args))
            arena_eval_envs.append(make_eval_env(all_args))
    
    if all_args.arena and all_args.multi_player:
        all_args.map_name = '3s5z_mp'
        enemy_envs = make_train_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "arena_envs": arena_envs,
        "arena_eval_envs": arena_eval_envs,
        "enemy_envs": enemy_envs,
        "eval_envs": eval_envs,
        "static_envs": static_envs,
        "num_agents": num_agents,
        "num_enemies": num_enemies,
        "num_agents_arena": num_agents_arena,
        "num_enemies_arena": num_enemies_arena,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.smac_runner import SMACRunner as Runner
    else:
        from onpolicy.runner.separated.smac_runner import SMACRunner as Runner

    runner = Runner(config)
    if all_args.arena:
        # runner.eval_in_multi_map(1)
        runner.run_arena()
    else:
        runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
