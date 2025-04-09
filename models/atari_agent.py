import torch
import sys
import os
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from torchvision import models
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1].absolute()))

from utils.helper import get_targets_from_annotations
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics
from models.PPO import PPO

from rl_zoo3 import ALGOS, get_saved_hyperparams, create_test_env
from rl_zoo3.utils import get_model_path
from rl_zoo3.load_from_hub import download_from_hub
from huggingface_sb3 import EnvironmentName
import yaml

from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel

def get_rl_baselines3_model_path(algo, env_name, folder_trained_agents):
    # Experiment ID (default: 0: latest, -1: no exp folder)"
    exp_id = 0
    folder = folder_trained_agents
    try:
        _, model_path, log_path = get_model_path(
            exp_id,
            folder,
            algo,
            env_name,
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
                exp_id=exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                exp_id,
                folder,
                algo,
                env_name,
            )

    print(f"Loading {model_path}")
    return model_path, log_path

def get_rl_baselines3_model(algo:str, environment_name:str, seed:int, device, folder_trained_agents:str=None):

    # Options when loading rl_zoo3 models
    env_name: EnvironmentName = EnvironmentName(environment_name)

    # adopted from rl_zoo3
    model_path, log_path = get_rl_baselines3_model_path(
        algo=algo,
        env_name=env_name,
        folder_trained_agents=folder_trained_agents
    )

    # adopted from rl_zoo3
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        
    # adopted from rl_zoo3
    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(
        stats_path,
        norm_reward=False,
        test_mode=True
    )

    # adopted from rl_zoo3
    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
    kwargs = dict(seed=seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)
        
    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    
    env = create_test_env(
        env_name.gym_id,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(
        model_path,
        custom_objects=custom_objects,
        device=device,
        **kwargs
    )
    return model

class AtariPPOModel(pl.LightningModule):
    def __init__(self, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__()
        self.setup_model(**kwargs)

    def setup_model(self, env_name, folder_trained_agents, algo='ppo', fix_model=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Options when loading rl_zoo3 models
        # algo = 'ppo'
        seed = 0
        
        self.model = get_rl_baselines3_model(
            algo=algo,
            environment_name=env_name,
            seed=seed,
            folder_trained_agents=folder_trained_agents,
            device=self.device
        )
        self.model.policy.to(device)

        if fix_model:
            self.model.policy.set_training_mode(False)
            self.model.policy.eval()
            for param in self.model.policy.parameters():
                param.requires_grad = False

    # (batch, 4, 84, 84)
    def forward(self, x):
        # preprocessed_obs = preprocess_obs(x, self.model.policy.observation_space, normalize_images=self.model.policy.normalize_images)
        # features = self.model.policy.pi_features_extractor(preprocess_obs)

        # print(type(self.model.policy), type(self.model.policy.extract_features))
        # print(super(type(self.model.policy), self.model.policy), super(type(self.model.policy), self.model.policy).extract_features)
        # print(super(ActorCriticPolicy, self.model.policy), super(ActorCriticPolicy, self.model.policy).extract_features)
        # print(super(BaseModel, self.model.policy), super(BaseModel, self.model.policy).extract_features)
        features = super(ActorCriticPolicy, self.model.policy).extract_features(x, self.model.policy.pi_features_extractor)
        latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
        logits = self.model.policy.action_net(latent_pi)
        # return self.model.policy.action_dist.proba_distribution(action_logits=logits).get_actions(deterministic=True)
        return logits

class AtariDQNModel(pl.LightningModule):
    def __init__(self, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__()
        self.setup_model(**kwargs)

    def setup_model(self, env_name, folder_trained_agents, algo='ppo', fix_model=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Options when loading rl_zoo3 models
        # algo = 'ppo'
        seed = 0
        
        self.model = get_rl_baselines3_model(
            algo=algo,
            environment_name=env_name,
            seed=seed,
            folder_trained_agents=folder_trained_agents,
            device=self.device
        )
        self.model.policy.to(device)

        if fix_model:
            self.model.policy.set_training_mode(False)
            self.model.policy.eval()
            for param in self.model.policy.parameters():
                param.requires_grad = False

    # (batch, 4, 84, 84)
    def forward(self, x):
        # action = self.model.policy.q_net(
            # x,  # type: ignore[arg-type]
            # deterministic=True,
        # )
        # print("self.model", self.model)
        # print("self.model.policy", self.model.policy)
        # print("self.model.policy.q_net", self.model.policy.q_net)
        # print("self.model.policy.q_net.q_net", self.model.policy.q_net.q_net)
        return self.model.policy.q_net.q_net(
            self.model.policy.q_net.extract_features(
                x, self.model.policy.q_net.features_extractor
        ))
 
def main():
    x = torch.randn((1, 4, 84, 84))
    module = AtariPPOModel(
        env_name= 'RoadRunner' + 'NoFrameskip-v4',
        folder_trained_agents='/home/rzuo02/work/rl-baselines3-zoo/rl-trained-agents'
    )
    print(module(x))

if __name__ == "__main__":
    main()