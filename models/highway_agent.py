import torch
import pytorch_lightning as pl
from stable_baselines3 import DQN
import gymnasium as gym

import highway_env
# highway_env.register_highway_envs()

class HighwayDQNModel(pl.LightningModule):
    def __init__(self, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__()
        self.setup_model(**kwargs)

    def setup_model(self, ckpt, fix_model=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Options when loading rl_zoo3 models
        # algo = 'ppo'
        seed = 0

        env = gym.make(
            "highway-fast-v0",
            render_mode='rgb_array',
            config={
                "observation": {
                    "type": "GrayscaleObservation",
                    "observation_shape": (128, 64),
                    "stack_size": 4,
                    "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                    "scaling": 1.75,
                },
            },
        )
        
        self.model = DQN.load(ckpt, env=env)

        self.model.policy.to(device)

        if fix_model:
            self.model.policy.set_training_mode(False)
            self.model.policy.eval()
            for param in self.model.policy.parameters():
                param.requires_grad = False

    # (batch, 4, 84, 84)
    def forward(self, x):
        # action, _states = self.model.predict(x, deterministic=True)
        return self.model.policy.q_net.q_net(
            self.model.policy.q_net.extract_features(
                x, self.model.policy.q_net.features_extractor
        ))

def main():
    x = torch.randn((1, 4, 128, 64))
    module = HighwayDQNModel(ckpt="/home/rzuo02/jobs/highway_env/cnn_dqn/model.zip")
    print(module(x))

if __name__ == "__main__":
    main()
