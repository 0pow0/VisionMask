import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch.utils.data.dataset import IterableDataset

class ReplayBuffer(object):
    
    def __init__(self, size):
        self.__storage = []
        self.__maxsize = size
        self.__next_idx = 0

    def __len__(self):
        return len(self.__storage)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, np.float32(reward), next_state, done)

        if self.__next_idx >= len(self.__storage):
            self.__storage.append(data)
        else:
            self.__storage[self.__next_idx] = data
        self.__next_idx = (self.__next_idx + 1) % self.__maxsize

    def sample(self):
        idx = random.randint(0, len(self.__storage) - 1)
        return self.__storage[idx]

class Agent(ABC):

    @abstractmethod    
    def get_action(self, game_state):
        pass
    
    def init(self, game):
        pass

    def reset(self):
        pass

class ConvNetwork(nn.Module):

    def __init__(self, screen_size, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),

            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),

            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),

            nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(),


            nn.Flatten(),
        )

    def forward(self, image):
        return self.net(image)


class DenseNetwork(nn.Module):

    def __init__(self, n_actions, n_variables):
        super().__init__()

        self.fc1 = nn.Linear(1536 + n_variables, 256)
        self.fc2 = nn.Linear(256 + n_variables, 64)
        self.fc3 = nn.Linear(64 + n_variables, n_actions)

    def forward(self, screen, variables):
        out = F.relu(self.fc1(torch.cat([screen, variables], dim=1)))
        out = F.relu(self.fc2(torch.cat([out, variables], dim=1)))
        out = F.relu(self.fc3(torch.cat([out, variables], dim=1)))
        return out

class DQNNetwork(nn.Module):

    def __init__(self, n_actions, screen_size, n_variables):
        super().__init__()

        self.screen_net = self.screen_net = ConvNetwork(screen_size, [5, 64, 128, 256, 512, 1024])
        self.neck_net = DenseNetwork(n_actions, n_variables)

    def forward(self, data):
        # summary(self.screen_net, input_data=data['screen'], col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        screen_out = self.screen_net(data['screen'])
        # summary(self.neck_net, input_data=(screen_out, data['variables']), col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        neck_out = self.neck_net(screen_out, data['variables'])
        return neck_out

    def forward_state(self, state, device=None):
        screens = torch.tensor(state['screen'], device=device, dtype=torch.float32).unsqueeze(0)
        variables = torch.tensor(state['variables'], device=device, dtype=torch.float32).unsqueeze(0)
        data = {'screen': screens, 'variables': variables}
        return self.forward(data)[0]
    
class ReplayDataset(IterableDataset):
    
    def __init__(self, replay, game):
        self.replay = replay
        self.game = game

    def __iter__(self):
        self.game.new_episode()
        while not self.game.is_episode_finished():
            state, action, reward, next_state, done = self.replay.sample()

            state = {
                'screen': state['screen'].astype(np.float32) / 255,
                'variables': state['variables']
            }

            next_state = {
                'screen': next_state['screen'].astype(np.float32) / 255,
                'variables': next_state['variables']
            }

            yield state, action, reward, next_state, done

class DQNAgent(LightningModule, Agent):

    def __init__(
            self,
            n_actions,
            screen_size,
            n_variables,
            lr=0.001,
            batch_size=32,
            frames_skip=1,
            epsilon=0.5,
            gamma=0.99,
            buffer_size=50_000,
            populate_steps=1_000,
            actions_per_step=10,
            validation_interval=10,
            weights_update_interval=1_000,
            epsilon_update_interval=200,
            epsilon_decay=0.99,
            epsilon_min=0.02,
            replay_update_skip=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.target_model = DQNNetwork(self.hparams.n_actions, self.hparams.screen_size, self.hparams.n_variables)
        self.model = DQNNetwork(self.hparams.n_actions, self.hparams.screen_size, self.hparams.n_variables)

        self.target_model.eval()
        self.model.train()
        self.__update_weights()

        self.buffer = ReplayBuffer(self.hparams.buffer_size)
        self.dataset = None

        self.env = None
        self.train_metrics = {}
        self.val_metrics = {}
    
    def set_train_environment(self, env):
        self.env = env
        self.dataset = ReplayDataset(self.buffer, self.env)

    def get_action(self, state, epsilon=None):
        epsilon = epsilon if epsilon is not None else self.hparams.epsilon
        if random.random() < epsilon:
            return self.__get_random_action()
        else:
            return self.__get_best_action(state)

    def __get_random_action(self):
        return random.randint(0, self.hparams.n_actions - 1)

    def __get_best_action(self, state):
        with torch.no_grad():
            qvalues = self.model.forward_state(state, self.device)
        return torch.argmax(qvalues).item()

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.hparams.lr, amsgrad=True)

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def on_fit_start(self):
        self.env.init()
        self.__populate_buffer()

    def on_fit_end(self):
        self.env.close()

    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.validation_interval == 0:
            self.__validate()

    def on_train_batch_start(self, batch, batch_idx):
        for i in range(self.hparams.actions_per_step):
            should_update_replay = (self.global_step+i) % self.hparams.replay_update_skip == 0
            done = self.__play_step(should_update_replay)
            if done:
                self.train_metrics = self.env.get_metrics(prefix='train_')
                break

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % self.hparams.weights_update_interval == 0:
            self.__update_weights()

        if self.global_step % self.hparams.epsilon_update_interval == 0:
            self.hparams.epsilon = max(self.hparams.epsilon * self.hparams.epsilon_decay, self.hparams.epsilon_min)

    def training_step(self, batch, batch_no):
        loss = self.__calculate_loss(batch)

        self.log('train_loss', loss),
        self.log('train_epsilon', self.hparams.epsilon),
        self.log('train_buffer_size', float(len(self.buffer)))
        self.log_dict(self.train_metrics)
        self.log_dict(self.val_metrics)
        return loss

    def __play_step(self, update_buffer=True):
        state = self.env.get_state()
        action = self.get_action(state)

        try:
            self.env.make_action(action, skip=self.hparams.frames_skip)
        except Exception:
            raise KeyboardInterrupt

        done = self.env.is_episode_finished()

        if update_buffer:
            reward = self.env.get_last_reward()
            next_state = self.env.get_state()
            self.buffer.add(state, action, reward, next_state, done)

        return done

    def __calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        state_action_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_model(next_states).max(1)[0]
            next_state_values[dones] = 0.0
        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def __update_weights(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def __populate_buffer(self):
        while len(self.buffer) < self.hparams.populate_steps:
            done = self.__play_step()
            if done:
                self.env.new_episode()

    def __validate(self):
        self.env.new_episode()
        while not self.env.is_episode_finished():
            state = self.env.get_state()
            action = self.get_action(state, epsilon=0)
            self.env.make_action(action)
        self.val_metrics = self.env.get_metrics(prefix='val_')
        self.env.new_episode()

class DoomDQNModel(LightningModule):
    def __init__(self, agent_ckpt, fix_model=True) -> None:
        super().__init__()
        self.setup_model(agent_ckpt, fix_model)

    def setup_model(self, agent_ckpt, fix_model=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = DQNAgent.load_from_checkpoint(agent_ckpt)
        self.model.to(device)

        if fix_model:
            # self.set_training_mode(False)
            self.model.eval()
            self.model.target_model.eval()
            self.model.model.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model.model.forward(x)
