import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from torchvision import models

from utils.helper import get_targets_from_annotations
from utils.metrics import SingleLabelMetrics
from models.PPO import PPO
from pathlib import Path

class PPOPolicyModel(pl.LightningModule):
    def __init__(self, num_inputs, num_actions, saved_model_path, learning_rate=1e-5, fix_model=True):
        super().__init__()
        
        self.setup_model(num_inputs=num_inputs, num_actions=num_actions, fix_model=fix_model, saved_model_path=saved_model_path)
        self.setup_losses()
        self.setup_metrics(num_actions)
        self.learning_rate = learning_rate
    
    def custom_load_from_checkpoint(self, classifier_checkpoint):
        classifier_checkpoint = Path(classifier_checkpoint)
        print("[RUI] Load from checkpoint")
        for world_stage, model in self.models.items():
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(classifier_checkpoint / ('ppo_super_mario_bros_' + world_stage)))
                model.cuda()
            else:
                model.load_state_dict(torch.load(classifier_checkpoint / ('ppo_super_mario_bros_' + world_stage), map_location=lambda storage, loc: storage))
    
    def setup_model(self, num_inputs, num_actions, fix_model, saved_model_path):
        files = list(Path(saved_model_path).rglob("*"))
        self.models = {}
        for f in files:
            name = str(f)[-3:] 
            model = PPO(num_inputs=num_inputs, num_actions=num_actions) 
            self.models[name] = model

        if fix_model:
            for _, model in self.models.items():
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
    
    def setup_losses(self):
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def setup_metrics(self, num_actions):
        self.train_metrics = SingleLabelMetrics(num_classes=num_actions)
        self.valid_metrics = SingleLabelMetrics(num_classes=num_actions)
        self.test_metrics = SingleLabelMetrics(num_classes=num_actions)

    def forward(self, xs, lables):
        logits = []
        values = []
        for i in range(len(xs)):
            x = xs[i].unsqueeze(0)
            world_stage_idx = lables[i]['world_stage']
            logit, value = self.models[world_stage_idx](x)
            logits.append(logit)
            values.append(value)
        logits = torch.cat(logits, dim=0)
        values = torch.cat(values, dim=0)
        return logits, values 
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, value = self(x, y)
        targets = get_targets_from_annotations(y, dataset="MARIO")

        loss = self.classification_loss_fn(logits, targets)

        self.log('train_loss', loss)
        self.train_metrics(logits, targets)

        return loss
    
    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x, y)
        targets = get_targets_from_annotations(y, dataset="MARIO")

        loss = self.classification_loss_fn(logits, targets)

        self.log('val_loss', loss)
        self.val_metrics(logits, targets)

    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y)
        targets = get_targets_from_annotations(y, dataset="MARIO")

        loss = self.classification_loss_fn(logits, targets)

        self.log('test_loss', loss)
        self.test_metrics(logits, targets)
    
    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.save(model="classifier", classifier_type="PPO", dataset="MARIO")
        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
