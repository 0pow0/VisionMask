import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from pathlib import Path
from models.PPO import PPO
import torchvision.transforms as T
from timeit import default_timer

from models.explainer import Deeplabv3Resnet50ExplainerModel, FCNResnet50ExplainerModel, UNet 
from models.agent import PPOPolicyModel
from models.atari_agent import AtariPPOModel, AtariDQNModel
from models.doom_agent import DoomDQNModel
from models.highway_agent import HighwayDQNModel
from utils.helper import get_targets_from_annotations, get_filename_from_annotations
from utils.image_utils import save_mask, save_masked_image, save_heatmap_mario, save_masked_image_mario, atari_save_heatmap, atari_save_masked_image, mario_save_mask, atari_save_mask, save_confusion_matrix, atari_save_pil_img
from utils.image_utils import highway_save_heatmap
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, MarioClassMaskAreaLoss
from utils.metrics import SingleLabelMetrics
from utils.transformers import rgb_to_grayscale, squeeze_rgb_gray_channel
from utils.mario_utils import MarioBackgroundAdapter, mario_seperate_content_info
from utils.atari_utils import atari_seperate_content_info, EnduroBackgroundAdapter, atari_concat_content_info, SeaquestBackgroundAdapter, MsPacmanBackgroundAdapter
from utils.doom_metrics import DoomDeletion, DoomInsertion
import numpy as np
from evaluation.lime_mario import MarioLimeImageExplainer, batch_predict
from utils.atari_metrics import AtariDeletion, AtariInsertion
from utils.mario_metrics import MarioDeletion, MarioInsertion
from utils.highway_utils import HighwayBackgroundAdapter
from utils.highway_metrics import HighwayDeletion, HighwayInsertion

from models.vit import create_segmenter
import torch.nn.functional as F
from utils.metrics import Metrics
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

class VisionMask(pl.LightningModule):
    def __init__(self, num_action=20, dataset="MARIO", agent_type="ppo", agent_ckpt=None, fix_classifier=True, learning_rate=1e-5, class_mask_min_area=0.05, 
                 mask_max_area=0.3, entropy_regularizer=1.0, use_mask_variation_loss=False, mask_variation_regularizer=1.0, use_mask_area_loss=True, 
                 mask_area_constraint_regularizer=1.0, mask_total_area_regularizer=0.1, ncmask_total_area_regularizer=0.3, metrics_threshold=-1.0,
                 save_masked_images=False, save_masks=False, save_all_class_masks=False, save_path="./results/",
                 mario_use_greyscale=False, deletion_fraction=0.5, insertion_fraction=0.5,
                 atari_env='BeamRider',
                 use_deletion_insertion=False, weight_decay=0.01, save_confusion_matrix=False, explainer_type='deeplab',
                 use_stadler_mask_area_loss=False, ref_value_mode="background"):

        super().__init__()

        self.dataset = dataset
        self.num_action = num_action
        self.agent_type = agent_type
        self.explainer_type = explainer_type
        self.use_deletion_insertion = use_deletion_insertion
        self.save_confusion_matrix = save_confusion_matrix 
        self.mask_max_area = mask_max_area
        self.use_stadler_mask_area_loss = use_stadler_mask_area_loss
        self.ref_value_mode = ref_value_mode

        self.weight_decay = weight_decay
        self.train_time = []

        # Atari settings
        if self.dataset == 'ATARI':
            self.atari_env = atari_env

        self.setup_explainer(num_classes=num_action, mario_use_greyscale=mario_use_greyscale)
        self.setup_agent(agent_type=agent_type, agent_ckpt=agent_ckpt, fix_classifier=fix_classifier)

        self.setup_losses(dataset=dataset, class_mask_min_area=class_mask_min_area, class_mask_max_area=mask_max_area)
        self.setup_metrics(num_classes=num_action, metrics_threshold=metrics_threshold)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.entropy_regularizer = entropy_regularizer
        self.use_mask_variation_loss = use_mask_variation_loss
        self.mask_variation_regularizer = mask_variation_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.mask_total_area_regularizer = mask_total_area_regularizer
        self.ncmask_total_area_regularizer = ncmask_total_area_regularizer

        # Image display/save settings
        self.save_masked_images = save_masked_images
        self.save_masks = save_masks
        self.save_all_class_masks = save_all_class_masks
        self.save_path = save_path
        
        if self.use_deletion_insertion:
            self.deletion_fraction = deletion_fraction
            self.insertion_fraction = insertion_fraction
            self.deletion_auc = []
            self.insertion_auc = []

        # Mario settings
        if self.dataset == 'MARIO':
            self.mario_use_greyscale = mario_use_greyscale
            self.background_adapter = MarioBackgroundAdapter(mode=self.ref_value_mode)

            if self.use_deletion_insertion:
                self.deletion = MarioDeletion()
                self.insertion = MarioInsertion()
        elif self.dataset == 'ATARI':
            if self.use_deletion_insertion:
                self.deletion = AtariDeletion(atari_env=self.atari_env)
                self.insertion = AtariInsertion(atari_env=self.atari_env)

            if self.atari_env == 'Enduro':
                self.background_adapter = EnduroBackgroundAdapter()
            elif self.atari_env == 'Seaquest':
                self.background_adapter = SeaquestBackgroundAdapter()
            elif self.atari_env == 'MsPacman':
                self.background_adapter = MsPacmanBackgroundAdapter()
        elif self.dataset == "DOOM":
            if self.use_deletion_insertion:
                self.deletion = DoomDeletion()
                self.insertion = DoomInsertion()
        elif self.dataset == "HIGHWAY":
            if self.use_deletion_insertion:
                self.deletion = HighwayDeletion()
                self.insertion = HighwayInsertion()
            self.background_adapter = HighwayBackgroundAdapter()

        if self.save_confusion_matrix:
            self.confusion_matrix = torch.zeros((num_action, num_action), dtype=torch.int)
        
        self.total_time = 0.0
        
    def setup_explainer(self, num_classes, mario_use_greyscale):
        if self.explainer_type == "deeplab":
            # self.explainer = SimpleMarioExplainer(num_classes=num_classes)
            self.explainer = Deeplabv3Resnet50ExplainerModel(num_classes=num_classes)
            # a = {'image_size': (512, 512), 'patch_size': 16, 'd_model': 192, 'n_heads': 3, 'n_layers': 12, 'normalization': 'vit', 'distilled': False, 'backbone': 'vit_tiny_patch16_384', 'dropout': 0.0, 'drop_path_rate': 0.1, 'decoder': {'name': 'linear'}, 'n_cls': 150}
            # a["n_cls"] = num_classes
            # self.explainer = create_segmenter(a)
            # If use greyscale as input change first convolutional layer to take 1 channel as input
            if mario_use_greyscale:
                self.explainer.explainer.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if self.dataset in ["DOOM", "HIGHWAY"]:
                self.explainer.explainer.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif self.explainer_type == "fcn":
            self.explainer = FCNResnet50ExplainerModel(num_classes=num_classes) 
        elif self.explainer_type == "unet":
            self.explainer = UNet(n_channels=3, n_classes=num_classes)

    def setup_agent(self, agent_type, agent_ckpt, fix_classifier):
        if self.dataset == "MARIO":
            if agent_type == "ppo":
                mario_agent_ckpt = Path(agent_ckpt) / 'mario'
                self.agent = PPOPolicyModel(num_inputs=4, num_actions=7,
                                        saved_model_path=mario_agent_ckpt)
        elif self.dataset == 'ATARI':
            if agent_type == "ppo":
                atari_agent_ckpt = Path(agent_ckpt) / 'atari'
                print("RUI: ppo atari")
                self.agent = AtariPPOModel(env_name=self.atari_env + 'NoFrameskip-v4',
                                         folder_trained_agents=atari_agent_ckpt)
            elif agent_type == "dqn":
                print("RUI: dqn atari")
                self.agent = AtariDQNModel(env_name=self.atari_env + 'NoFrameskip-v4',
                                            folder_trained_agents=atari_agent_ckpt,
                                            algo=agent_type)
        elif self.dataset == "DOOM":
            print("RUI: dqn doom")
            doom_agent_ckpt = Path(agent_ckpt) / 'doom'
            self.agent = DoomDQNModel(agent_ckpt=doom_agent_ckpt)
        elif self.dataset == "HIGHWAY":
            print("RUI: dqn highway")
            highway_agent_ckpt = Path(agent_ckpt) / 'highway'
            self.agent = HighwayDQNModel(ckpt=highway_agent_ckpt)
        else:
            raise Exception("Unknown classifier type " + agent_type)
            
        if self.dataset == "MARIO":
            mario_agent_ckpt = Path(agent_ckpt) / 'mario'
            self.agent.custom_load_from_checkpoint(mario_agent_ckpt)

        if fix_classifier:
            if self.dataset == 'MARIO':
                pass
            else:
                self.agent.freeze()

    def setup_losses(self, dataset, class_mask_min_area, class_mask_max_area):
        self.total_variation_conv = TotalVariationConv()

        if dataset == "CUB":
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()

        if self.use_stadler_mask_area_loss:
            if dataset == "MARIO":
                self.class_mask_area_loss_fn = MarioClassMaskAreaLoss(image_size_h=209, image_size_w=256 * 4, min_area=class_mask_min_area, max_area=class_mask_max_area)
            elif dataset == "DOOM":
                self.class_mask_area_loss_fn = MarioClassMaskAreaLoss(image_size_h=60, image_size_w=40 * 5, min_area=class_mask_min_area, max_area=class_mask_max_area)
            elif dataset == "ATARI" and self.atari_env == 'Enduro':
                self.class_mask_area_loss_fn = MarioClassMaskAreaLoss(image_size_h=104, image_size_w=152 * 4, min_area=class_mask_min_area, max_area=class_mask_max_area)
            elif dataset == "ATARI" and self.atari_env == 'Seaquest':
                self.class_mask_area_loss_fn = MarioClassMaskAreaLoss(image_size_h=132, image_size_w=152 * 4, min_area=class_mask_min_area, max_area=class_mask_max_area)
            elif dataset == "ATARI" and self.atari_env == 'MsPacman':
                self.class_mask_area_loss_fn = MarioClassMaskAreaLoss(image_size_h=172, image_size_w=160 * 4, min_area=class_mask_min_area, max_area=class_mask_max_area)
            elif dataset == "HIGHWAY":
                self.class_mask_area_loss_fn = MarioClassMaskAreaLoss(image_size_h=128, image_size_w=64 * 4, min_area=class_mask_min_area, max_area=class_mask_max_area)
            else:
                self.class_mask_area_loss_fn = ClassMaskAreaLoss(min_area=class_mask_min_area, max_area=class_mask_max_area)

    def setup_metrics(self, num_classes, metrics_threshold):
        self.metrics = SingleLabelMetrics(self.num_action)
        self.ro_sp_metrics = Metrics()

    # ('screen': (B, 5, 60, 40), 'variables': (B, 10))
    def doom_forward(self, input):
        # (B, 7)
        var = input['variables']
        state = input['screen']
        shape = state.shape

        q_ori = self.agent(input)
        target_indices = q_ori.argmax(-1)
        targets = q_ori.new_zeros(target_indices.size(0), self.num_action)
        targets.scatter_(1, target_indices.unsqueeze(1), 1)

        # (B, 60, 40*5)
        state = torch.cat(state.unbind(dim=1), dim=2)
        segmentations = self.explainer(state.unsqueeze(1))
        # (B, n_action, 60, 200)
        mask = segmentations.sigmoid()
        inversed_mask = torch.ones_like(mask) - mask 

        background = torch.zeros_like(state)

        masked_image = mask * state.unsqueeze(1) + inversed_mask * background.unsqueeze(1)
        inversed_image = inversed_mask * state.unsqueeze(1) + mask * background.unsqueeze(1)
        variables = var.unsqueeze(1).repeat(1, self.num_action, 1)

        masked_image = torch.stack(masked_image.split(40, 3), 2)
        inversed_image = torch.stack(inversed_image.split(40, 3), 2)
        logits_mask = self.agent({'screen': masked_image.reshape(-1, 5, 60, 40), 'variables': variables.reshape(-1, 10)})
        logits_inversed_mask = self.agent({'screen': inversed_image.reshape(-1, 5, 60, 40), 'variables': variables.reshape(-1, 10)})

        logits_mask = logits_mask.reshape(shape[0], self.num_action, -1)
        logits_inversed_mask = logits_inversed_mask.reshape(shape[0], self.num_action, -1)

        # print("RUI Attention: ", logits.argmax(-1) == targets.argmax(-1), logits.argmax(-1), targets.argmax(-1))
        # print((logits.argmax(-1) == targets.argmax(-1)))
        # assert((logits.argmax(-1) == targets.argmax(-1)).all())

        return logits_mask, logits_inversed_mask, mask, segmentations, None, targets

    def atari_forward(self, image, targets, annotations):
        # (batch, 3, H, W * frame)
        shape = image.shape
        # (batch, 3, H_c, W_c * frame)
        backgrounds = self.background_adapter.get_content_background(image)
        # (batch, 3, H_c, W_c * frame), (batch, 3, H, W * frame)
        contents, info = atari_seperate_content_info(image, atari_env=self.atari_env) 
        # (batch, num_actions, H_c, W_c * frame)
        segmentations = self.explainer(contents / 255.)
        # (batch, n_actions, H_c, W_c * frame)
        mask = segmentations.sigmoid()
        inversed_mask = torch.ones_like(mask) - mask 
        # (batch, num_action, 3, H_c, W_c * frame)
        masked_contents = mask.unsqueeze(2) * contents.unsqueeze(1) + inversed_mask.unsqueeze(2) * backgrounds.unsqueeze(1)
        inversed_masked_contents = inversed_mask.unsqueeze(2) * contents.unsqueeze(1) + mask.unsqueeze(2) * backgrounds.unsqueeze(1)

        # (batch, 3, H, W * frame) 
        masked_image = [atari_concat_content_info(content=masked_contents[:, i], info=info, atari_env=self.atari_env) for i in range(self.num_action)]
        masked_image = torch.stack(masked_image, dim=1)
        inversed_masked_image = [atari_concat_content_info(content=inversed_masked_contents[:, i], info=info, atari_env=self.atari_env) for i in range(self.num_action)]
        inversed_masked_image = torch.stack(inversed_masked_image, dim=1)

        transformer = T.Compose([T.Lambda(rgb_to_grayscale), T.Lambda(squeeze_rgb_gray_channel), T.Resize(size=(84, 4*84))])
        # (batch, 84, frame * 84)
        masked_image = transformer(masked_image.reshape(-1, 3, shape[2], shape[3]))
        inversed_masked_image = transformer(inversed_masked_image.reshape(-1, 3, shape[2], shape[3]))
        # (batch, frame, 84, 84)
        masked_image = torch.stack(masked_image.split(84, 2), 1)
        inversed_masked_image = torch.stack(inversed_masked_image.split(84, 2), 1)
        logits_mask = self.agent(masked_image)
        logits_mask = logits_mask.reshape(shape[0], self.num_action, -1)
        logits_inversed_mask = self.agent(inversed_masked_image)
        logits_inversed_mask = logits_inversed_mask.reshape(shape[0], self.num_action, -1)

        return logits_mask, logits_inversed_mask, mask, segmentations, None, targets

    def forward(self, image, targets, annotations):
        if self.dataset == "ATARI":
            return self.atari_forward(image=image, targets=targets, annotations=annotations)
        elif self.dataset == "DOOM":
            return self.doom_forward(input=image)
        elif self.dataset == "HIGHWAY":
            return self.highway_forward(image=image, targets=targets, annotations=annotations)
        elif self.dataset == "MARIO":
            # (batch, 3, H, W * frame)
            shape = image.shape
            # (batch, 3, H - 31, W * frame)
            backgrounds = self.background_adapter.get_content_background(image, annotations)
            # (batch, 3, H - 31, W * frame), (batch, 3, 31, W * frame)
            contents, info = mario_seperate_content_info(image)
            if self.mario_use_greyscale:
                # (batch, num_actions, H - 31, W * frame)
                segmentations = self.explainer(T.Lambda(rgb_to_grayscale)(contents))
            else:
                # (batch, num_actions, H - 31, W * frame)
                # shape = contents.shape
                # contents = F.interpolate(contents, size=(512, 512), mode='bilinear', align_corners=False)
                segmentations = self.explainer(contents)
                # segmentations = F.interpolate(segmentations, size=(shape[-2], shape[-1]), mode='bilinear', align_corners=False)
                # contents = F.interpolate(contents, size=(shape[-2], shape[-1]), mode='bilinear', align_corners=False)
            # (batch, num_actions, H - 31, W * frame)
            masks = segmentations.sigmoid()
            inversed_masks = torch.ones_like(masks) - masks
            # (batch, num_actions, 3, H - 31, W * frame)
            masked_contents = masks.unsqueeze(2) * contents.unsqueeze(1) + inversed_masks.unsqueeze(2) * backgrounds.unsqueeze(1)
            inversed_masked_contents = inversed_masks.unsqueeze(2) * contents.unsqueeze(1) + masks.unsqueeze(2) * backgrounds.unsqueeze(1)

            # (batch, num_actions, 3, H, W * frame)
            masked_image = torch.cat((info.unsqueeze(1).expand(-1, self.num_action, -1, -1, -1), masked_contents), dim=3)
            inversed_masked_image = torch.cat((info.unsqueeze(1).expand(-1, self.num_action, -1, -1, -1), inversed_masked_contents), dim=3)
            transformer = T.Compose([T.Lambda(rgb_to_grayscale), T.Lambda(squeeze_rgb_gray_channel), T.Resize(size=(84, 4*84))])
            # (batch * num_action, 84, frame * 84)
            masked_image = transformer(masked_image.reshape(-1, 3, shape[2], shape[3]))
            inversed_masked_image = transformer(inversed_masked_image.reshape(-1, 3, shape[2], shape[3]))
            # (batch * num_action, frame, 84, 84)
            masked_image = torch.stack(masked_image.split(84, 2), 1)
            inversed_masked_image = torch.stack(inversed_masked_image.split(84, 2), 1)
            annotations = [e for e in annotations for _ in range(self.num_action)]
            logits, _ = self.agent(masked_image, annotations)
            logits = logits.reshape(shape[0], self.num_action, -1)
            logits_inversed, _ = self.agent(inversed_masked_image, annotations)
            logits_inversed = logits_inversed.reshape(shape[0], self.num_action, -1)

            return logits, logits_inversed, masks, segmentations, None, targets

    def highway_forward(self, image, targets, annotations):
        # (B, 4, 128, 64)
        state = torch.stack(image.squeeze(1).split(64, 2), 1)
        q_ori = self.agent(state)
        target_indices = q_ori.argmax(-1)
        targets = q_ori.new_zeros(target_indices.size(0), self.num_action)
        targets.scatter_(1, target_indices.unsqueeze(1), 1)

        image = image.float()
        # (batch, 1, H, W * frame)
        shape = image.shape
        # (batch, 1, H, W * frame)
        backgrounds = self.background_adapter.get_background(image)
        # backgrounds = torch.zeros_like(image)
        # (batch, num_actions, H, W * frame)
        segmentations = self.explainer(image)

        # (batch, num_actions, H, W * frame)
        masks = segmentations.sigmoid()
        inversed_masks = torch.ones_like(masks) - masks
        # (batch, num_actions, H, W * frame)
        masked_image = masks * image + inversed_masks * backgrounds
        inversed_masked_image = inversed_masks * image + masks * backgrounds

        # (batch * num_actions, H, W * frame)
        masked_image = masked_image.reshape(-1, shape[2], shape[3])
        inversed_masked_image = inversed_masked_image.reshape(-1, shape[2], shape[3])
        # (batch * num_action, frame, 128, 64)
        masked_image = torch.stack(masked_image.split(64, 2), 1)
        inversed_masked_image = torch.stack(inversed_masked_image.split(64, 2), 1)
        logits = self.agent(masked_image)
        logits = logits.reshape(shape[0], self.num_action, -1)
        logits_inversed = self.agent(inversed_masked_image)
        logits_inversed = logits_inversed.reshape(shape[0], self.num_action, -1)

        return logits, logits_inversed, masks, segmentations, None, targets

    def training_step(self, batch, batch_idx):
        image, annotations = batch

        if self.dataset == "DOOM":
            B = image['screen'].shape[0]
        else:
            B = image.shape[0]

        if self.dataset == 'ATARI':
            targets = get_targets_from_annotations(annotations, dataset=self.dataset, atari_env=self.atari_env)
        elif self.dataset in ["DOOM", "HIGHWAY"]:
            # image = batch
            targets = None
            # annotations = None
        else:
            targets = get_targets_from_annotations(annotations, dataset=self.dataset)

        logits, logits_inversed, masks, segmentations, logits_ori, targets = self(image, targets, annotations)
        _, idx = targets.max(dim=1)
        
        loss = (
            F.cross_entropy(logits[torch.arange(B), idx, :], targets) +
            F.cross_entropy(logits[torch.arange(B), :, idx], targets)
        ) / 2 

        loss += self.entropy_regularizer * (
            entropy_loss(logits_inversed) +
            entropy_loss(logits_inversed, dim=1)
        ) / 2

        logits_target = logits[torch.arange(B), idx, :]
        # mask_target = masks[torch.arange(B), idx, :]

        if self.use_stadler_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            mask_area_loss += self.mask_total_area_regularizer * masks.mean()
            # mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            loss += mask_area_loss
        else:
            loss += masks.mean()
            loss += self.mask_area_constraint_regularizer * torch.abs(masks.mean() - self.mask_max_area)

        if self.use_mask_variation_loss:
            shape = masks.shape
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(masks.view(-1, shape[-2], shape[-1])))
            loss += mask_variation_loss

        self.log('train_loss', loss, prog_bar=True)
        
        # self.metrics.update(logits_target, targets.argmax(dim=1))
        # self.qdiff.update(logits_ori=logits_ori, logits_mask=logits, targets=targets.argmax(-1))
        # self.log_dict(self.train_metrics.compute(), prog_bar=True)

        return loss

    # def on_train_epoch_start(self):
    #     self.train_start_time = default_timer()

    def on_train_epoch_end(self):
        # train_end_time = default_timer()
        # elapsed_time_seconds = train_end_time - self.train_start_time 
        # elapsed_time_hours = elapsed_time_seconds / 3600
        # print(f"Epoch Training time: {elapsed_time_hours:.6f} hours")

        # self.train_time.append(elapsed_time_hours)
        # print(self.train_time)
        # print(np.mean(self.train_time))

        # self.log_dict(self.metrics.compute(), prog_bar=True)
        # self.metrics.reset()

        # self.log('Q diff', self.qdiff.compute(), prog_bar=True)
        # self.qdiff.reset()
        pass

    # (screen: (B, 5, 60, 40), "variables": (B, 10))
    def validation_step(self, batch, batch_idx):
        image, annotations = batch

        if self.dataset == "DOOM":
            B = image['screen'].shape[0]
        else:
            B = image.shape[0]

        if self.dataset == 'ATARI':
            targets = get_targets_from_annotations(annotations, dataset=self.dataset, atari_env=self.atari_env)
        elif self.dataset in ["DOOM", "HIGHWAY"]:
            # image = batch
            targets = None
            # annotations = None
        else:
            targets = get_targets_from_annotations(annotations, dataset=self.dataset)

        logits, logits_inversed, masks, segmentations, logits_ori, targets = self(image, targets, annotations)
        _, idx = targets.max(dim=1)
        
        loss = (
            F.cross_entropy(logits[torch.arange(B), idx, :], targets) +
            F.cross_entropy(logits[torch.arange(B), :, idx], targets)
        ) / 2 

        loss += self.entropy_regularizer * (
            entropy_loss(logits_inversed) +
            entropy_loss(logits_inversed, dim=1)
        ) / 2

        if self.use_stadler_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            mask_area_loss += self.mask_total_area_regularizer * masks.mean()
            # mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            loss += mask_area_loss
        else:
            loss += masks.mean()
            loss += self.mask_area_constraint_regularizer * torch.abs(masks.mean() - self.mask_max_area)

        logits_target = logits[torch.arange(B), idx, :]
        # mask_target = masks[torch.arange(B), idx, :]

        if self.use_mask_variation_loss:
            shape = masks.shape
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(masks.view(-1, shape[-2], shape[-1])))
            loss += mask_variation_loss

        self.log('val_loss', loss)
        
        self.metrics.update(logits_target, targets.argmax(dim=1))
        # self.qdiff.update(logits_ori=logits_ori, logits_mask=logits_mask, targets=targets.argmax(-1))

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics.compute(), prog_bar=True)
        self.metrics.reset()

        # self.log('Q diff', self.qdiff.compute(), prog_bar=True)
        # self.qdiff.reset()

    def test_step(self, batch, batch_idx):
        image, annotations = batch

        if self.dataset == "DOOM":
            B = image['screen'].shape[0]
        else:
            B = image.shape[0]

        if self.dataset == 'ATARI':
            targets = get_targets_from_annotations(annotations, dataset=self.dataset, atari_env=self.atari_env)
        elif self.dataset in ["DOOM", "HIGHWAY"]:
            # image = batch
            targets = None
            # annotations = None
        else:
            targets = get_targets_from_annotations(annotations, dataset=self.dataset)

        start_time = default_timer()
        logits, logits_inversed, masks, segmentations, logits_ori, targets = self(image, targets, annotations)
        _, idx = targets.max(dim=1)
        
        loss = (
            F.cross_entropy(logits[torch.arange(B), idx, :], targets) +
            F.cross_entropy(logits[torch.arange(B), :, idx], targets)
        ) / 2 

        loss += self.entropy_regularizer * (
            entropy_loss(logits_inversed) +
            entropy_loss(logits_inversed, dim=1)
        ) / 2

        # loss += self.mask_area_constraint_regularizer * (masks.mean() - self.class_mask_max_area)
        if self.use_stadler_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            mask_area_loss += self.mask_total_area_regularizer * masks.mean()
            # mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            loss += mask_area_loss
        else:
            loss += masks.mean()
            loss += self.mask_area_constraint_regularizer * torch.abs(masks.mean() - self.mask_max_area)

        if self.use_mask_variation_loss:
            shape = masks.shape
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(masks.view(-1, shape[-2], shape[-1])))
            loss += mask_variation_loss

        end_time = default_timer()
        self.total_time += end_time - start_time

        if self.dataset == "ATARI":
            # (batch, 3, H_c, W_c * frame)
            backgrounds = self.background_adapter.get_content_background(image)
        elif self.dataset == "MARIO":
            backgrounds = self.background_adapter.get_content_background(image, annotations)
        elif self.dataset in ["DOOM"]:
            backgrounds = torch.zeros_like(image['screen'])
        elif self.dataset in ["HIGHWAY"]:
            backgrounds = self.background_adapter.get_background(image)

        mask_target = masks[torch.arange(B), idx, :]
        logits_target = logits[torch.arange(B), idx, :]

        if self.use_deletion_insertion:
            deletion_filename = [Path(self.save_path) / "deletion" / name for name in get_filename_from_annotations(annotations, dataset=self.dataset)]
            auc = self.deletion(self.agent, image, mask_target, backgrounds, annotations, targets, deletion_filename, self.deletion_fraction)
            self.deletion_auc.extend(auc)

            insertion_filename = [Path(self.save_path) / "insertion" / name for name in get_filename_from_annotations(annotations, dataset=self.dataset)]
            auc = self.insertion(self.agent, image, mask_target, backgrounds, annotations, targets, insertion_filename, self.insertion_fraction)
            self.insertion_auc.extend(auc)

        if self.save_masked_images:
            heat_filename = [Path(self.save_path) / "heatmap" / name for name in get_filename_from_annotations(annotations, dataset=self.dataset)]
            masked_filename = [Path(self.save_path) / "masked_images" / name for name in get_filename_from_annotations(annotations, dataset=self.dataset)]
            if self.dataset == "MARIO":
                # print(heat_filename)
                # print(annotations)
                # for a in range(self.num_action):
                #     save_heatmap_mario(image.detach().cpu(), masks[torch.arange(B), a, :].detach().cpu(), [e / f"{a}" for e in heat_filename])
                # exit()

                save_heatmap_mario(image.detach().cpu(), mask_target.detach().cpu(), heat_filename)
                save_masked_image_mario(image.detach().cpu(), mask_target.detach().cpu(), backgrounds.detach().cpu(), masked_filename)
            elif self.dataset == "ATARI":
                atari_save_heatmap(image.detach().cpu(), mask_target.detach().cpu(), heat_filename, atari_env=self.atari_env)
                atari_save_masked_image(image.detach().cpu(), mask_target.detach().cpu(), backgrounds.detach().cpu(), masked_filename, atari_env=self.atari_env)
            elif self.dataset in ["HIGHWAY"]:
                highway_save_heatmap(image.detach().cpu(), mask_target.detach().cpu(), heat_filename)
            elif self.dataset in ["DOOM"]:
                state = image['screen']
                var = image['variables']
                state = torch.cat(state.unbind(dim=1), dim=2)
                highway_save_heatmap(state.unsqueeze(1).detach().cpu(), mask_target.detach().cpu(), heat_filename)
            else:
                save_masked_image(image, mask_target, masked_filename)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)
            if self.dataset == "MARIO":
                mario_save_mask(mask_target, Path(self.save_path) / "masks" / filename[0])
            elif self.dataset == "ATARI":
                atari_save_mask(mask_target, Path(self.save_path) / "masks" / filename[0])
            else:
                save_mask(mask_target, Path(self.save_path) / "masks" / filename[0])

        self.log('test_loss', loss, prog_bar=True)
        self.metrics.update(logits_target, targets.argmax(dim=1))

        if self.dataset == "MARIO":
            contents, info = mario_seperate_content_info(image)
            self.ro_sp_metrics.update(x=contents, y=targets, exp_func=exp_func,
                                   model=self.agent, annotations=annotations, info=info, explainer=self,
                                   env=self.dataset)
        elif self.dataset in ["HIGHWAY"]:
            self.ro_sp_metrics.update(x=image, y=targets, model=self.agent, exp_func=exp_func,
                                   env=self.dataset, explainer=self)
        elif self.dataset in ["DOOM"]:
            state = image['screen']
            state = torch.cat(state.unbind(dim=1), dim=2) # (1, 60, 200)
            self.ro_sp_metrics.update(x=state.unsqueeze(1), y=targets, model=self.agent, exp_func=exp_func,
                                   env=self.dataset, explainer=self, info=image['variables'])
        elif self.dataset == "ATARI":
            contents, info = atari_seperate_content_info(image, atari_env=self.atari_env)
            self.ro_sp_metrics.update(x=contents, y=targets, model=self.agent, exp_func=exp_func,
                                   env=self.dataset, explainer=self, annotations=annotations, info=info, atari_env=self.atari_env)
        # self.foo_metrics.update(logits_mask, targets.argmax(dim=1))
        # self.qdiff.update(logits_ori=logits_ori, logits_mask=logits_mask, targets=targets.argmax(-1))
        # print("vm ", self.test_metrics.compute())
        # self.log_dict(self.test_metrics.compute(), prog_bar=True)
        
        if self.save_confusion_matrix:
            self.confusion_matrix[targets.argmax(dim=1).cpu(), logits_target.argmax(dim=1).cpu()] += 1
        
    def on_test_epoch_end(self):
        self.log_dict(self.metrics.compute(), prog_bar=True)
        self.log_dict(self.ro_sp_metrics.compute(), prog_bar=True)
        self.metrics.reset()
        # self.foo_metrics.compute()
        # self.test_metrics.save(model="explainer", classifier_type=self.classifier_type, dataset=self.dataset)
        # self.mario_testing_res_saver.save(self.save_path + '/' + 'testing_sample_details.csv')
        # print('RUIIII score is', self.fidelity.compute())
        print("TIME IS ", self.total_time, " seconds")

        if self.use_deletion_insertion:
            self.deletion_auc = torch.cat(self.deletion_auc)
            print('self.deletion_auc.std()', self.deletion_auc.std())
            print('self.deletion_auc.mean()', self.deletion_auc.mean())
 
            self.insertion_auc = torch.cat(self.insertion_auc)
            print('self.insertion_auc.std()', self.insertion_auc.std())
            print('self.insertion_auc.mean()', self.insertion_auc.mean())

        if self.save_confusion_matrix:
            save_confusion_matrix(self.confusion_matrix.numpy(), self.dataset,
                                  Path(self.save_path) / "confusion_matrix", atari_env=self.atari_env if self.dataset == "ATARI" else None)

        # self.log_dict(self.qdiff.compute(), prog_bar=True)
        # self.qdiff.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

def exp_func(model=None, inputs=None, targets=None, **kwargs):
    # print("exp_func")
    explainer = kwargs['explainer']
    inputs = torch.from_numpy(inputs).to(explainer.device)
    targets = torch.from_numpy(targets).to(explainer.device)
    B = inputs.shape[0]

    env = kwargs['env']
    if env == "MARIO":
        info = kwargs['info']
        image = torch.cat((info, inputs), dim=2)
        logits, logits_inversed, masks, segmentations, logits_ori, targets = explainer(image=image, targets=targets, annotations=kwargs['annotations'])
    elif env == "HIGHWAY":
        logits, logits_inversed, masks, segmentations, logits_ori, targets = explainer(image=inputs, targets=targets, annotations=None)
    elif env == "ATARI":
        info = kwargs['info']
        image = atari_concat_content_info(content=inputs, info=info, atari_env=kwargs['atari_env'])
        logits, logits_inversed, masks, segmentations, logits_ori, targets = explainer(image=image, targets=targets, annotations=kwargs['annotations'])
    elif env == "DOOM":
        info = kwargs['info']
        # (1, 1, 60, 200)
        inputs = torch.cat(inputs.split(40, 3), 1)
        image = {'screen': inputs, 'variables': info}
        logits, logits_inversed, masks, segmentations, logits_ori, targets = explainer(image=image, targets=targets, annotations=None)


    _, idx = targets.max(dim=1)
    mask_target = masks[torch.arange(B), idx, :]

    return mask_target.detach().cpu().numpy()