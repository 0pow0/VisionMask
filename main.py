import os
import sys

import torch
import os
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path

from data.dataloader import MarioDataModule, AtariDataModule, DoomDataModule, HighwayDataModule
from utils.argparser import get_parser
from models.vm import VisionMask
from utils.atari_utils import get_atari_number_of_actions

main_dir = Path(os.path.dirname(os.path.abspath(__file__)))

parser = get_parser()
args = parser.parse_args()
os.makedirs(args.save_path, exist_ok=True)

pl.seed_everything(args.seed)

# Set up Logging
log_dir = args.save_path + "/tb_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logger = pl.loggers.TensorBoardLogger(log_dir, name="VisionMask")

if args.dataset == "MARIO":
    data_path = main_dir / args.data_base_path / 'MARIO'/ args.agent_type / args.mario_dataset 
    data_module = MarioDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_action = 7
elif args.dataset == "ATARI":
    data_path = main_dir / args.data_base_path / 'ATARI'/ args.agent_type / args.atari_env
    data_module = AtariDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size
    )
    num_action = get_atari_number_of_actions(args.atari_env)
elif args.dataset == "DOOM":
    data_path = main_dir / args.data_base_path / 'DOOM'/ args.agent_type
    data_module = DoomDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size
    )
    num_action = 7
elif args.dataset == "HIGHWAY":
    data_path = main_dir / args.data_base_path / 'HIGHWAY'/ args.agent_type
    data_module = HighwayDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size
    )
    num_action = 5
else:
    raise Exception("Unknown dataset " + args.dataset)

# Set up model
model = VisionMask(
    num_action=num_action, dataset=args.dataset, agent_type=args.agent_type, agent_ckpt=args.agent_ckpt, fix_classifier=args.fix_classifier, learning_rate=args.learning_rate, 
    class_mask_min_area=args.class_mask_min_area, mask_max_area=args.mask_max_area, entropy_regularizer=args.entropy_regularizer, use_mask_variation_loss=args.use_mask_variation_loss, 
    mask_variation_regularizer=args.mask_variation_regularizer, use_mask_area_loss=args.use_mask_area_loss, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, 
    mask_total_area_regularizer=args.mask_total_area_regularizer, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, 
    save_masked_images=args.save_masked_images, save_masks=args.save_masks,
    save_all_class_masks=args.save_all_class_masks, save_path=args.save_path,
    mario_use_greyscale=args.mario_use_greyscale,
    deletion_fraction=args.deletion_fraction, insertion_fraction=args.insertion_fraction,
    atari_env=args.atari_env,
    use_deletion_insertion=args.use_deletion_insertion,
    weight_decay=args.weight_decay, save_confusion_matrix=args.save_confusion_matrix,
    explainer_type=args.explainer_type, 
    use_stadler_mask_area_loss=args.use_stalder_mask_area_loss,
    ref_value_mode=args.ref_value_mode
)

if args.vm_checkpoint is not None:
        model = VisionMask.load_from_checkpoint(
            args.explainer_classifier_checkpoint,
            num_action=num_action, dataset=args.dataset, agent_type=args.agent_type, agent_ckpt=args.agent_ckpt, fix_classifier=args.fix_classifier, learning_rate=args.learning_rate, 
            class_mask_min_area=args.class_mask_min_area, mask_max_area=args.mask_max_area, entropy_regularizer=args.entropy_regularizer, use_mask_variation_loss=args.use_mask_variation_loss, 
            mask_variation_regularizer=args.mask_variation_regularizer, use_mask_area_loss=args.use_mask_area_loss, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, 
            mask_total_area_regularizer=args.mask_total_area_regularizer, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, 
            save_masked_images=args.save_masked_images, save_masks=args.save_masks, save_all_class_masks=args.save_all_class_masks, save_path=args.save_path,
            mario_use_greyscale=args.mario_use_greyscale,
            deletion_fraction=args.deletion_fraction, insertion_fraction=args.insertion_fraction,
            atari_env=args.atari_env,
            use_deletion_insertion=args.use_deletion_insertion,
            weight_decay=args.weight_decay, save_confusion_matrix=args.save_confusion_matrix,
            explainer_type=args.explainer_type,
            use_stadler_mask_area_loss=args.use_stalder_mask_area_loss,
        ref_value_mode=args.ref_value_mode
        )

# Define Early Stopping condition
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=args.early_stop_min_delta,
    patience=args.early_stop_patience,
    verbose=False,
    mode="min",
)

trainer = pl.Trainer(
    logger = logger,
    callbacks = [early_stop_callback],
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
    # terminate_on_nan = True,
    # checkpoint_callback = args.checkpoint_callback,
)

if args.train_model:
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(datamodule=data_module)
else:
    trainer.test(model=model, datamodule=data_module)
