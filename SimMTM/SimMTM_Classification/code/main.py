import numpy as np
from datetime import datetime
import argparse
from utils.utils import _logger
from model import get_model
from dataloader_custom import data_generator
from trainer import Trainer, Trainer_ft
import os
import torch

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
parser.add_argument('--seed', default=2023, type=int, help='seed value')

parser.add_argument('--training_mode', default='pre_train', type=str, help='pre_train, fine_tune')
parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str,
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--model', type=str, default='cnn')
parser.add_argument('--use_pretrain', action='store_true')
parser.add_argument('--wandb', action='store_true')

parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
parser.add_argument('--subset', action='store_true', default=False, help='use the subset of datasets')
parser.add_argument('--log_epoch', default=1, type=int, help='print loss and metrix')
parser.add_argument('--draw_similar_matrix', default=10, type=int, help='draw similarity matrix')
parser.add_argument('--pretrain_lr', default=0.0001, type=float, help='pretrain learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--use_pretrain_epoch_dir', default=None, type=str,
                    help='choose the pretrain checkpoint to finetune')
parser.add_argument('--pretrain_epoch', default=10, type=int, help='pretrain epochs')
parser.add_argument('--finetune_epoch', default=300, type=int, help='finetune epochs')

parser.add_argument('--masking_ratio', default=0.5, type=float, help='masking ratio')
parser.add_argument('--positive_nums', default=3, type=int, help='positive series numbers')
parser.add_argument('--lm', default=3, type=int, help='average masked lenght')

parser.add_argument('--finetune_result_file_name', default="finetune_result.json", type=str,
                    help='finetune result json name')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature')


def set_seed(seed):
    SEED = seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    return seed

def reconstruct_conv_lead(pretrain_dict):
    """
    Reconstruct pretrain dict: add module name and idx before the ConvLead
    """
    all_lead_dict = dict()
    n_leads = 12
    for l in range(n_leads):
        for k in pretrain_dict.keys():
            all_lead_dict[f"convs.{l}.{k}"] = pretrain_dict[k].clone()
            
    return all_lead_dict


def main(args, configs, seed=None):
    method = 'SimMTM'
    sourcedata = args.pretrain_dataset
    targetdata = args.target_dataset
    training_mode = args.training_mode
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    masking_ratio = args.masking_ratio
    pretrain_lr = args.pretrain_lr
    pretrain_epoch = args.pretrain_epoch
    lr = args.lr
    finetune_epoch = args.finetune_epoch
    temperature = args.temperature
    experiment_description = f"{sourcedata}_2_{targetdata}"

    os.makedirs(logs_save_dir, exist_ok=True)

    # Load datasets
    sourcedata_path = "/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/SPECIFIC_DATA/TFC/pretrain"
    targetdata_path = "/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED/PTBXL_REDUCE/"
    subset = args.subset # if subset= true, use a subset for debugging.
    train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset = subset)

    # set seed
    if seed is not None:
        seed = set_seed(seed)
    else:
        seed = set_seed(args.seed)

    # experiments_logs/SleepEEG/run1/pre_train_2023_pt_0.5_0.0001_50_ft_0.0003_100
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      training_mode + f"_{seed}_pt_{masking_ratio}_{pretrain_lr}_{pretrain_epoch}_ft_{lr}_{finetune_epoch}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Pre-training Dataset: {sourcedata}')
    logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
    logger.debug(f'Seed: {seed}')
    logger.debug(f'Method:  {method}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug(f'Pretrain Learning rate:    {pretrain_lr}')
    logger.debug(f'Masking ratio:    {masking_ratio}')
    logger.debug(f'Pretrain Epochs:    {pretrain_epoch}')
    logger.debug(f'Finetune Learning rate:    {lr}')
    logger.debug(f'Finetune Epochs:    {finetune_epoch}')
    logger.debug(f'Temperature: {temperature}')
    logger.debug("=" * 45)

    # Load Model
    TFC_model, classifier = get_model(configs, args)
    TFC_model = TFC_model.to(device)
    classifier = classifier.to(device)
    
    if args.use_pretrain:
        # load saved model of this experiment
        # load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, args.model,
        # f"pre_train_seed_{SEED}", "saved_models"))
        load_from = '/mnt/disk1/nmduong/ECG-Pretrain/SimMTM/SimMTM_Classification/code/experiments_logs/ECG_2_PTBXL/ecgOnly/pre_train_2023_pt_0.5_0.0001_10_ft_0.0001_300/saved_models'
        
        print("The loading file path", load_from)
        
        chkpoint = torch.load(os.path.join(load_from, "_last.pt"), map_location=device)
        torch.save(chkpoint, os.path.join(load_from, "_last_bkup.pt"))
        
        pretrained_dict = chkpoint["model_state_dict"]
        
        if args.training_mode=='pre_train':
            TFC_model.load_state_dict(pretrained_dict, strict=True)
            print("Load for pretrain")
        else:
            pretrained_dict = reconstruct_conv_lead(pretrained_dict)
            TFC_model.load_state_dict(pretrained_dict, strict=False)
            print("Load for finetune")

    else:
        print("Not use pretrain, train from scratch")

    for md in [TFC_model, classifier]:
        for pram in md.parameters():
            pram.require_grads=True
        
    model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=configs.lr_f, betas=(configs.beta1, configs.beta2), weight_decay=0)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=args.finetune_epoch)
    
    # Trainer
    if args.training_mode == 'pre_train':
        print("Pretraining mode")
        best_performance = Trainer(TFC_model, model_optimizer, model_scheduler, train_dl, valid_dl, test_dl, device, logger,
                               args, configs, experiment_log_dir, seed)
    else:
        print("Finetune mode")
        best_performance = Trainer_ft(TFC_model, model_optimizer, model_scheduler, train_dl, valid_dl, test_dl, device, logger,
                               args, configs, experiment_log_dir, seed)

    return best_performance


if __name__ == '__main__':
    import wandb
    args, unknown = parser.parse_known_args()
    device = torch.device(args.device)
    # exec (f'from config_files.{args.pretrain_dataset}_Configs import Config as Configs')
    from config_files.OnlyECG_Configs import Config as Configs
    configs = Configs()
    
    if args.wandb:
        wandb.login(key='60fd0a73c2aefc531fa6a3ad0d689e3a4507f51c')
        wandb.init(
            project='TimeSeriesPretrain',
            group=f'SimMTM-{args.training_mode}',
            name=f'SimMTM-{args.model}-ECG2PTBXL-{args.training_mode}', 
            entity='aiotlab',
            config=vars(configs)
        )    

    main(args, configs)

