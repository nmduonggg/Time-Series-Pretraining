import os
import numpy as np
from datetime import datetime
import argparse
import torch

from utils import _logger
from model import get_model
# from dataloader import data_generator
from dataloader_custom import data_generator
from trainer import Trainer

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=42, type=int, help='seed value')

# Model selection
parser.add_argument("--model", default="cnn", type=str,
                    help="CNN or Transformer-based TFC encoder")

# 1. self_supervised pre_train; 2. finetune (itself contains finetune and test)
parser.add_argument('--training_mode', default='fine_tune_test', type=str,
                    help='pre_train, fine_tune_test')
parser.add_argument('--use_pretrain', action='store_true', help='use pretrain or not')

parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str,
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args, unknown = parser.parse_known_args()

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

pretrain_dataset = args.pretrain_dataset
targetdata = args.target_dataset
# experiment_description = str(pretrain_dataset) + '_2_' + str(targetdata)
experiment_description = 'ECG2PTBXL'


method = 'TF-C'
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
use_pretrain = args.use_pretrain
os.makedirs(logs_save_dir, exist_ok=True)
# exec(f'from config_files.{pretrain_dataset}_Configs import Config as Configs')
from config_files.OnlyECG_Configs import Config as Configs
configs = Configs()

# # ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, args.model, training_mode + f"_seed_{SEED}")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {pretrain_dataset}')
logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
# sourcedata_path = f"../../datasets/{pretrain_dataset}"
# targetdata_path = f"../../datasets/{targetdata}"
sourcedata_path = "/mnt/disk4/nmduong/Time-Series-Pretraining/data_processing/ECG/SPECIFIC_DATA/TFC/pretrain"
targetdata_path = "/mnt/disk4/nmduong/Time-Series-Pretraining/data_processing/ECG/SPECIFIC_DATA/TFC/finetune"
subset = False # if subset= true, use a subset for debugging.
train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset = subset)
logger.debug("Data loaded ...")

# Load Model
"""Here are two models, one basemodel, another is temporal contrastive model"""
TFC_model, classifier = get_model(args, configs)
TFC_model = TFC_model.to(device)
classifier = classifier.to(device)
temporal_contr_model = None

if use_pretrain:
    # load saved model of this experiment
    # load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, args.model,
    # f"pre_train_seed_{SEED}", "saved_models"))
    load_from = '/mnt/disk4/nmduong/Time-Series-Pretraining/TFC/code/experiments_logs/ECG2PTBXL/ecgOnly/pre_train_seed_42_inception1dbase/saved_models/'
    
    print("The loading file path", load_from)
    
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)

else:
    print("Not use pretrain, train from scratch")

for md in [TFC_model, classifier]:
    for pram in md.parameters():
        pram.require_grads=True
        
model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=configs.lr_f, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# Trainer
Trainer(args, TFC_model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode)

logger.debug(f"Training time is : {datetime.now()-start_time}")
