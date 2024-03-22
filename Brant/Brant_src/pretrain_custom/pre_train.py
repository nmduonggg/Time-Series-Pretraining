import random
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import autocast
from torch.cuda.amp import GradScaler

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from pre_utils import generate_mask, master_print, master_save, get_loss, _load_data_by_file_indices
from pre_dataset import PreDataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def feed_forward(time_enc, ch_enc,
                 mask, batch, use_power,
                 mask_loss_fn, mask_by_ch, rand_mask, mask_len,
                 rec_down_samp_rate,
                 ):
    data = batch[0]
    bat_size, ch_num, seq_len, seg_len = data.shape
    power = batch[1]

    # time encoder capture long-term dependency
    time_z = time_enc(mask, data, power,
                      need_mask=True,
                      mask_by_ch=mask_by_ch,
                      rand_mask=rand_mask,
                      mask_len=None,
                      use_power=use_power)  

    _, _, d_model = time_z.shape
    time_z = time_z.view(bat_size, ch_num, seq_len, d_model)    
    time_z = torch.transpose(time_z, 1, 2)                      
    time_z = time_z.reshape(bat_size*seq_len, ch_num, d_model)  

    _, rec = ch_enc(time_z)             # rec.shape: bat_size*seq_len, ch_num, seg_len // rec_down_samp_rate
    rec = rec.view(bat_size, seq_len, ch_num, seg_len // rec_down_samp_rate)
    rec = torch.transpose(rec, 1, 2)    # transpose back  rec.shape: bat_size, ch_num, seq_len, seg_len
    rec = rec.reshape(bat_size*ch_num*seq_len, seg_len // rec_down_samp_rate)

    data = data.view(bat_size * ch_num * seq_len, seg_len)[::rec_down_samp_rate]
    mask_loss = mask_loss_fn(data, rec)

    return mask_loss

def freeze_model(time_enc, ch_enc):
    # freeze time_enc
    enc_nb = 4
    for n, param in time_enc.named_parameters():
        # projector
        if 'input_embedding' in n:
            if 'proj' in n: param.requires_grad = True
            if 'band' in n or 'positional' in n: param.requires_grad = True
            else: param.requires_grad = False
        
        for i in range(enc_nb):
            if 'trans_enc' in n and str(i) in n:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    # freeze ch_enc
    for n, param in ch_enc.named_parameters():
        param.requires_grad = False
        
    return time_enc, ch_enc
            
def do_epoch(args, epoch,
             time_enc, ch_enc,
             optimizer, scheduler,
             files, scaler
             ):
    time_enc, ch_enc = freeze_model(time_enc, ch_enc)
    
    time_enc.train()
    ch_enc.train()
    mask_loss_fn = torch.nn.MSELoss(reduction='mean')
    tot_mask_loss = 0
    
    folder, file_indices = files
    nb_indices_per_group = 2000
    file_indices = file_indices[
        : file_indices.shape[0] // nb_indices_per_group * nb_indices_per_group
    ]
    file_indices = file_indices.reshape(nb_indices_per_group, -1)

    for idx in range(file_indices.shape[-1]):
        _file_idxs = file_indices[:, idx]
        data, power = _load_data_by_file_indices(_file_idxs, folder, cal_power=True, use_power=args.use_power)

        data = torch.tensor(data)
        # ch_num, seg_num, seg_len = data.shape

        # # truncation and reshape
        # board_num = seg_num // args.seq_len
        # seg_num = args.seq_len * board_num
        # data = data[:, :seg_num, :].view(ch_num, board_num, args.seq_len, -1)
        # data = torch.transpose(data, 0, 1)
        
        # print(data.shape)
        data = data.permute(1,0,2,3) # CxBxLD -> BxCxLD
        data = data.view(data.shape[0], data.shape[1], args.seq_len, -1)    # BxCxLxD
        n_samples, ch_num, seq_len, dim = data.shape
        power = power.permute(1,0,2,3)
        power = power.view(power.shape[0], power.shape[1], args.seq_len, -1)

        if args.use_power:
            power = torch.tensor(power)
            # power = power[:, :seg_num, :].view(ch_num, board_num, args.seq_len, -1)
            # power = torch.transpose(power, 0, 1)

        mask = generate_mask(ch_num, args.seq_len, args.mask_ratio)
        dataset = PreDataset(data, power)
        bat_size = args.train_batch_size
        if dist.is_initialized():
            sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=False, drop_last=False)
            dataloader = DataLoader(dataset, batch_size=bat_size, num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=True, sampler=sampler)
            dataloader.sampler.set_epoch(epoch)
        else:
            dataloader = DataLoader(dataset, batch_size=bat_size, shuffle=False, drop_last=False)

        for bat_idx, (_batch) in enumerate(dataloader):
            _batch[0] = _batch[0].to(device)
            _batch[1] = _batch[1].to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.amp):
                mask_loss = feed_forward(time_enc, ch_enc,
                                         mask, _batch, args.use_power,
                                         mask_loss_fn, args.mask_by_channel,
                                         args.rand_mask, args.mask_len,
                                         args.rec_down_samp_rate)
                mask_loss = mask_loss / args.accu_step
            if args.amp:
                scaler.scale(mask_loss).backward()
                if (bat_idx+1) % args.accu_step == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                mask_loss.backward()
                if (bat_idx + 1) % args.accu_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            tot_mask_loss += mask_loss.to(torch.float32)

    return tot_mask_loss


def run_pre_train(args, time_enc, ch_enc, optimizer, scheduler):
    scaler = GradScaler(enabled=args.amp)
    max_instances = 6300000
    file_indices = np.arange(max_instances)  # the data files used for pre-training
    folder = '/mnt/disk4/nmduong/Time-Series-Pretraining/data_processing/ECG/SPECIFIC_DATA/TFC/pretrain'
    for epo_idx in range(args.start_epo_idx + 1, args.num_epochs):
        master_print(f'\nEpoch {epo_idx} start')
        start = time.time()
        
        random.shuffle(file_indices)
        file_indices = file_indices[:(max_instances//args.train_batch_size * args.train_batch_size)]
        files = [folder, file_indices]
        
        mask_loss = do_epoch(args, epo_idx,
                             time_enc, ch_enc,
                             optimizer, scheduler,
                             files, scaler)

        mask_loss = get_loss(mask_loss)
        master_print(f'Train: mask_loss = %.6f' % mask_loss)

        master_save(time_enc.state_dict(), f'./encoder_ckpt/time_encoder_{args.start_epo_idx}.pt')
        master_save(ch_enc.state_dict(), f'./encoder_ckpt/channel_encoder_{args.start_epo_idx}.pt')
        master_print('This epoch spends %d s' % (time.time() - start))

