import copy
from .convTFC import ConvTFC, ConvLead
from .transTFC import TransTFC

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
    

def get_model(args, configs):
    if args.model.lower()=='transformer':
        print("User TransTFC")
        from .transTFC import target_classifier
        return TransTFC(configs), target_classifier(configs)
    elif args.model.lower()=='cnn' and args.training_mode=='pre_train':
        print("Use ConvLead")
        from .convTFC import target_classifier
        return ConvLead(configs), target_classifier(configs)
    elif args.model.lower()=='cnn' and args.training_mode!='pre_train':
        print("Use ConvTFC")
        from .convTFC import target_classifier
        return ConvTFC(configs), target_classifier(configs)
    else:
        assert 0, 'No model found'