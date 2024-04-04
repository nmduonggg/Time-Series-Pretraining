from .convMTM import ConvMTM, ConvLead

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
    

def get_model(configs, args):
    if args.model.lower()=='cnn' and args.training_mode=='pre_train':
        print("Use ConvLead")
        from .convMTM import target_classifier
        return ConvLead(configs, args), target_classifier(configs)
    elif args.model.lower()=='cnn' and args.training_mode!='pre_train':
        print("Use ConvTFC")
        from .convMTM import target_classifier
        return ConvMTM(configs, args), target_classifier(configs)
    else:
        assert 0, 'No model found'