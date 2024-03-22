from .convTFC import ConvTFC
from .transTFC import TransTFC

def get_model(args, configs):
    if args.model.lower()=='transformer':
        from .transTFC import target_classifier
        return TransTFC(configs), target_classifier(configs)
    elif args.model.lower()=='cnn':
        from .convTFC import target_classifier
        return ConvTFC(configs), target_classifier(configs)
    else:
        assert 0, 'No model found'