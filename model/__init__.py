def build_model(dataset='mnist', model=None):
    if model == 'feature extractor':
        from .feature_extractor import Feature_Extractor
        return Feature_Extractor()
    elif model == 'region proposal':
        from .region_proposal import Region_Proposal
        return Region_Proposal()

