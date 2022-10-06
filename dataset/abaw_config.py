configs = {
    # setting
    'stride':10,
    'time_window':2,
    # type your data_path
    'data_path': '/mnt/c/Data/Yuxuan/ABAW',
    'limit_gpu':1024*4.5,
    'gpu_num': 0,
    # type your result_path need to evaluate
    'eval_path': '',

    # train
    'epochs':20,
    'batch_size':256,
    'early_stop': 5,
    'learning_rate': 0.0001,

    # model
    'feature_extractor_layers':[1024,512],
    'classifier_layers': [256,128],
    'domain_layers': [64,32],
    'lstm_num': 512,
    'dropout_rate':0.5,

    # hyper parametre
    'temperature':2.5,
    'alpha':10,
    'beta':0.9,
    'gamma':1,
    'au_threshold':0.5,

    # task_weight
    'task_weight_exp':0.5,
    'task_weight_flag':True,

     # domain adaptation
    'domain_weight':1,
    'adaptation_factor':1,
    'linear_domain_weight':True

}