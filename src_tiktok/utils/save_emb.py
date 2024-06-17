# coding: utf-8

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import torch
import numpy as np


def save_emb(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)
        save_dir = '/data1/gpxu/MM_pre/src/log/' + config['model']+ '_' + str(config['hyper_parameters']) + str(hyper_tuple)

        model.load_state_dict(torch.load(save_dir + '/best.pth'))

        t_feat_online = model.text_trs(model.text_embedding.weight)
        v_feat_online = model.image_trs(model.image_embedding.weight)

        t_feat_online=t_feat_online.detach().cpu().numpy() # cpu().numpy()
        v_feat_online=v_feat_online.detach().cpu().numpy()# .cpu().numpy()

        np.save(save_dir + '/t_feat_online.npy',t_feat_online)
        np.save(save_dir + '/v_feat_online.npy',v_feat_online)
        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug

