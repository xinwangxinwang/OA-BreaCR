import os
import shutil
import argparse
import json
import logging
import wandb
wandb.login()
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# Append relative paths to sys.path
sys.path.append(os.path.join(current_dir))
sys.path.append(os.path.join(current_dir, 'losses'))
sys.path.append(os.path.join(current_dir, 'learning'))
sys.path.append(os.path.join(current_dir, 'models'))
sys.path.append(os.path.join(current_dir, 'models', 'onconet'))
import warnings
warnings.filterwarnings('ignore')
from utils.opts import arg_parse, get_criterion, get_model, get_learning_demo, get_optimizer, get_dataset
from utils.utils import *
from utils.mylogging import open_log
from utils import backup


def main():
    args = arg_parse()
    seed_reproducer(seed=args.random_seed)

    open_log(args)
    logging.info(str(args).replace(',', "\n"))

    if 'Inhouse' in args.csv_dir or 'inhouse' in args.csv_dir or 'nki' in args.csv_dir:
        dataset_name = 'Inhouse'
    elif 'embed' in args.csv_dir:
        dataset_name = 'EMBED'
    else:
        dataset_name = 'Unknow'

    run = wandb.init(
        # Set the project where this run will be logged
        project=f"GL-Mammo-risk-{args.model_method}-{dataset_name}-followup{args.max_followup}",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            # "batch_size": args.batch_size
        })

    # Training settings
    # ---------------------------------
    best_c_index = 0.0

    data_info = pd.read_csv(args.csv_dir)
    data_info.to_csv(args.results_dir + '/data_info.csv')
    train_data_info = data_info[data_info['split_group'] == 'train']
    train_data_info = train_data_info.reset_index()
    valid_data_info = data_info[data_info['split_group'] == 'valid']
    valid_data_info = valid_data_info.reset_index()
    test_data_info = data_info[data_info['split_group'] == 'test']
    test_data_info = test_data_info.reset_index()

    train_data_info.to_csv(args.results_dir + '/train_data_info.csv')
    valid_data_info.to_csv(args.results_dir + '/valid_data_info.csv')
    test_data_info.to_csv(args.results_dir + '/test_data_info.csv')

    # define dataset class
    # ---------------------------------
    mammo_dataset = get_dataset(args)
    train_loader = mammo_dataset.dataloador(train_data_info, args, train=True)
    valid_loader = mammo_dataset.dataloador(valid_data_info, args, val_shuffle=True)
    test_loader = mammo_dataset.dataloador(test_data_info, args, val_shuffle=True)
    # --------------------------------------
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    # define Model and loss
    # ----------------------------------------
    logging.info('finish data loader')

    # define criterion
    # ----------------------------------------
    criterion = get_criterion(args)

    # define Model
    # ----------------------------------------
    model = get_model(args)

    # define train val test demo
    # ----------------------------------------
    train, validate, test = get_learning_demo(args)
    # ---------------------------------

    # define optimizer
    # ---------------------------------
    optimizer = get_optimizer(args, model)

    epoch_start = args.start_epoch
    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=1, mode='max', patience=args.lr_decay_patient, factor=0.5)
    early_stopping = EarlyStopping(patience=args.early_stop_patient, verbose=True, mode='max')
    set_backup(args.results_dir)
    # load resume model
    # ----------------------------------------
    if args.resume is not '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        best_c_index = checkpoint['best_c_index']
        logging.info('Loaded from: {}'.format(args.resume))
    # ---------------------------------

    cudnn.benchmark = True
    for epoch in range(epoch_start, args.epochs + 1):
        # =========== start train =========== #
        if epoch == 0:
            # =========== zero shoot test with initial weights =========== #
            # valid_result = validate(model, valid_loader, criterion, args)
            # logging.info('epoch: {}: valid_loss: {}, valid_MAE: {}, valid_ACC:{}, C-index: {}'.format(
            #     epoch, valid_result['loss'], valid_result['mae'], valid_result['acc'], valid_result['c_index']))

            test_result = test(model, test_loader, criterion, args)
            logging.info(
                'Init pretrained Model in val dataset, loss : {}, MAE: {}, ACC: {}, C-index: {}'.format(
                    test_result['loss'], test_result['mae'], test_result['acc'], test_result['c_index']))

        else:
            # ===========  train for one epoch   =========== #
            train_result = train(model, train_loader, criterion, optimizer, epoch, args)
            logging.info('epoch: {}: train_loss: {}, train_MAE: {}, train_ACC:{}, C-index: {}'.format(
                epoch, train_result['loss'], train_result['mae'], train_result['acc'], train_result['c_index']))

            # ===========  evaluate on validation set ===========  #
            valid_result = validate(model, valid_loader, criterion, args)
            logging.info('epoch: {}: valid_loss: {}, valid_MAE: {}, valid_ACC:{}, C-index: {}'.format(
                epoch, valid_result['loss'], valid_result['mae'], valid_result['acc'], valid_result['c_index']))

            # ===========  learning rate decay =========== #
            scheduler.step(valid_result['c_index'])
            for param_group in optimizer.param_groups:
                print("\n*learning rate {:.2e}*\n".format(param_group['lr']))

            # ===========  record the  best metric and save checkpoint ===========  #
            is_best = valid_result['c_index'] > best_c_index
            best_c_index = max(valid_result['c_index'], best_c_index)

            if is_best:
                # ===========  evaluate on test set ===========  #
                test_result = test(model, test_loader, criterion, args, save_pkl=f'best_{epoch}')
                # print('P_value', P_value)
                logging.info(
                    'epoch: {} is test best now, Model_best in test dataset, MAE: {}, ACC:{}, C-index: {}'.format(
                        epoch, test_result['mae'], test_result['acc'], test_result['c_index']))

            wandb.log({
                "Test loss": test_result['loss'],
                "Test MAE": test_result['mae'],
                "Test acc": test_result['acc'],
                "Test c_index": test_result['c_index'],
                "Train loss": train_result['loss'],
                "Train MAE": train_result['mae'],
                "Train acc": train_result['acc'],
                "Train c_index": train_result['c_index'],
                "Val loss": valid_result['loss'],
                "Val MAE": valid_result['mae'],
                "Val acc": valid_result['acc'],
                "Val c_index": valid_result['c_index'],
            })

            save_checkpoint(args.results_dir,
                {'epoch': epoch + 1,
                 'arch': args.arch,
                 'state_dict': model.state_dict(),
                 'best_c_index': best_c_index,
                 'optimizer': optimizer.state_dict(),
                 'train_result': train_result,
                 'valid_result': valid_result,
                 'test_result': test_result,
                 }, is_best)

            # ===========  early_stopping needs the validation loss or MAE to check if it has decresed
            early_stopping(valid_result['c_index'])
            if early_stopping.early_stop:
                logging.info("======= Early stopping =======")
                break

    # ===========  evaluate final best model on test set ===========  #
    checkpoint = torch.load(args.results_dir + '/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    test_result = test(model, test_loader, criterion, args, save_pkl='best')

    wandb.log({
        "Final Test loss": test_result['loss'],
        "Final Test MAE": test_result['mae'],
        "Final Test acc": test_result['acc'],
        "Final Test c_index": test_result['c_index'],
    })

    logging.info('Final test Model_best in test dataset, MAE: {}, ACC:{}, C-index: {}'.format(
        test_result['mae'], test_result['acc'], test_result['c_index']))


def set_backup(custom_backup_dir="custom_backups"):
    custom_backup_dir = os.path.join(custom_backup_dir, "src_backup")
    # Save backup of the current script
    backup.save_script_backup(__file__, custom_backup_dir)
    # Backup all imported modules within the project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    backup.backup_imported_modules(project_root, custom_backup_dir)


if __name__ == '__main__':
    main()
