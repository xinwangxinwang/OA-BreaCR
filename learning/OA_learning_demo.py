import random
import gc
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
from utils.utils import *
from utils.compute_auc_c_index import compute_auc_cindex
from tqdm import tqdm
from scipy import stats

import numpy as np
import logging
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from torchvision.transforms import v2 as T

def cal_mae_acc(logits, targets, years_last_followup, is_sto=False, threshold=1, weights=None):
    """
        Calculate Mean Absolute Error (MAE) and Accuracy (ACC) for model (time-to-event) predictions.

        Parameters:
            logits (torch.Tensor): The predicted logits from the model.
            targets (torch.Tensor): The ground truth labels.
            years_last_followup (torch.Tensor): The years since the last follow-up for each sample.
            is_sto (bool, optional): Whether the model uses stochastic outputs. Defaults to False.
            threshold (int, optional): The threshold for accuracy calculation. Defaults to 1.
            weights (list or np.ndarray, optional): Weights for each class. Defaults to None.

        Returns:
            dict: A dictionary containing:
                - 'mae' (float): Mean Absolute Error.
                - 'acc' (float): Accuracy of time prediction.
                - 'mae_batch' (list): Batch-wise MAE values.
                - 'error_batch' (list): Batch-wise error values.
                - 'pred_time' (np.ndarray): Predicted ages for valid samples.
                - 'pred_time_all' (np.ndarray): Predicted ages for all samples.
                - 'count' (int): Number of valid samples.
        """

    if is_sto:
        r_dim, s_dim, out_dim = logits.shape
        assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
        logits = torch.mean(logits, dim=0)
        probs = F.softmax(logits, -1)
        # probs = torch.mean(probs, dim=0)
        # probs_data = probs.cpu().data.numpy()
        # label_arr = np.array(range(out_dim))
        # exp_data_all = np.sum(probs_data * label_arr, axis=-1)
        # exp_data_all = np.mean(exp_data_all, axis=0)
    else:
        s_dim, out_dim = logits.shape
        probs = F.softmax(logits, -1)
    probs_data = probs.cpu().data.numpy()
    label_arr = np.array(range(out_dim))
    exp_data_all = np.sum(probs_data * label_arr, axis=-1)

    target_data = targets.cpu().data.numpy()
    years_last_followup_data = years_last_followup.cpu().data.numpy()
    # label_arr = np.array(range(out_dim))
    # exp_data_all = np.sum(probs_data * label_arr, axis=1)
    target_data[target_data > (out_dim - 1)] = out_dim - 1
    mask = 1 - ((target_data == (out_dim - 1)) & (years_last_followup_data < (out_dim - 1))).astype(int)

    count = sum(mask)
    if count != 0:
        exp_data_all_ = exp_data_all[mask==1]
        target_data_ = target_data[mask==1]

        error_batch = exp_data_all_ - target_data_
        mae_batch = abs(error_batch)
        if weights is not None:
            weights = np.asarray(weights).reshape(1,-1)
            weights_ = np.repeat(weights, count, axis=0)
            weights_ = weights_[range(count), target_data_]
            # mae = sum(mae_batch * weights_) / sum(weights_)
            # acc = sum((np.rint(abs(exp_data_all_ - target_data_)) <= threshold) * weights_) * 1.0 / sum(weights_)
            mae = np.mean(mae_batch * weights_)
            acc = np.mean((np.rint(abs(exp_data_all_ - target_data_)) <= threshold) * weights_) * 1.0
        else:
            mae = sum(mae_batch) / len(target_data_)
            acc = sum(np.rint(abs(exp_data_all_ - target_data_)) <= threshold) * 1.0 / len(target_data_)
    else:
        mae = 0
        acc = 0
        mae_batch = []
        error_batch = []
        exp_data_all_ = []

    return {
        'mae': mae,
        'acc': acc,
        'mae_batch': mae_batch,
        'error_batch': error_batch,
        'pred_time': exp_data_all_,
        'pred_time_all': exp_data_all,
        'count': count
    }


def compute_losses(args, criterion, risk, risk_label_, years_last_followup_, emb, log_var, is_sto=False):
    # BCE loss
    criterion_BCE = criterion['criterion_BCE']
    risk_label = risk_label_.clone()
    years_last_followup = years_last_followup_.clone()

    if is_sto and emb is not None:
        sample_size, s_dim, out_dim = risk.shape
        assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
        loss = criterion_BCE(risk.view(-1, out_dim), risk_label.repeat([sample_size, 1]).view(-1),
                             years_last_followup.repeat([sample_size, 1]).view(-1), weights=args.time_to_events_weights)
    else:
        s_dim, out_dim = risk.shape
        loss = criterion_BCE(risk, risk_label, years_last_followup, weights=args.time_to_events_weights)

    # MV loss
    risk_label = risk_label_.clone()
    years_last_followup = years_last_followup_.clone()
    criterion_MV = criterion['criterion_MV']
    if is_sto and emb is not None:
        sample_size, s_dim, out_dim = risk.shape
        assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
        loss_MV = criterion_MV(risk.view(-1, out_dim), risk_label.repeat([sample_size, 1]).view(-1),
                            years_last_followup.repeat([sample_size, 1]).view(-1), weights=args.time_to_events_weights)
    else:
        s_dim, out_dim = risk.shape
        loss_MV = criterion_MV(risk, risk_label, years_last_followup, weights=args.time_to_events_weights)
    loss = loss + loss_MV

    # POE loss
    if emb is not None:
        risk_label = risk_label_.clone()
        years_last_followup = years_last_followup_.clone()
        criterion_POE = criterion['criterion_POE']
        _, _, _, loss_POE = criterion_POE(
            risk, emb, log_var, risk_label, years_last_followup, None, use_sto=is_sto, weights=args.time_to_events_weights)
        loss = loss + loss_POE
    return loss


def data_transform_on_GPU(args, img, prior_img, mask, prior_mask, train=False):
    """
        To improve GPU utilization and accelerate data processing, this function performs data transformations on the GPU.

        Note: the current implementation does not use masks or prior masks, so these parameters are set to None.

        Apply data transformation to input images and masks based on training or testing mode. The transformations include
        operations such as rotation, affine transformations, color jittering, and normalization. These operations are applied
        differently depending on whether the mode is training or testing.

        Parameters:
            args: object
                The configuration object containing parameters, including image size, used for transformations.
            img: torch.Tensor
                The input image tensor to be transformed.
            prior_img: torch.Tensor, optional
                The prior image tensor to be transformed. Can be None.
            mask: torch.Tensor or None
                The mask tensor associated with the input image. This is set to None as per current implementation.
            prior_mask: torch.Tensor or None
                The mask tensor associated with the prior image. This is set to None as per current implementation.
            train: bool, default=False
                Indicates whether the transformations are applied for training (augmentations enabled) or for testing
                (only basic normalizations applied).

        Returns:
            tuple
                A tuple containing four elements:
                - img: Transformed image tensor.
                - prior_img: Transformed prior image tensor, or None if prior_img was None.
                - mask: None, as masks are not used in the current implementation.
                - prior_mask: None, as prior masks are not used in the current implementation.
    """

    train_transform = torch.nn.Sequential(
        T.RandomRotation(10),
        T.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),
        T.ColorJitter(0.4, 0.4, 0.4, 0),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.5], [0.5])
        # transforms.CenterCrop(args.img_size),
    )
    test_transform = torch.nn.Sequential(
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.5], [0.5])
        # transforms.CenterCrop(args.img_size),
    )

    if train:
        img = train_transform(img)
        mask = None
        prior_img = train_transform(prior_img) if prior_img is not None else None
        prior_mask = None
    else:
        img = test_transform(img)
        mask = None
        prior_img = test_transform(prior_img) if prior_img is not None else None
        prior_mask = None

    return img, prior_img, mask, prior_mask


def input_and_output(args, model, input, train=False):
    """
    Processes the input data and passes it through the model to obtain the output.

    Parameters:
        args (object): Configuration object containing model and training parameters.
        model (torch.nn.Module): The model to process the input data.
        input (dict): Dictionary containing input data such as images, prior images, and masks.
        train (bool, optional): Indicates whether the function is being used in training mode. Defaults to False.

    Returns:
        dict: A dictionary containing the model's output and intermediate results:
            - 'risk': Final risk predictions.
            - 'emb': Embedding of the final output.
            - 'log_var': Log variance of the final output.
            - 'loss': Loss value if available in the model's output.
            - 'current_risk': Current risk predictions.
            - 'current_emb': Embedding of the current output.
            - 'current_log_var': Log variance of the current output.
            - 'difference_risk': Difference risk predictions.
            - 'difference_emb': Embedding of the difference output.
            - 'difference_log_var': Log variance of the difference output.
            - 'prior_risk': Prior risk predictions.
            - 'prior_emb': Embedding of the prior output.
            - 'prior_log_var': Log variance of the prior output.
            - 'output': The raw output from the model.
    """

    img = input['img'].cuda()
    prior_img = input['prior_img'].cuda()
    gap = input['gap'].cuda()
    mask = input['mask'].cuda() if 'mask' in input else None
    prior_mask = input['prior_mask'].cuda() if 'prior_mask' in input else None

    # Apply data transformations on GPU
    img, prior_img, mask, prior_mask = data_transform_on_GPU(args, img, prior_img, mask, prior_mask, train=train)

    # Pass the transformed data through the model
    output = model(img, prior_x=prior_img, time=gap, max_t=args.max_t, use_sto=args.use_sto,)

    # Extract outputs and intermediate results
    loss = output['loss'] if 'loss' in output else None
    risk = output['final']
    current_risk = output['current'] if 'current' in output else None
    prior_risk = output['prior'] if 'prior' in output else None
    difference_risk = output['difference'] if 'difference' in output else None
    emb, log_var = output['emb_final'], output['log_var_final']
    current_emb, current_log_var = output['emb_current'] if 'emb_current' in output else None, output['log_var_current'] if 'log_var_current' in output else None
    prior_emb, prior_log_var = output['emb_prior'] if 'emb_prior' in output else None, output['log_var_prior'] if 'log_var_prior' in output else None
    difference_emb, difference_log_var = output['emb_difference'] if 'emb_difference' in output else None, output['log_var_difference'] if 'log_var_difference' in output else None

    return {
        'risk': risk, 'emb': emb, 'log_var': log_var,
        'loss': loss,
        'current_risk': current_risk, 'current_emb': current_emb, 'current_log_var': current_log_var,
        'difference_risk': difference_risk, 'difference_emb': difference_emb, 'difference_log_var': difference_log_var,
        'prior_risk': prior_risk, 'prior_emb': prior_emb, 'prior_log_var': prior_log_var,
        'output': output,}


def plot_attent(args, input, output, i):
    if "OA-BreaCR" in args.model_method:
        dict = {'args': args, 'input': input, 'output': output,}
        pickle.dump(dict, open('{}/result_attent{}.pkl'.format(args.results_dir, i), 'wb'))


def direct_train(model, data_loader, criterion, optimizer, epoch, args):

    losses = AverageMeter()
    # mae = AverageMeter()
    # acc = AverageMeter()
    model.train()
    # adjust_learning_rate(optimizer, epoch, args)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    # all_risk_probabilities = []
    # all_followups = []
    # all_risk_label = []

    i_debug = 0
    for input in train_bar:
        if i_debug > 30 and args.debug:
            break
        i_debug += 1
        risk_label, years_last_followup = input['years_to_cancer'].cuda(), input['years_to_last_followup'].cuda()

        output = input_and_output(args, model, input, train=True)
        risk, emb, log_var = output['risk'], output['emb'], output['log_var']
        # compute loss
        loss = compute_losses(args, criterion, risk, risk_label, years_last_followup, emb, log_var, is_sto=args.use_sto)
        if output['loss'] is not None:
            loss += output['loss']

        if output['current_risk'] is not None:
            loss += (compute_losses(args, criterion, output['current_risk'], risk_label, years_last_followup,
                                    output['current_emb'], output['current_log_var'], is_sto=args.use_sto) * 0.2)

        if output['difference_risk'] is not None:
            loss += (compute_losses(args, criterion, output['difference_risk'], risk_label, years_last_followup,
                                    output['difference_emb'], output['difference_log_var'], is_sto=args.use_sto) * 0.2)

        if output['prior_risk'] is not None:
            loss += (compute_losses(args, criterion, output['prior_risk'], input['prior_years_to_cancer'].cuda(),
                                    input['prior_years_to_last_followup'].cuda(),
                                    output['prior_emb'], output['prior_log_var'], is_sto=args.use_sto) * 0.2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.cpu().data.numpy())

        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.3f}, '
            .format(epoch, args.epochs, optimizer.param_groups[0]['lr'], losses.avg, ))

    #     result = cal_mae_acc(risk, risk_label, years_last_followup, is_sto=args.use_sto, weights=args.time_to_events_weights)
    #     if result['count'] > 0:
    #         mae.update(result['mae'], n=result['count'])
    #         acc.update(result['acc'], n=result['count'])
    #
    #     total_num += data_loader.batch_size
    #     total_loss += loss.item() * data_loader.batch_size
    #
    #     if 'POE' in args.model_method and args.use_sto:
    #         risk = torch.mean(risk, dim=0)
    #     pred_risk_label = F.softmax(risk, dim=-1)
    #
    #     # pred_risk_label = F.softmax(risk, dim=-1)
    #     # if args.model_method == 'POE' and args.use_sto:
    #     #     pred_risk_label = torch.mean(pred_risk_label, dim=0)
    #     all_risk_probabilities.append(pred_risk_label.detach().cpu().numpy())
    #     all_risk_label.append(risk_label.cpu().numpy())
    #     all_followups.append(years_last_followup.cpu().numpy())
    #
    #     train_bar.set_description(
    #         'Train Epoch: [{}/{}], '
    #         'lr: {:.6f}, '
    #         'mae: {:.3f}, '
    #         'acc: {:.2f}%, '
    #         'Loss: {:.3f}, '
    #             .format(epoch, args.epochs,
    #                     optimizer.param_groups[0]['lr'],
    #                     mae.avg, acc.avg * 100,
    #                     losses.avg, ))
    #
    #     del img
    #     del input
    #     gc.collect()
    #
    # del train_bar
    # gc.collect()
    #
    # metrics_, _ = compute_auc_cindex(all_risk_probabilities, all_risk_label, all_followups, args.num_output_neurons, args.max_followup)
    #
    # try:
    #     logging.info('c-index is {:.4f} '.format(metrics_['c_index']))
    # except:
    #     logging.info('c-index is None')
    #
    # try:
    #     for i in range(args.max_followup):
    #         x = int(i + 1)
    #         logging.info('AUC {} Year is {:.4f} '
    #               .format(x,metrics_[x]))
    # except:
    #     for i in range(args.max_followup):
    #         x = int(i + 1)
    #         logging.info('AUC {} Year is None')
    #
    # logging.info('TraEpo:[{}/{}], '
    #              'lr:{:.4f}, '
    #              'TraLos:{:.2f}, '
    #              'mae:{:.2f}, '
    #              'acc:{:.2f}, '
    #              .format(
    #     epoch, args.epochs,
    #     optimizer.param_groups[0]['lr'],
    #     losses.avg,
    #     mae.avg, acc.avg * 100,
    #
    # ))
    #
    # return {
    #     'loss': losses.avg,
    #     'mae': mae.avg,
    #     'acc': acc.avg,
    #     'c_index': metrics_['c_index'],
    #     'metrics_': metrics_,
    #     'metrics': _
    # }
    return {'loss': losses.avg, 'mae': 0.0, 'acc': 0.0, 'c_index': 0.0, 'metrics_': None, 'metrics': None}


def direct_validate(model, valid_loader, criterion, args):
    use_sto = args.use_sto
    # use_sto = False

    model.eval()
    total_loss, total_num = 0.0, 0
    losses, mae, acc = AverageMeter(), AverageMeter(), AverageMeter()
    all_risk_probabilities, all_followups, all_risk_label = [], [], []

    with torch.no_grad():
        valid_bar = tqdm(valid_loader)
        i_debug = 0
        for input in valid_bar:
            if i_debug > 30 and args.debug:
                break
            i_debug += 1
            img, risk_label = input['img'].cuda(), input['years_to_cancer'].cuda()
            years_last_followup = input['years_to_last_followup'].cuda()

            output = input_and_output(args, model, input)
            risk, emb, log_var = output['risk'], output['emb'], output['log_var']

            # compute loss
            loss = compute_losses(args, criterion, risk, risk_label, years_last_followup, emb, log_var, is_sto=use_sto)
            if output['loss'] is not None:
                loss += output['loss']

            if output['current_risk'] is not None:
                loss += (compute_losses(args, criterion, output['current_risk'], risk_label, years_last_followup,
                                       output['current_emb'], output['current_log_var'], is_sto=args.use_sto) * 0.2)

            if output['difference_risk'] is not None:
                loss += (compute_losses(args, criterion, output['difference_risk'], risk_label, years_last_followup,
                                        output['difference_emb'], output['difference_log_var'],
                                        is_sto=args.use_sto) * 0.2)

            if output['prior_risk'] is not None:
                loss += (compute_losses(args, criterion, output['prior_risk'], input['prior_years_to_cancer'].cuda(),
                                       input['prior_years_to_last_followup'].cuda(),
                                       output['prior_emb'], output['prior_log_var'], is_sto=args.use_sto) * 0.2)

            result = cal_mae_acc(risk, risk_label, years_last_followup, is_sto=use_sto, weights=args.time_to_events_weights)

            losses.update(loss.cpu().data.numpy())
            if result['count'] > 0:
                mae.update(result['mae'], n=result['count'])
                acc.update(result['acc'], n=result['count'])

            total_num += valid_loader.batch_size
            total_loss += loss.item() * valid_loader.batch_size

            valid_bar.set_description('Valid MAE: {:.4f}, Valid ACC: {:.2f}%, Valid Loss: {:.4f}, '
                    .format(mae.avg, acc.avg*100, losses.avg, ))

            # pred_risk_label = F.softmax(risk, dim=-1)
            # if 'POE' in args.model_method and use_sto:
            #     pred_risk_label = torch.mean(pred_risk_label, dim=0)
            if 'POE' in args.model_method and use_sto:
                risk = torch.mean(risk, dim=0)
            pred_risk_label = F.softmax(risk, dim=-1)

            all_risk_probabilities.append(pred_risk_label.cpu().numpy())
            all_risk_label.append(risk_label.cpu().numpy())
            all_followups.append(years_last_followup.cpu().numpy())

        del img, input
        gc.collect()

    del valid_bar
    gc.collect()

    metrics_, _ = compute_auc_cindex(all_risk_probabilities, all_risk_label, all_followups, args.num_output_neurons,
                                    args.max_followup)

    try:
        logging.info('c-index is {:.4f} '.format(metrics_['c_index']))
    except:
        logging.info('c-index is None')

    try:
        for i in range(args.max_followup):
            x = int(i + 1)
            logging.info('AUC {} Year is {:.4f}'.format(x, metrics_[x]))
    except:
        for i in range(args.max_followup):
            x = int(i + 1)
            logging.info('AUC {} Year is None')

    logging.info('ValLos:{:.2f}, mae:{:.2f}, acc:{:.2f}%'.format(losses.avg, mae.avg, acc.avg * 100,))

    return {'loss': losses.avg, 'mae': mae.avg, 'acc': acc.avg, 'c_index': metrics_['c_index'],
            'metrics_': metrics_, 'metrics': _}


def direct_test(model, test_loader, criterion, args, save_pkl=None, **kwargs):
    use_sto = args.use_sto
    model.eval()
    total_loss, total_num = 0.0, 0
    losses, mae, acc = AverageMeter(), AverageMeter(), AverageMeter()

    all_risk_probabilities, all_followups, all_risk_label = [], [], []
    all_patient_ids, all_exam_ids, all_views, all_lateralitys = [], [], [], []

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        i_debug = 0
        for input in test_bar:
            if i_debug > 30 and args.debug:
                break
            i_debug += 1
            img, risk_label = input['img'].cuda(), input['years_to_cancer'].cuda()
            years_last_followup = input['years_to_last_followup'].cuda()

            output = input_and_output(args, model, input)
            if 'inference' in kwargs and kwargs['inference']:
                plot_attent(args, input, output, i_debug)

            risk, emb, log_var = output['risk'], output['emb'], output['log_var']
            # compute loss
            if 'inference' in kwargs and kwargs['inference']:
                loss = torch.tensor(0.0).cuda()
            else:
                loss = compute_losses(args, criterion, risk, risk_label, years_last_followup, emb, log_var, is_sto=use_sto)
                if output['loss'] is not None:
                    loss += output['loss']

                if output['current_risk'] is not None:
                    loss += (compute_losses(args, criterion, output['current_risk'], risk_label, years_last_followup,
                                           output['current_emb'], output['current_log_var'], is_sto=args.use_sto) * 0.2)

                if output['difference_risk'] is not None:
                    loss += (compute_losses(args, criterion, output['difference_risk'], risk_label, years_last_followup,
                                           output['difference_emb'], output['difference_log_var'],
                                           is_sto=args.use_sto) * 0.2)

                if output['prior_risk'] is not None:
                    loss += (compute_losses(args, criterion, output['prior_risk'], input['prior_years_to_cancer'].cuda(),
                                           input['prior_years_to_last_followup'].cuda(),
                                           output['prior_emb'], output['prior_log_var'], is_sto=args.use_sto) * 0.2)

            result = cal_mae_acc(risk, risk_label, years_last_followup, is_sto=use_sto, weights=args.time_to_events_weights)

            losses.update(loss.cpu().data.numpy())
            if result['count'] > 0:
                mae.update(result['mae'], n=result['count'])
                acc.update(result['acc'], n=result['count'])

            total_num += test_loader.batch_size
            total_loss += loss.item() * test_loader.batch_size

            test_bar.set_description('Test MAE: {:.4f}, Valid ACC: {:.2f}%, Valid Loss: {:.4f}, '
                                      .format(mae.avg, acc.avg*100, losses.avg, ))

            if 'POE' in args.model_method and use_sto:
                risk = torch.mean(risk, dim=0)
            pred_risk_label = F.softmax(risk, dim=-1)

            all_risk_probabilities.append(pred_risk_label.cpu().numpy())
            all_risk_label.append(risk_label.cpu().numpy())
            all_followups.append(years_last_followup.cpu().numpy())

            all_patient_ids.append(input['patient_id'])
            all_exam_ids.append(input['exam_id'])
            all_views.append(input['view'])
            all_lateralitys.append(input['laterality'])

        del img
        del input
        gc.collect()

    del test_bar
    gc.collect()

    metrics_, metrics = compute_auc_cindex(all_risk_probabilities, all_risk_label, all_followups, args.num_output_neurons,
                                    args.max_followup, confidence_interval=False)

    all_patient_ids = np.concatenate(all_patient_ids)
    all_exam_ids = np.concatenate(all_exam_ids)
    all_views = np.concatenate(all_views)
    all_lateralitys = np.concatenate(all_lateralitys)
    all_risk_probabilities = np.concatenate(all_risk_probabilities).reshape(-1, args.num_output_neurons)
    all_risk_label = np.concatenate(all_risk_label)
    all_followups = np.concatenate(all_followups)

    try:
        logging.info('c-index is {:.4f} '.format(metrics_['c_index']))
    except:
        logging.info('c-index is None')

    try:
        for i in range(args.max_followup):
            x = int(i + 1)
            logging.info('AUC {} Year is {:.4f}'.format(x, metrics_[x]))
    except:
        for i in range(args.max_followup):
            x = int(i + 1)
            logging.info('AUC {} Year is None')

    logging.info('TestLos:{:.2f}, mae:{:.2f}, acc:{:.2f}%,'.format(losses.avg, mae.avg, acc.avg * 100,))

    if save_pkl is not None:
        save_dict = {
            'patient_id': all_patient_ids, 'exam_id': all_exam_ids, 'view': all_views, 'laterality': all_lateralitys,
            'risk_probabilitie': all_risk_probabilities, 'risk_label': all_risk_label, 'followup': all_followups,}

        pickle.dump(save_dict, open('{}/result_{}.pkl'.format(args.results_dir, save_pkl), 'wb'))


    return {'loss': losses.avg, 'mae': mae.avg, 'acc': acc.avg, 'c_index': metrics_['c_index'], 'metrics_': metrics_,}


def get_train_val_test_demo():
    return direct_train, direct_validate, direct_test
