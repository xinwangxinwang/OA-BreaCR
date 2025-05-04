import os
import warnings
import argparse
import pickle
import json
import torch
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import resample
from lifelines import KaplanMeierFitter
import sklearn.metrics
from scipy.special import softmax
from utils.c_index import concordance_index


def get_censoring_dist(times, event_observed):
    # _dataset = train_dataset.dataset
    # times, event_observed = [d['time_at_event'] for d in _dataset], [d['y'] for d in _dataset]
    # times, event_observed = [d for d in times], [d for d in event_observed]
    times = list(times)
    all_observed_times = set(list(times))
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)

    censoring_dist = {time: kmf.predict(time) for time in all_observed_times}
    return censoring_dist


def compute_auc_metrics_given_curve(probs, censor_times, golds, max_followup, censor_distribution):
    metrics = {}
    sample_sizes = {}
    for followup in range(max_followup):
        min_followup_if_neg = followup + 1

        auc, golds_for_eval = compute_auc_x_year_auc(probs, censor_times, golds, followup)
        key = min_followup_if_neg
        metrics[key] = auc
        sample_sizes[key] = golds_for_eval
    try:
        c_index = concordance_index(censor_times, probs, golds, censor_distribution)
    except Exception as e:
            warnings.warn("Failed to calculate C-index because {}".format(e))
            c_index = 'NA'

    metrics['c_index'] = c_index
    end_probs = np.array(probs)[:,-1].tolist()
    sorted_golds = [g for p,g in sorted( zip(end_probs, golds))]
    metrics['decile_recall'] = sum( sorted_golds[-len(sorted_golds)//10:]) / sum(sorted_golds)
    return metrics, sample_sizes


def compute_auc_x_year_auc(probs, censor_times, golds, followup):

    def include_exam_and_determine_label( prob_arr, censor_time, gold):
        valid_pos = gold and censor_time <= followup
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        return included, label

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        include, label = include_exam_and_determine_label(prob_arr, censor_time, gold)
        if include:
            probs_for_eval.append(prob_arr[followup])
            golds_for_eval.append(label)

    try:
        auc = sklearn.metrics.roc_auc_score(golds_for_eval, probs_for_eval, average='samples')
    except Exception as e:
        warnings.warn("Failed to calculate AUC because {}".format(e))
        auc = 'NA'

    return auc, golds_for_eval


def compute_mean_ci_(c_index_bs):
    c_index_bs = np.array(c_index_bs)

    # get mean
    c_index_bs_mean = np.mean(c_index_bs)
    # get median
    c_index_bs_median = np.percentile(c_index_bs, 50)
    # get 95% interval
    alpha = 100 - 95
    c_index_bs_lower_ci = np.percentile(c_index_bs, alpha / 2)
    c_index_bs_upper_ci = np.percentile(c_index_bs, 100 - alpha / 2)
    return c_index_bs_mean, c_index_bs_lower_ci, c_index_bs_upper_ci, c_index_bs_median


def compute_yala_metrics_with_CI(probs, golds, censor_times, max_followup=5):
    n_iterations = 1000
    metrics = {
        'c_index':[],
        # 'AUC1':[],
        # 'AUC2':[],
        # 'AUC3':[],
        # 'AUC4':[],
        # 'AUC5':[],
    }
    for year_ in range(max_followup):
        dict_add = {'AUC{:.0f}'.format(year_ + 1): []}
        metrics.update(dict_add)
    # c_indexs = []
    # AUCs = []
    for i in range(n_iterations):
        print('n_iterations: {}'.format(i + 1))
        probs_bs, golds_bs, censor_times_bs = resample(probs, golds, censor_times, replace=True)
        censor_distribution_bs = get_censoring_dist(censor_times_bs, golds_bs)
        # c_index = concordance_index(censor_times_bs, probs_bs, golds_bs, censor_distribution_bs)
        metrics_, sample_sizes_ = compute_auc_metrics_given_curve(
            probs_bs, censor_times_bs, golds_bs, max_followup, censor_distribution_bs)

        # c_index = metrics_['c_index']
        metrics['c_index'].append(metrics_['c_index'])
        for year_ in range(max_followup):
            metrics['AUC{}'.format(year_ + 1)].append(metrics_[(year_ + 1)])
        # metrics['AUC1'].append(metrics_[1])
        # metrics['AUC2'].append(metrics_[2])
        # metrics['AUC3'].append(metrics_[3])
        # metrics['AUC4'].append(metrics_[4])
        # metrics['AUC5'].append(metrics_[5])
        # AUCs.append()
    try:
        metrics['c_index'] = compute_mean_ci_(metrics['c_index'])
    except Exception as e:
        warnings.warn("Failed to calculate c_index ci because {}".format(e))
        metrics['c_index'] = 'NA'
    metrics['c_index'] = compute_mean_ci_(metrics['c_index'])
    for year_ in range(max_followup):
        try:
            metrics['AUC{}'.format(year_ + 1)] = compute_mean_ci_(metrics['AUC{}'.format(year_ + 1)])
        except Exception as e:
            warnings.warn("Failed to calculate AUC ci because {}".format(e))
            metrics['AUC{}'.format(year_ + 1)] = 'NA'

    # metrics['AUC1'] = compute_mean_ci_(metrics['AUC1'])
    # metrics['AUC2'] = compute_mean_ci_(metrics['AUC2'])
    # metrics['AUC3'] = compute_mean_ci_(metrics['AUC3'])
    # metrics['AUC4'] = compute_mean_ci_(metrics['AUC4'])
    # metrics['AUC5'] = compute_mean_ci_(metrics['AUC5'])

    return metrics


def prob_to_score(prob, max_followup=5):
    # print('prob')
    # for i in range(15):
    score = np.zeros_like(prob)[:, 0:max_followup]
    for i in range(max_followup):
        # i_ = -(i + 1)
        # score[:, i] = prob[:, i_]
        for i_in in range(i+1):
            i_ = i_in
            # i_ = -(i_in + 1)
            score[:, i] += prob[:, i_]
    return score


# def prob_to_score(prob_, max_followup=5):
#     prob = prob_.copy()
#     score = prob
#     for i in range(max_followup-1):
#         score = np.concatenate([score, prob], 1)
#     return score


def get_censor_info(labels, followups, max_followup=5):
    labels = np.squeeze(labels)
    followups = np.squeeze(followups)
    #
    years_to_cancer = labels
    years_to_last_followup = followups + 1

    any_cancer = years_to_cancer < max_followup
    # cancer_key = "years_to_cancer"

    y = any_cancer
    shape = np.shape(any_cancer)[0]
    y_seq = np.zeros([shape, max_followup])

    time_at_event = np.zeros_like(years_to_cancer)
    y_mask = np.zeros_like(y_seq)

    for i in range(shape):
        if y[i]:
            time_at_event[i] = int(years_to_cancer[i])
            y_seq[i, time_at_event[i]:] = 1
        else:
            time_at_event[i] = int(min(years_to_last_followup[i], max_followup) - 1)

        # y_mask[i, :] = np.array([1] * (time_at_event+1) + [0]* (max_followup - (time_at_event+1)))
        y_mask[i, :] = np.array([1] * (time_at_event[i] +1) + [0]* (max_followup - (time_at_event[i] +1)))
    # y_mask = np.array([1] * (time_at_event+1) + [0]* (max_followup - (time_at_event+1)))
    # assert len(y_mask) == max_followup
    return any_cancer, y_seq.astype('float64'), y_mask.astype('float64'), time_at_event


def compute_auc_cindex(
        all_probabilities,
        all_labels,
        all_followups,
        num_classes,
        max_followup,
        confidence_interval=False
):


    all_labels_np = np.array(all_labels)
    all_probabilities_np = np.array(all_probabilities)
    all_followups_np = np.array(all_followups)
    all_labels_np = all_labels_np.reshape(-1, 1)
    all_probabilities_np = all_probabilities_np.reshape(-1, num_classes)
    all_followups_np = all_followups_np.reshape(-1, 1)

    score = prob_to_score(all_probabilities_np, max_followup=max_followup)
    y, y_seq, y_mask, time_at_event = get_censor_info(
        all_labels_np, all_followups_np, max_followup=max_followup)
    censor_distribution = get_censoring_dist(time_at_event, y)

    probs = score
    censor_times = time_at_event
    golds = y
    censor_distribution = censor_distribution

    metrics_, _ = compute_auc_metrics_given_curve(probs, censor_times, golds, max_followup, censor_distribution)
    if confidence_interval:
        metrics = compute_yala_metrics_with_CI(probs, golds, censor_times, max_followup=max_followup)
    else:
        metrics = None

    # print('c-index is {:.4f} '
    #       # 'Bootstrap mean c-index is {:.4f} [{:.4f}, {:.4f}]'
    #     .format(
    #     metrics_['c_index'],
    #     # metrics['c_index'][0], metrics['c_index'][1], metrics['c_index'][2]
    # ))
    #
    # for i in range(max_followup):
    #     x = int(i + 1)
    #     print('AUC {} Year is {:.4f} '
    #           # 'Bootstrap mean AUC is {:.4f} [{:.4f}, {:.4f}]'
    #           .format(x,
    #                   metrics_[x],
    #                   # metrics['AUC{}'.format(x)][0],
    #                   # metrics['AUC{}'.format(x)][1],
    #                   # metrics['AUC{}'.format(x)][2]
    #                   ))

    return metrics_, metrics


def compute_auc_cindex_with_scores(
        score,
        all_labels_np,
        all_followups_np,
        max_followup,
        confidence_interval=False
):

    y, y_seq, y_mask, time_at_event = get_censor_info(
        all_labels_np, all_followups_np, max_followup=max_followup)
    censor_distribution = get_censoring_dist(time_at_event, y)

    probs = score
    censor_times = time_at_event
    golds = y
    censor_distribution = censor_distribution

    metrics_, _ = compute_auc_metrics_given_curve(probs, censor_times, golds, max_followup, censor_distribution)
    if confidence_interval:
        metrics = compute_yala_metrics_with_CI(probs, golds, censor_times, max_followup=max_followup)
    else:
        metrics = None

    # print('c-index is {:.4f} '
    #       # 'Bootstrap mean c-index is {:.4f} [{:.4f}, {:.4f}]'
    #     .format(
    #     metrics_['c_index'],
    #     # metrics['c_index'][0], metrics['c_index'][1], metrics['c_index'][2]
    # ))
    #
    # for i in range(max_followup):
    #     x = int(i + 1)
    #     print('AUC {} Year is {:.4f} '
    #           # 'Bootstrap mean AUC is {:.4f} [{:.4f}, {:.4f}]'
    #           .format(x,
    #                   metrics_[x],
    #                   # metrics['AUC{}'.format(x)][0],
    #                   # metrics['AUC{}'.format(x)][1],
    #                   # metrics['AUC{}'.format(x)][2]
    #                   ))

    return metrics_, metrics
