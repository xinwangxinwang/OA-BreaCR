import os
import torch
import logging
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def arg_parse():
    parser = argparse.ArgumentParser(description='OA-BreaCR Model')

    parser.add_argument('-a', '--arch', default='resnet18',
                        help='resnet18, resnet50, densenet121, densenet169, vgg16, vgg19,'
                             'convnext_tiny, convnext_small, vit_b_16, regnet_x_8gf')

    parser.add_argument('--debug', action='store_true',
                        # default=True,
                        help='Quick setting params for debugging')

    parser.add_argument('--random_seed', type=int, default=42,
                        help='Set seed of the experiment')

    # parser.add_argument('--rich_transform', action='store_true',
    #                     # default=True,
    #                     help='<True> means more data augmentation methods from 1st of RSNA 2023 kaggle challenge')

    parser.add_argument('--balance_training', action='store_true',
                        # default=True,
                        help='Balance pos and neg sample for training')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='Number of total epochs to run')

    parser.add_argument('--batch-size', default=6, type=int, metavar='N',
                        help='Mini-batch size')

    parser.add_argument('--optimizer', default='Adam', type=str,
                        help='optimizer')

    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='Initial learning rate', dest='lr')

    # parser.add_argument('--schedule',
    #                     # default=[20, 40, 60, 80],
    #                     default=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    #                     nargs='*', type=int,
    #                     help='learning rate schedule (when to drop lr by a ratio)')
    # parser.add_argument('--cos', action='store_true',
    #                     # default=True,
    #                     help='use cosine lr schedule')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum')

    parser.add_argument('--wd', '--weight-decay', default=0., type=float, metavar='W',
                        help='Weight decay (default: 0.)', dest='weight_decay')

    parser.add_argument('--weight_bce', default=5.0, type=float, metavar='M',
                        help='Weights of the custom bce loss function')

    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='Manual epoch number (useful on restarts)')

    parser.add_argument('--results-dir',
                        default='./Path/to/logs/',
                        type=str, metavar='PATH', help='Path to cache (default: none)')

    parser.add_argument('--resume',
                        default='',
                        type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')

    parser.add_argument('--csv-dir',
                        # default='/Path/to/data_info',
                        type=str, metavar='PATH', help='Path to csv (default: none)')

    parser.add_argument('--image-dir',
                        # default='/Path/to/image',
                        type=str, metavar='PATH', help='Path to image data (default: none)')

    parser.add_argument('--num-workers', default='12', type=int, metavar='N',
                        help='Number of data loading workers (default: 16)')

    parser.add_argument('--max_followup', default='5', type=int, metavar='N',
                        help='')

    parser.add_argument('--img-size', type=int, nargs='+', default=[1024, 512],
                        help='Height and width of image in pixels. [default: [256,256]')

    parser.add_argument('--model-method', default='OA-BreaCR', type=str,
                        help='Only support OA-BreaCR')

    # parser.add_argument('--attention',
    #                     default=None,
    #                     # default='fa', type=str,
    #                     help='fa , cmab')

    # parser.add_argument('--mask', action='store_true',
    #                     # default=True,
    #                     default=False,
    #                     help='Use mirai pretrained encoder (resnet18)')

    # parser.add_argument('--screening_clean', action='store_true',
    #                     # default=True,
    #                     # default=False,
    #                     help='Use mirai pretrained encoder (resnet18)')

    parser.add_argument('--no_prior', action='store_true',
                        help='load prior mammogram for the MTP learning, '
                             'could set Ture to skip loading prior images for saving loading time')

    parser.add_argument('--num-output-neurons', type=int, default=6,
                        help='number of ouput neurons of your model, it should be [max_followup+1]')

    parser.add_argument('--start_label', type=int, default=0,
                        help="start label of the dataset for ordinal learning, it should be [0, 1, 2, 3, 4, 5]')")

    # For POE model
    # ---------------------------------
    parser.add_argument('--max-t', type=int, default=50,
                        help='number of samples during sto.')
    parser.add_argument('--no-sto', action='store_true',
                        help='not using stochastic sampling when training or testing.')
    parser.add_argument('--distance', type=str, default='JDistance',
                        help='distance metric between two gaussian distribution')
    parser.add_argument('--alpha-coeff', type=float, default=1e-5, metavar='M',
                        help='alpha_coeff (default: 0)')
    parser.add_argument('--beta-coeff', type=float, default=1e-4, metavar='M',
                        help='beta_coeff (default: 1.0)')
    parser.add_argument('--margin', type=float, default=2, metavar='M',
                        help='margin (default: 1.0)')
    # ---------------------------------

    parser.add_argument('--early_stop_patient', default=15, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr_decay_patient', default=3, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--time_to_events_weights', default=None,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--weight_class_loss', action='store_true',
                        # default=True,
                        help='Balance pos and neg sample for training')

    parser.add_argument('--accumulation_steps',
                        default=1,
                        type=int, metavar='N',
                        help='Accumulate gradients over multiple mini-batches')

    parser.add_argument('--training_step',
                        default=1000,
                        type=int, metavar='N',
                        help='Accumulate gradients over multiple mini-batches')

    parser.add_argument('--val_step',
                        default=1000,
                        type=int, metavar='N',
                        help='Accumulate gradients over multiple mini-batches')

    parser.add_argument('--test_step',
                        default=1000,
                        type=int, metavar='N',
                        help='Accumulate gradients over multiple mini-batches')

    args = parser.parse_args()

    args.results_dir = args.results_dir + str(args.arch) + '_' + str(args.model_method) + '_' \
                       + str(args.lr) + '_lr_' \
                       + str(args.epochs) + '_epochs_' \
                       + str(args.batch_size) + '_bs_' \
                       + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'

    os.makedirs(args.results_dir, exist_ok=True)
    print(args.results_dir)
    args.use_sto = True
    return args


def get_criterion(args): # define loss function
    from losses.risk_bce_loss import risk_BCE_loss
    from losses.POEloss import ProbOrdiLoss
    from losses.mean_variance_loss import MeanVarianceLoss
    criterion_BCE = risk_BCE_loss(weight_loss=args.weight_bce, batch_size=args.batch_size,
                                  num_pred_years=args.num_output_neurons).cuda()
    criterion_POE = ProbOrdiLoss(distance=args.distance, alpha_coeff=args.alpha_coeff,
                                 beta_coeff=args.beta_coeff, margin=args.margin,
                                 main_loss_type='cls', criterion='l1',
                                 start_label=args.start_label).cuda()
    criterion_MV = MeanVarianceLoss(cumpet_ce_loss=False, start_age=args.start_label).cuda()
    criterion = {
        'criterion_BCE': criterion_BCE,
        'criterion_MV': criterion_MV,
        'criterion_POE': criterion_POE
    }
    return criterion


def get_model(args): # define model
    if args.model_method in ['OA-BreaCR']:
        from models.OA_Risk_model import OA_BreaCR
        model = OA_BreaCR(args, mtp=True).cuda()
    else:
        raise NotImplementedError

    logging.info(model)
    return model


def get_learning_demo(args): # define train val test demo
    if args.model_method in ['OA-BreaCR']:
        from learning.OA_BreaCR_learning_demo import get_train_val_test_demo
    else:
        raise NotImplementedError

    train, validate, test = get_train_val_test_demo()
    return train, validate, test


def get_dataset(args):
    if args.model_method in ['OA-BreaCR',]:
        import dataload.MTP_SV_dataset as mammo_dataset
    else:
        raise NotImplementedError

    return mammo_dataset


def get_optimizer(args, model): # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'Adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    return optimizer
