import os
import cv2
import logging
import torch
import datetime
from dateutil import parser
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.transforms import v2 as transforms
from sklearn.utils.class_weight import compute_class_weight

# Deprecated, because I moved some of the data augmentation to the GPU by "learning.OA_learning_demo.py"
# additional_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])
#
# additional_transform_ = transforms.Compose([
#     transforms.RandomRotation(10),
#     transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),
#     # transforms.CenterCrop(args.img_size),
# ])

additional_transform_mask = transforms.Compose([transforms.ToTensor(),])


def imgunit8(img):
    mammogram_scaled = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    mammogram_uint8_by_function = mammogram_scaled.astype(np.uint8)
    return mammogram_uint8_by_function


def imgunit16(img):
    # min-max normalization & covert to unit16
    mammogram_dicom = img
    orig_min = mammogram_dicom.min()
    orig_max = mammogram_dicom.max()
    target_min = 0.0
    target_max = 65535.0
    mammogram_scaled = (mammogram_dicom-orig_min)*((target_max-target_min)/(orig_max-orig_min))+target_min
    mammogram_uint8_by_function = mammogram_scaled.astype(np.uint16)
    return mammogram_uint8_by_function


def old_transform_method(args):
    # Data Aug
    # 1 Resized or Random Resized Crop (check)
    Resized = transforms.Resize(args.img_size)
    # RandomResizedCrop = transforms.RandomResizedCrop(args.img_size, scale=(0.5, 1.))
    # 2 Random Horizontal Flip
    RandomHorizontalFlip = transforms.RandomHorizontalFlip(p=0.5)
    # 3 Random Vertical Flip
    RandomVerticalFlip = transforms.RandomVerticalFlip(p=0.5)
    # 4 ColorJitter
    ColorJitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0)
    # # 5 Gaussian blur
    # Gaussianblur = GaussianBlur([.1, 2.])
    # # 6 Noise
    # Noise = transforms.Compose([
    #     ImgAugGaussianNoiseTransform(),
    #     lambda x: Image.fromarray(x)
    # ])
    # # 7 Gamma Correction
    # GammaCorrection = transforms.Compose([
    #     ImgAugGammaCorrectionTransform(),
    #     lambda x: Image.fromarray(x)
    # ])
    # # 8 Elastic
    # Elastic = transforms.Compose([
    #     ImgAugElasticTransform(),
    #     lambda x: Image.fromarray(x)
    # ])
    # 9 Random Rotation
    RandomRotation = transforms.RandomRotation(10)

    train_transform = transforms.Compose([
        Resized,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        # transforms.RandomApply([ColorJitter], p=0.5),  # important
        # # added more augument
        # transforms.RandomApply([Gaussianblur], p=0.5),
        # transforms.RandomApply([Noise], p=0.5),
        # transforms.RandomApply([GammaCorrection], p=0.5),
        # transforms.RandomApply([Elastic], p=0.5),
        #
        # RandomRotation,
        # transforms.CenterCrop(args.img_size),
        # transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
        # transforms.ToPILImage()
    ])

    test_transform = transforms.Compose([
        Resized,
        # transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
    ])

    return train_transform, test_transform


class Risk_Mammo_Dataset(Dataset):
    def __init__(self, args, data_info, file_path, transform, train=False):
        data_info['density'][data_info['density'].isna()] = -1
        data_info['birads'][data_info['birads'].isna()] = -1
        data_info['age'][data_info['age'].isna()] = -1
        data_info['age'][data_info['age'] == 'none'] = -1

        self.file_path = file_path
        self.data_info = data_info
        self.img_size = args.img_size
        self.args = args
        years_to_cancer = np.asarray(data_info['years_to_cancer'], dtype='int64')
        self.years_to_cancer = np.asarray(years_to_cancer, dtype='int64')
        years_to_last_followup = np.asarray(data_info['years_to_last_followup'], dtype='int64')
        self.years_to_last_followup = np.asarray(years_to_last_followup, dtype='int64')

        data_index = list(range(len(data_info)))
        self.data_index = data_index
        self.data_info['data_index'] = self.data_index

        event_labels = args.max_followup > years_to_cancer
        self.event_labels = event_labels.astype(int)
        self.data_info['event_labels'] = event_labels
        time_to_events = years_to_cancer.copy()
        time_to_events[time_to_events >= args.max_followup] = args.max_followup
        self.time_to_events = time_to_events

        self.data_info['time_to_events'] = self.time_to_events

        image_file_path = np.asarray(data_info['file_path'])
        self.img_file_path = image_file_path
        self.transform = transform
        self.train = train
        if not train:
            self.history_selct_method = 'last'
        else:
            self.history_selct_method = 'random'

    def __getitem__(self, index):
        data_index = self.data_index[index]
        labels = self.__getlabel(data_index)
        img = self.__getimg(labels['path'], labels['laterality'], prior=False)

        prior_index, prior_img, mask, prior_mask, gap = None, None, None, None, None

        if 'no_prior' in self.args and self.args.no_prior:
            if 'mask' in self.args and self.args.mask:
                mask = self.__getmask(img, labels['path'], labels['laterality'], prior=False)
        else:
            prior_index = self.__get_prior_index(data_index)
            prior_labels = self.__getlabel(prior_index)
            # exam_date = datetime.datetime.strptime(str(labels['exam_date']), '%Y/%m/%d')
            exam_date = parser.parse(str(labels['exam_date']))
            prior_exam_date = parser.parse(str(prior_labels['exam_date']))
            # prior_exam_date = datetime.datetime.strptime(str(prior_labels['exam_date']), '%Y/%m/%d')
            gap = (exam_date - prior_exam_date).days / 30

            if prior_index != data_index:
                prior_labels = self.__getlabel(prior_index)
                prior_img = self.__getimg(prior_labels['path'], prior_labels['laterality'], prior=True)
                if 'mask' in self.args and self.args.mask:
                    prior_mask = self.__getmask(prior_img, prior_labels['path'], prior_labels['laterality'], prior=True)
            else:
                prior_labels = labels.copy()
                prior_img = img.copy()
                if 'mask' in self.args and self.args.mask:
                    prior_mask = mask

        if prior_img is None and mask is None:
            if self.transform is not None:
                img = self.transform(img)
                img = additional_transform_mask(img)

        elif prior_img is None and mask is not None:
            if self.transform is not None:
                img, mask = self.transform(img, mask)
                img = additional_transform_mask(img)
                mask = additional_transform_mask(mask)

        elif prior_img is not None and mask is None:
            if self.transform is not None:
                img, prior_img = self.transform(img, prior_img)
                img = additional_transform_mask(img)
                prior_img = additional_transform_mask(prior_img)

        elif prior_img is not None and mask is not None:
            if self.transform is not None:
                img, prior_img, mask, prior_mask = self.transform(img, prior_img, mask, prior_mask)
                img = additional_transform_mask(img)
                prior_img = additional_transform_mask(prior_img)
                mask = additional_transform_mask(mask)
                prior_mask = additional_transform_mask(prior_mask)
        else:
            raise logging.info('Data mode are not supported')

        data_dict = {
            'patient_id': labels['patient_id'],
            'exam_id': labels['exam_id'],
            'exam_date': labels['exam_date'],
            'view': labels['view'],
            'laterality': labels['laterality'],
            'img': img,
            'event_label': torch.as_tensor(labels['event_label']),
            'time_to_event': torch.as_tensor(labels['time_to_event']),
            'years_to_cancer': torch.as_tensor(labels['years_to_cancer']),
            'years_to_last_followup': torch.as_tensor(labels['years_to_last_followup']),
            'density': torch.as_tensor(labels['density']),
            'birads': torch.as_tensor(labels['birads']),
            'age': torch.as_tensor(labels['age']),
        }
        if mask is not None:
            data_dict['mask'] = mask

        if prior_img is not None:
            data_dict['prior_exam_id'] = prior_labels['exam_id']
            data_dict['prior_exam_date'] = prior_labels['exam_date']
            data_dict['prior_img'] = prior_img
            data_dict['prior_event_label'] = torch.as_tensor(prior_labels['event_label'])
            data_dict['prior_time_to_event'] = torch.as_tensor(labels['time_to_event'])
            data_dict['prior_years_to_cancer'] = torch.as_tensor(prior_labels['years_to_cancer'])
            data_dict['prior_years_to_last_followup'] = torch.as_tensor(prior_labels['years_to_last_followup'])
            data_dict['prior_density'] = torch.as_tensor(prior_labels['density'])
            data_dict['prior_birads'] = torch.as_tensor(prior_labels['birads'])
            data_dict['prior_age'] = torch.as_tensor(prior_labels['age'])
            data_dict['gap'] = torch.as_tensor(gap)
            if prior_mask is not None:
                data_dict['prior_mask'] = prior_mask

        return data_dict

    def __len__(self):
        return len(self.data_index)

    # Get the all labels of the image
    def __getlabel(self, index):
        # image_info = self.data_info.iloc[index, :]
        image_info = self.data_info[self.data_info['data_index'] == index]
        patient_id = image_info['patient_id'].iloc[0]
        exam_id = image_info['exam_id'].iloc[0]
        exam_date = image_info['exam_date'].iloc[0]
        view = image_info['view'].iloc[0]
        path = image_info['file_path'].iloc[0]
        laterality = image_info['laterality'].iloc[0]
        density = image_info['density'].iloc[0]
        birads = image_info['birads'].iloc[0]
        age = image_info['age'].iloc[0]
        event_label = image_info['event_labels'].iloc[0]
        time_to_event = image_info['time_to_events'].iloc[0]
        years_to_cancer = image_info['years_to_cancer'].iloc[0]
        years_to_last_followup = image_info['years_to_last_followup'].iloc[0]
        return {
            'patient_id': str(patient_id),
            'exam_id': str(exam_id),
            'exam_date': str(exam_date),
            'view': str(view),
            'path': str(path),
            'laterality': str(laterality),
            'event_label': int(event_label),
            'time_to_event': int(time_to_event),
            'years_to_cancer': int(years_to_cancer),
            'years_to_last_followup': int(years_to_last_followup),
            'density': int(density),
            'birads': int(birads),
            'age': float(age),
        }

    # Get the prior image index on the CSV file to load the prior image
    def __get_prior_index(self, index):
        self.data_info['exam_date'] = pd.to_datetime(self.data_info['exam_date'])

        image_info = self.data_info[self.data_info['data_index'] == index]
        patient_id = image_info['patient_id'].iloc[0]
        exam_id = image_info['exam_id'].iloc[0]
        exam_date = image_info['exam_date'].iloc[0]
        view = image_info['view'].iloc[0]
        laterality = image_info['laterality'].iloc[0]

        patient_data_info_history = self.data_info[
            (self.data_info['patient_id'] == patient_id) &
            (self.data_info['view'] == view) &
            (self.data_info['laterality'] == laterality)
            ]

        history_df = patient_data_info_history[patient_data_info_history['exam_date'] < exam_date]

        if not history_df.empty:
            history_df = history_df.sort_values(by='exam_date', ascending=True)
            if self.history_selct_method == 'last':
                selected = history_df.iloc[-1]
            else:
                selected = history_df.sample(n=1).iloc[0]
            prior_index = selected['data_index']
        else:
            prior_index = index
        return prior_index

    # Load the mammogram image
    def __getimg(self, file_path, laterality, prior=False):
        img_path = file_path
        # image = Image.open(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if image is not None:
            if "R" in laterality:
                image = cv2.flip(image, 1)
            image = imgunit16(image) / 65535.0
            image = Image.fromarray(image)
        else:
            logging.info('{} wrong!!!'.format(img_path))
            image = None

        return image

    # Acturally, I don't need to load the mask in this study
    def __getmask(self, img, file_path, laterality, prior=False):
        w, h = img.size
        img_path = file_path
        img_path = str(img_path).replace('img-2048', 'img-2048-mask')
        try:
            image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            image = imgunit8(image)
        except:
            image = None

        if image is not None:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
            if "R" in laterality:
                image = cv2.flip(image, 1)
            # image = image / 255.0
            image = Image.fromarray(image)
        else:
            # logging.info('{} wrong!!!'.format(img_path))
            image = img
            image = imgunit8(np.array(image))
            image = Image.fromarray(image)

        return image


def dataloador(data_info, args, train=False, predict=False, val_shuffle=False):

    train_transform, test_transform = old_transform_method(args)
    if train:
        transform = train_transform
        shuffle = True if args.batch_size != 1 else False
        batch_size = args.batch_size
    else:
        transform = test_transform
        shuffle = val_shuffle
        batch_size = args.batch_size
    dataset = Risk_Mammo_Dataset(args, data_info, args.image_dir, transform, train=train)

    weights_time_to_events = compute_class_weight(
        class_weight="balanced", classes=np.unique(dataset.time_to_events), y=dataset.time_to_events
    )

    class_time_to_events_counts = np.bincount(dataset.time_to_events)
    # weights_time_to_events = 1 / class_time_to_events_counts
    # weights_time_to_events = weights_time_to_events / min(weights_time_to_events)
    logging.info(f'time_to_events_counts= {class_time_to_events_counts}')
    logging.info(f'weights= {weights_time_to_events.tolist()}')
    if train and 'weight_class_loss' in args and args.weight_class_loss:
        args.time_to_events_weights = weights_time_to_events.tolist()

    class_sample_counts = np.bincount(dataset.event_labels)
    # print(class_sample_counts)
    classcount = class_sample_counts.tolist()
    if args.balance_training:
        weights = 1. / torch.tensor(classcount, dtype=torch.float)
        sampleweights = weights[dataset.event_labels]
        data_sampler = WeightedRandomSampler(sampleweights, len(dataset), replacement=True)
        data_loader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=False, drop_last=True)

    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=args.num_workers, pin_memory=True, prefetch_factor=2, drop_last=True)

    if predict:
        args.time_to_events_weights = weights_time_to_events.tolist()
    return data_loader
