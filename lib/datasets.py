import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy import stats
from collections import defaultdict

from lib.transforms import lowfreqnoise

import torch
from torch.utils.data import Dataset

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def collate_longest_in_batch(batch):
    longest_ecg = max(item[0].shape[1] for item in batch)
    ecg_channels = batch[0][0].shape[0]

    ecgs_new = torch.zeros((len(batch), ecg_channels, longest_ecg))
    labels = torch.tensor([item[1] for item in batch]).long()
    names = [item[2] for item in batch]

    for i, (ecgraw, label, npypath) in enumerate(batch):
        ecgs_new[i, :, :ecgraw.shape[1]] = ecgraw

    return ecgs_new, labels, names


class ECGDataset(Dataset):
    def __init__(self, cfg, train_or_test, fold, final_test=False):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.fold = fold
        self.final_test = final_test

        self.paceddata_path = cfg['data']['paceddata_path']
        self.intrinsicdata_path = cfg['data']['intrinsicdata_path']
        self.max_samples_per_beattype = cfg['data']['max_samples_per_beattype']
        self.stack_intrinsic = cfg['data']['stack_intrinsic']
        self.beat_types = cfg['data']['beat_types'][self.train_or_test]
        self.n_folds = cfg['data']['n_folds']
        self.excluded_folds = cfg['data']['excluded_folds']
        self.df = pd.read_excel(cfg['data']['csv_path'], sheet_name='baselinedata_JH', skiprows=1)
        self.median, self.iqr = cfg['data']['median'], cfg['data']['iqr']

        self.cases = self.get_cases()
        self.traces = self.get_traces()

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        npypath = self.traces[idx]
        beat_type = npypath.split('_')[3]
        assert beat_type in self.beat_types
        beat_type = self.beat_types[beat_type]

        # ECG
        ecg = np.load(npypath)
        ecg = ecg - ecg[0]

        # Normalize
        if self.median and self.iqr:
            ecg = ecg - self.median
            ecg = ecg / self.iqr
        else:
            print(f"WARNING: Not normalising ECG")

        ecg = ecg.transpose((1, 0))  # Channels first

        if self.stack_intrinsic:
            intrinsicstudyfolder = os.path.join(self.intrinsicdata_path, os.path.basename(os.path.dirname(npypath)))
            intrinsicnpypaths = sorted(glob(os.path.join(intrinsicstudyfolder, "*.npy")))
            try:
                if self.train_or_test == 'train':
                    intrinsicnpypath = random.choice(intrinsicnpypaths)
                else:
                    intrinsicnpypath = intrinsicnpypaths[0]
                intrinsic_ecg = np.load(intrinsicnpypath)
            except IndexError:
                raise ValueError(f"Failed to load intrinsic NPY file for study {intrinsicstudyfolder}: {intrinsicnpypaths}")
            try:
                intrinsic_ecg = intrinsic_ecg - intrinsic_ecg[0]
            except IndexError:
                raise ValueError(f"Error with file {intrinsicnpypath}: of shape: {intrinsic_ecg.shape}")
            if self.median and self.iqr:
                intrinsic_ecg = intrinsic_ecg - self.median
                intrinsic_ecg = intrinsic_ecg / self.iqr
            intrinsic_ecg = intrinsic_ecg.transpose((1, 0))  # Channels first

            intrinsic_ecg_shaped = np.zeros_like(ecg)

            # Crop/0-pad intrinsic ECG to same length as paced ECG
            min_len = min(intrinsic_ecg.shape[1], ecg.shape[1])
            intrinsic_ecg_shaped[:, :min_len] = intrinsic_ecg[:, :min_len]

            ecg = np.concatenate((ecg, intrinsic_ecg_shaped), axis=0)

        # Transforms
        ecg = self.transform_ecg(ecg)

        # Label
        label = list(self.beat_types.values()).index(beat_type)

        x = torch.tensor(ecg).float()
        y = torch.tensor(label).long()

        return x, y, npypath

    def transform_ecg(self, ecg):
        cfg_transforms = self.cfg['transforms'][self.train_or_test]
        if not cfg_transforms:
            return ecg

        if cfg_transforms.get('lowfreqnoise', False):
            min_freq, max_freq = cfg_transforms['lowfreqnoise']['min_freq'], cfg_transforms['lowfreqnoise']['max_freq']
            ecg = lowfreqnoise(ecg, min_freq, max_freq)

        return ecg

    def get_cases(self):
        def get_train_test_exclude_for_case(case):
            patient_root = case[:4]  # H036b -> H036
            random.seed(patient_root)
            assert 1 <= self.fold <= self.n_folds, f"Fold should be between 1 and {self.n_folds}, not {self.fold}"
            test_fold = random.randint(1, self.n_folds)
            if test_fold == self.fold:
                return 'test'
            elif test_fold in self.excluded_folds:
                return 'exclude'
            else:
                return 'train'

        cases_csv = set(self.df.code)
        cases_dir = os.listdir(self.paceddata_path)

        cases_both = [p for p in cases_dir if p[:4] in cases_csv]
        missing_csv = [p for p in cases_dir if p[:4] not in cases_csv]
        missing_dir = [p for p in cases_csv if p not in cases_dir]

        if self.stack_intrinsic:
            cases_intrinsic_dir = os.listdir(self.intrinsicdata_path)
            missing_intrinsic = [p for p in cases_both if p not in cases_intrinsic_dir]
            cases_both = [p for p in cases_both if p in cases_intrinsic_dir]

        cases = [p for p in cases_both if get_train_test_exclude_for_case(p) == self.train_or_test]
        print(
            f"{self.train_or_test.upper() + ':': <6} {len(cases)} of {len(cases_both)} cases\n"
            f"({len(missing_csv)} missing from CSV {'(' + ','.join(missing_csv) + ')' if missing_csv else ''}, "
            f"{len(missing_dir)} missing from files {'(' + ','.join(missing_dir) + ')' if missing_dir else ''})")
        if self.stack_intrinsic:
            print(f"{len(missing_intrinsic)} missing intrinsic beats: {'(' + ','.join(missing_intrinsic) + ')' if missing_intrinsic else ''}")
        return cases

    def get_traces(self):
        n_traces_valid = 0
        n_traces_included = 0
        included_traces_by_beattype_by_case = defaultdict(lambda: defaultdict(list))
        invalid_traces_by_beattype = defaultdict(list)

        all_traces = [f for case in self.cases for f in glob(os.path.join(self.paceddata_path, case, "*.npy"))]
        print(f"{self.train_or_test.upper() + ':': <6} {len(all_traces)} ECGs available")

        # Build up dictionary of cases by beat-type per case, and track invalid beats
        for case in self.cases.copy():
            found_valid_traces_for_case = False
            npyfiles = sorted(glob(os.path.join(self.paceddata_path, case, "*.npy")))
            for npyfile in npyfiles:
                beattype = npyfile.split('_')[3]
                if beattype in self.beat_types:
                    n_traces_valid += 1

                    if self.final_test and len(included_traces_by_beattype_by_case[case][self.beat_types[beattype]]) >= self.cfg['data']['max_samples_per_beattype']:
                        # If we're doing the final test, only a certain number of beats of a certain type per case can be included
                        continue
                    else:
                        included_traces_by_beattype_by_case[case][beattype].append(npyfile)
                        n_traces_included += 1
                        found_valid_traces_for_case = True
                else:
                    invalid_traces_by_beattype[beattype].append(npyfile)
            if not found_valid_traces_for_case:
                self.cases.remove(case)

        traces_by_beattype_valid = defaultdict(list)
        for case, traces_by_beattype in included_traces_by_beattype_by_case.items():
            for beattype, traces in traces_by_beattype.items():
                if self.final_test and len(traces) < self.cfg['data']['max_samples_per_beattype']:
                    # Don't include in final testing set if insufficient numbers of beats
                    continue
                elif self.final_test and len(traces) > self.cfg['data']['max_samples_per_beattype']:
                    raise ValueError(f"Should not have grabbed more cases than we need if we're testing!")
                else:
                    traces_by_beattype_valid[beattype].extend(traces)

        print(f"{self.train_or_test.upper() + ':': <6} loaded {n_traces_included} of {n_traces_valid} valid ECGs")
        for beattype, traces in traces_by_beattype_valid.items():
            print(f"\t{beattype: <3} {len(traces)} ECGs -> {self.beat_types[beattype]}")
        for beattype, traces in invalid_traces_by_beattype.items():
            print(f"\t{beattype: <3} {len(traces)} ECGs (excluded)")
        traces_valid = [t for traces in traces_by_beattype_valid.values() for t in traces]
        return traces_valid

    def get_median_iqr(self):
        median, iqr = np.zeros((12)), np.zeros((12))
        print(f"Getting normalisation parameters...")
        for npypath in tqdm(self.traces):
            ecg = np.load(npypath)
            ecg = ecg - ecg[0]
            print(ecg.shape)

            median = median + np.median(ecg, axis=0)
            iqr = iqr + stats.iqr(ecg, axis=0)

        median = median / len(self.traces)
        iqr = iqr / len(self.traces)

        print(f"Medians: [{','.join([str(int(v)) for v in median])}]")
        print(f"IQRs: [{','.join([str(int(v)) for v in iqr])}]")
        return median, iqr
