import os
from os.path import *
import random
from pathlib import Path
from typing import *

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)
from torch.utils.data import DataLoader, Dataset

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.mix import *
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from scipy.signal import resample_poly


class CHiME3MovingDataset(Dataset):

    def __init__(
        self,
        dataset: str,
        target: str,
        wsj0_dir: str = '~/datasets/wsj0',
        rir_dir: str = '~/datasets/CHiME3_moving_rirs',
        chime3_dir: str = '~/datasets/CHiME3',
        snr: Tuple[float, float] = [-5, 10],
        audio_time_len: Optional[float] = None,
        sample_rate: int = 8000,
        return_noise: bool = False,
        return_rvbt: bool = False,
    ) -> None:
        """The CHiME3-moving dataset

        Args:
            wsj0_dir: a dir contains [si_tr_s, si_dt_05, si_dt_20, si_et_05, si_et_20, ...]
            chime3_dir: a dir contains [data/audio/16kHz/backgrounds, ...]
            rir_dir: a dir contains [train, validation, test, rir_cfg.npz]
            target: revb_image, direct_path
            dataset: train, val, test
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert target in ['revb_image', 'direct_path'] or target.startswith('RTS'), target
        assert dataset.startswith('train') or dataset.startswith('val') or dataset.startswith('test'), dataset

        self.speed = None  # e.g. 'moving(0.12,0.4)' means moving with a speed in 0.12 ~ 0.4 m/s
        if 'moving' in dataset:
            speed = dataset.split('_')[-1].replace('moving(', '').replace(')', '').split(',')
            assert len(speed) == 2 or len(speed) == 3, speed
            self.speed = [float(spd) for spd in speed[:2]]
            # e.g. moving(0.12,0.4,0.5) means: with a probability of 0.5, moving with a speed in 0.12 ~ 0.4 m/s; with p=0.5, not moving
            self.prob_moving = float(speed[2]) if len(speed) == 3 else 1
            rir_cfg = dict(np.load(Path(rir_dir.split(',')[-1]).expanduser() / 'rir_cfg.npz', allow_pickle=True))
            self.adjacent_points_distance = rir_cfg['args'].item()['trajectory'][1]
        else:
            self.prob_moving = 0

        self.dataset0 = dataset
        dataset = dataset.split('_')[0]

        self.target = target
        self.dataset = dataset
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate
        self.return_rvbt = return_rvbt
        self.return_noise = return_noise
        assert sample_rate == 8000, ('Not implemented for sample rate ', sample_rate)

        # find clean speech signals
        self.wsj0_dir = Path(wsj0_dir).expanduser()
        self.spk2uttrs = dict()
        subdirs = {'train': ['si_tr_s'], 'val': ['si_dt_05', 'si_dt_20'], 'test': ['si_et_05', 'si_et_20']}[dataset]
        self.uttrs = []
        for subdir in subdirs:
            spks = [p.name for p in (self.wsj0_dir / subdir).glob('*')]
            for spk in spks:
                if spk not in self.spk2uttrs:
                    self.spk2uttrs[spk] = []
                uttrs = list((self.wsj0_dir / subdir / spk).glob('*.wav'))
                self.spk2uttrs[spk] += uttrs
                self.spk2uttrs[spk].sort()
                self.uttrs += uttrs
        self.uttrs.sort()
        self.length = {'train': 20000, 'val': 2000, 'test': 2000}[dataset]

        # find noises
        self.chime3_dir = Path(chime3_dir).expanduser()
        noise_dir = self.chime3_dir / "data" / "audio" / "16kHz" / "backgrounds"
        self.noises = list(noise_dir.rglob('*.CH1.wav'))  # for each noise, the first 80% is used for training, while the last two 10% are for validation and test
        self.noises.sort()
        self.noise_time_range = {'train': [0.0, 0.8], 'val': [0.8, 0.9], 'test': [0.9, 1.0]}[dataset]

        # find rirs
        self.shuffle_rir = True if dataset == "train" else False
        self.snr = snr
        self.rir_dir = Path(rir_dir).expanduser() / {"train": "train", "val": 'validation', 'test': 'test'}[dataset]
        self.rirs = [str(r) for r in list(Path(self.rir_dir).expanduser().rglob('*.npz'))]
        self.rirs.sort()

    def __getitem__(self, index_seed: tuple[int, int]):
        # for each item, an index and seed are given. The seed is used to reproduce this dataset on any machines
        index, seed = index_seed

        rng = np.random.default_rng(np.random.PCG64(seed))

        num_spk = 1
        # step 1: load single channel clean speech signals
        cleans, uttr_paths, cands = [], [], []
        for i in range(num_spk):
            uttr_paths.append(self.uttrs[rng.choice(range(len(self.uttrs)))])
            cands.append(self.spk2uttrs[Path(uttr_paths[i]).parent.name])
            wav, sr_src = sf.read(uttr_paths[i], dtype='float32')
            if sr_src != self.sample_rate:
                wav = resample_poly(wav, up=self.sample_rate, down=sr_src, axis=0)
            cleans.append(wav)

        # step 2: load rirs
        if self.shuffle_rir:
            rir_this = self.rirs[rng.integers(low=0, high=len(self.rirs))]
        else:
            rir_this = self.rirs[index % len(self.rirs)]
        rir_dict = np.load(rir_this, allow_pickle=True)
        sr_rir = rir_dict['fs']
        assert sr_rir == self.sample_rate, (sr_rir, self.sample_rate)

        rir = rir_dict['rir']  # shape [nsrc,nmic,time]
        num_mic = rir_dict['pos_rcv'].shape[0]
        spk_rir_idxs = rng.choice(rir.shape[0], size=num_spk, replace=False).tolist()
        rir = rir[spk_rir_idxs]  # might be a path
        if isinstance(rir[0], str):
            rir = [np.load(self.rir_dir / rir_path, mmap_mode='r') for rir_path in rir]

        assert len(spk_rir_idxs) == num_spk, spk_rir_idxs
        if self.target == 'direct_path':  # read simulated direct-path rir
            rir_target = rir_dict['rir_dp']  # shape [nsrc,nmic,time] or [[nloc,nmic,time],...]
            rir_target = rir_target[spk_rir_idxs]
            if isinstance(rir_target[0], str):
                rir_target = [np.load(self.rir_dir / rir_path, mmap_mode='r') for rir_path in rir_target]
        elif self.target == 'revb_image':  # rvbt_image
            rir_target = rir  # shape [nsrc,nmic,time] or [[nloc,nmic,time],...]
        else:
            raise NotImplementedError('Unknown target: ' + self.target)

        # step 4: append signals if they are shorter than the length needed, then cut them to needed
        if self.audio_time_len is None:
            lens = [clean.shape[0] for clean in cleans]  # clean speech length of each speaker
            mix_frames = max(lens)
        else:
            mix_frames = int(self.audio_time_len * self.sample_rate)
            lens = [mix_frames] * len(cleans)

        for i, wav in enumerate(cleans):
            # repeat
            while len(wav) < lens[i]:
                wav2, fs = sf.read(rng.choice(cands[i], size=1)[0])
                if fs != self.sample_rate:
                    wav2 = resample_poly(wav2, up=self.sample_rate, down=fs, axis=0)
                wav = np.concatenate([wav, wav2])
            # cut to needed length
            if len(wav) > lens[i]:
                start = rng.integers(low=0, high=len(wav) - lens[i] + 1)
                wav = wav[start:start + lens[i]]
            cleans[i] = wav

        # step 5: convolve rir and clean speech, then place them at right place to satisfy the given overlap types
        # moving or not
        if self.prob_moving > 0 and self.prob_moving < 1:
            moving = True if rng.uniform() > self.prob_moving else False
        else:
            moving = False if self.speed is None else True

        if moving == False:
            if rir[0].ndim == 3:  # a trajectory, sample a point in the trajectory
                which_point = [rng.integers(low=0, high=rir_spk.shape[0]) for rir_spk in rir]
                rir = [rir_spk[which_point[i]] for i, rir_spk in enumerate(rir)]
                rir_target = [rir_spk[which_point[i]] for i, rir_spk in enumerate(rir_target)]
            rvbts, targets = zip(*[convolve_v2(wav=wav, rir=rir_spk, rir_target=rir_spk_t, ref_channel=0, align=True) for (wav, rir_spk, rir_spk_t) in zip(cleans, rir, rir_target)])
        else:
            speed_this = rng.uniform(low=self.speed[0], high=self.speed[1], size=1)
            samples_per_rir = np.round(self.adjacent_points_distance / speed_this * sr_rir).astype(np.int32)
            rvbts, targets = [], []
            for (wav, rir_spk, rir_spk_t, nsamp_spk) in zip(cleans, rir, rir_target, samples_per_rir):
                num_rirs = int(np.ceil(mix_frames / nsamp_spk)) + 1
                # sample indexes for rirs used for convolve_traj
                rir_idx_spk_cands = list(range(rir_spk.shape[0]))
                if rng.integers(low=0, high=2) == 0:
                    rir_idx_spk_cands.reverse()
                start = rng.integers(low=0, high=len(rir_idx_spk_cands))
                rir_idx_spk_sel = rir_idx_spk_cands[start:]
                while len(rir_idx_spk_sel) < num_rirs:
                    rir_idx_spk_sel += rir_idx_spk_cands
                rir_idx_spk_sel = rir_idx_spk_sel[:num_rirs]

                # sample rir
                rir_spk = rir_spk[rir_idx_spk_sel]
                rir_spk_t = rir_spk_t[rir_idx_spk_sel]

                # convolve_traj
                rvbts_i = convolve_traj_with_win(wav=wav, traj_rirs=rir_spk, samples_per_rir=nsamp_spk, wintype='trapezium20')
                targets_i = convolve_traj_with_win(wav=wav, traj_rirs=rir_spk_t, samples_per_rir=nsamp_spk, wintype='trapezium20')
                rvbts_i, targets_i = align(rir=rir_spk_t[0, 0], rvbt=rvbts_i, target=targets_i, src=wav)
                rvbts.append(rvbts_i), targets.append(targets_i)
        rvbts, targets = np.stack(rvbts, axis=0), np.stack(targets, axis=0)

        # step 7: load noise and mix with a sampled SNR
        mix = np.sum(rvbts, axis=0)
        noise_path = self.noises[rng.integers(low=0, high=len(self.noises))]
        noise_frames = sf.info(noise_path).duration * sf.info(noise_path).samplerate
        noise_start, noise_end = int(self.noise_time_range[0] * noise_frames), int(self.noise_time_range[1] * noise_frames)

        noise = np.zeros(shape=(num_mic, mix_frames), dtype=mix.dtype)
        for n in range(1 if self.dataset != 'train' else rng.integers(low=1, high=3)):
            # noise data augmentation for train
            noise_frames_needed = mix_frames * 2
            start = rng.integers(low=noise_start, high=noise_end - noise_frames_needed) if (noise_end - noise_start) > noise_frames_needed else noise_start
            for i in range(num_mic):
                if (noise_end - noise_start) > noise_frames_needed:
                    wav, sr = sf.read(str(noise_path).replace('.CH1.wav', f'.CH{i+1}.wav'), frames=noise_frames_needed, start=start, dtype='float32')
                else:
                    wav, sr = sf.read(str(noise_path).replace('.CH1.wav', f'.CH{i+1}.wav'), frames=noise_end - noise_start, start=start, dtype='float32')
                    wav = np.concatenate([wav] * (noise_frames_needed // (noise_end - noise_start) + 1))[:noise_frames_needed]
                assert self.sample_rate == 8000 and sr == 16000, (sr, self.sample_rate)
                wav = resample_poly(wav, up=self.sample_rate, down=sr, axis=0)
                noise[i] += wav

        snr_this = rng.uniform(low=self.snr[0], high=self.snr[1])
        coeff = cal_coeff_for_adjusting_relative_energy(wav1=mix, wav2=noise, target_dB=snr_this)
        assert coeff is not None
        noise[:, :] *= coeff
        snr_real = 10 * np.log10(np.sum(mix**2) / np.sum(noise**2))
        assert np.isclose(snr_this, snr_real, atol=0.5), (snr_this, snr_real)
        mix[:, :] = mix + noise

        # scale mix and targets to [-0.9, 0.9]
        scale_value = 0.9 / max(np.max(np.abs(mix)), np.max(np.abs(targets)))
        mix[:, :] *= scale_value
        targets[:, :] *= scale_value
        noise = noise * scale_value if self.return_noise else None
        rvbts = rvbts * scale_value if self.return_rvbt else None

        paras = {
            'index': index,
            'seed': seed,
            'saveto': [str(p).removeprefix(str(self.wsj0_dir))[1:] for p in uttr_paths],
            'target': self.target,
            'sample_rate': self.sample_rate,
            'dataset': f'CHiME3_moving/{self.dataset0}',
            'snr': float(snr_real),
            'audio_time_len': self.audio_time_len,
            'num_spk': num_spk,
            'rir': {
                'RT60': rir_dict['RT60'],
                'pos_src': rir_dict['pos_src'],
                'pos_rcv': rir_dict['pos_rcv'],
            },
            'data': {
                'rir': rir,
                'noise': noise,
                'rvbt': rvbts,
            }
        }

        return (
            torch.as_tensor(mix, dtype=torch.float32),  # shape [chn, time]
            torch.as_tensor(targets, dtype=torch.float32),  # shape [spk, chn, time]
            paras,
        )

    def __len__(self):
        return self.length


class CHiME3MovingDataModule(LightningDataModule):

    def __init__(
        self,
        wsj0_dir: str = '~/datasets/wsj0',
        rir_dir: str = '~/datasets/CHiME3_moving_rirs',
        chime3_dir: str = '~/datasets/CHiME3',
        target: str = "direct_path",  # e.g. rvbt_image, direct_path
        datasets: Tuple[str, str, str, str] = ['train', 'val', 'test', 'test'],  # datasets for train/val/test/predict
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None, None],  # audio_time_len (seconds) for train/val/test/predictS
        snr: Tuple[float, float] = [-5, 10],  # SNR dB
        return_noise: bool = False,
        return_rvbt: bool = False,
        batch_size: List[int] = [1, 1],  # batch size for [train, val, {test, predict}]
        num_workers: int = 10,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int, int] = [None, 2, 3, 3],  # random seeds for train/val/test/predict sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = False,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.wsj0_dir = wsj0_dir
        self.rir_dir = rir_dir
        self.chime3_dir = chime3_dir
        self.target = target
        self.datasets = datasets
        self.audio_time_len = audio_time_len
        self.snr = snr
        self.return_noise = return_noise
        self.return_rvbt = return_rvbt
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size
        while len(self.batch_size) < 4:
            self.batch_size.append(1)

        rank_zero_info("dataset: CHiME3_moving")
        rank_zero_info(f'train/val/test/predict: {self.datasets}')
        rank_zero_info(f'batch size: train/val/test/predict = {self.batch_size}')
        rank_zero_info(f'audio_time_length: train/val/test/predict = {self.audio_time_len}')
        rank_zero_info(f'target: {self.target}')

        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test, default_collate_func]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, audio_time_len, seed, shuffle, batch_size, collate_fn):
        ds = CHiME3MovingDataset(
            dataset=dataset,
            target=self.target,
            wsj0_dir=self.wsj0_dir,
            rir_dir=self.rir_dir,
            chime3_dir=self.chime3_dir,
            snr=self.snr,
            audio_time_len=audio_time_len,
            sample_rate=8000,
            return_noise=self.return_noise,
            return_rvbt=self.return_rvbt,
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),  #
            batch_size=batch_size,  #
            collate_fn=collate_fn,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[0],
            audio_time_len=self.audio_time_len[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            audio_time_len=self.audio_time_len[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[2],
            audio_time_len=self.audio_time_len[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            audio_time_len=self.audio_time_len[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
        )


if __name__ == '__main__':
    """python -m data_loaders.chime3_moving"""
    dset = CHiME3MovingDataset(
        target='direct_path',
        dataset='test_moving(0.12,0.4,0.5)',  #, cv_dev93, test_eval92
        audio_time_len=128,
    )
    for i in range(100):
        dset.__getitem__((i, i))

    from jsonargparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_class_arguments(CHiME3MovingDataModule, nested_key='data')
    parser.add_argument('--save_dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='val')
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    args_dict = args.data
    args_dict['num_workers'] = 10  # 1 for debuging
    args_dict['datasets'] = ['train_moving(0.12,0.4,0.5)', 'val_moving(0.12,0.4,0.5)', 'test_moving(0.12,0.4,0.5)', 'test_moving(0.12,0.4,0.5)']
    args_dict['rir_dir'] = '~/datasets/CHiME3_moving_rirs'
    args_dict['audio_time_len'] = [30.0, 30.0, 30.0, 30.0]
    args_dict['return_noise'] = True
    datamodule = CHiME3MovingDataModule(**args_dict)
    datamodule.setup()

    if args.dataset.startswith('train'):
        dataloader = datamodule.train_dataloader()
    elif args.dataset.startswith('val'):
        dataloader = datamodule.val_dataloader()
    elif args.dataset.startswith('test'):
        dataloader = datamodule.test_dataloader()
    else:
        assert args.dataset.startswith('predict'), args.dataset
        dataloader = datamodule.predict_dataloader()

    if type(dataloader) != dict:
        dataloaders = {args.dataset: dataloader}
    else:
        dataloaders = dataloader

    for ds, dataloader in dataloaders.items():

        for idx, (noisy, tar, paras) in enumerate(dataloader):
            print(f'{idx}/{len(dataloader)}', end=' ')
            if idx > 20:
                continue
            # write target to dir
            if args.gen_target and not args.dataset.startswith('predict'):
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/target").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(tar[0, :, 0, :].numpy())) <= 1
                for spk in range(tar.shape[1]):
                    sp = tar_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, tar[0, spk, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            # write unprocessed's 0-th channel
            if args.gen_unprocessed:
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noisy").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(noisy[0, 0, :].numpy())) <= 1
                for spk in range(len(paras[0]['saveto'])):
                    sp = tar_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, noisy[0, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            # write noise
            if paras[0]['data']['noise'] is not None:
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noise").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                sp = tar_path / basename(paras[0]['saveto'][0])
                sf.write(sp, paras[0]['data']['noise'][0], samplerate=paras[0]['sample_rate'])

            print(noisy.shape, None if args.dataset.startswith('predict') else tar.shape, paras)
