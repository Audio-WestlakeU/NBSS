# the SMS-WSJ-Plus dataset used in
# `Changsheng Quan, Xiaofei Li. SpatialNet: Extensively Learning Spatial Information for Multichannel Joint Speech Separation, Denoising and Dereverberation.`


import json
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
from data_loaders.utils.diffuse_noise import (gen_desired_spatial_coherence, gen_diffuse_noise)
from data_loaders.utils.window import reverberation_time_shortening_window


class SmsWsjPlusDataset(Dataset):

    def __init__(
        self,
        sms_wsj_dir: str,
        rir_dir: str,
        target: str,
        dataset: str,
        ovlp: str,
        speech_overlap_ratio: Tuple[float, float] = [0.1, 1.0],
        sir: Tuple[float, float] = [-5, 5],
        snr: Tuple[float, float] = [10, 20],
        audio_time_len: Optional[float] = None,
        sample_rate: int = 8000,
        num_spk: int = 2,
        noise_type: List[Literal['babble', 'white']] = ['babble', 'white'],
        return_noise: bool = False,
        return_rvbt: bool = False,
    ) -> None:
        """The SMS-WSJ-plus dataset

        Args:
            sms_wsj_dir: a dir contains [wsj_8k_zeromean, sms_wsj.json, ...]
            target:  revb_image, direct_path
            dataset: train_si284, cv_dev93, test_eval92
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert target in ['revb_image', 'direct_path'] or target.startswith('RTS'), target
        assert dataset in ['train_si284', 'cv_dev93', 'test_eval92'], dataset
        assert ovlp in ['mid', 'headtail', 'startend', 'full', 'hms', 'fhms'], ovlp
        assert num_spk == 2, ('Not implemented for spk num=', num_spk)
        assert len(set(noise_type) - set(['babble', 'white'])) == 0, noise_type

        if ovlp == 'full' and audio_time_len == None:
            rank_zero_warn(f'dataset {dataset} could not achieve full-overlap without giving a length, the overlap type will be one of startend/headtail/mid-overlap')
            ovlp = 'hms'

        self.sms_wsj_dir = Path(sms_wsj_dir).expanduser()
        self.target = target
        self.dataset = dataset
        self.ovlp = ovlp
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sir = sir
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate
        assert sample_rate == 8000, ('Not implemented for sample rate ', sample_rate)

        with open(self.sms_wsj_dir / 'sms_wsj.json', 'r') as f:
            d = json.load(f)
            self.dataset_info = d['datasets'][dataset]
            self.keys = list(self.dataset_info.keys())

        original_sources = []
        for k, v in self.dataset_info.items():
            del v['room_dimensions']
            del v['sound_decay_time']
            del v['source_position']
            del v['sensor_position']
            v['original_source'] = v['audio_path']['original_source']
            del v['audio_path']
            v['original_source'] = [str(self.sms_wsj_dir / ('wsj_8k_zeromean' + p.split('wsj_8k_zeromean')[-1])) for p in v['original_source']]
            original_sources += v['original_source']
            v['wavname'] = k + '.wav'
            v['saveto'] = [k + '_0.wav', k + '_1.wav']
            self.dataset_info[k] = v

        self.return_rvbt = return_rvbt
        self.return_noise = return_noise
        self.noises = list(set(original_sources))  # take the speech signal in this dataset as babble noise source
        self.noises.sort()
        self.snr = snr

        self.rir_dir = Path(rir_dir).expanduser() / {"train_si284": "train", "cv_dev93": 'validation', 'test_eval92': 'test'}[dataset]
        self.rirs = [str(r) for r in list(Path(self.rir_dir).expanduser().rglob('*.npz'))]
        self.rirs.sort()
        # load & save diffuse parameters
        diffuse_paras_path = (Path(rir_dir) / 'diffuse.npz').expanduser()
        if diffuse_paras_path.exists():
            self.Cs = np.load(diffuse_paras_path, allow_pickle=True)['Cs']
        else:
            pos_mics = np.load(self.rirs[0], allow_pickle=True)['pos_rcv']
            _, self.Cs = gen_desired_spatial_coherence(pos_mics=pos_mics, fs=self.sample_rate, noise_field='spherical', c=343, nfft=256)
            try:
                np.savez(diffuse_paras_path, Cs=self.Cs)
            except:
                ...
        assert len(self.rirs) > 0, f"{str(self.rir_dir)} is empty or not exists"
        self.shuffle_rir = True if dataset == "train_si284" else False

        self.num_spk = num_spk
        self.noise_type = noise_type

    def __getitem__(self, index_seed: tuple[int, int]):
        # for each item, an index and seed are given. The seed is used to reproduce this dataset on any machines
        index, seed = index_seed

        rng = np.random.default_rng(np.random.PCG64(seed))
        num_spk = self.num_spk
        info = self.dataset_info[self.keys[index]]

        # step 1: load single channel clean speech signals
        cleans, uttrsc = [], []
        for i in range(self.num_spk):
            uttrsc.append(info['original_source'][i])
            source, sr_src = sf.read(uttrsc[i], dtype='float32')
            cleans.append(source)
            assert sr_src == self.sample_rate, (sr_src, self.sample_rate)

        # step 2: load rirs
        if self.shuffle_rir:
            rir_this = self.rirs[rng.integers(low=0, high=len(self.rirs))]
        else:
            rir_this = self.rirs[index % len(self.rirs)]
        rir_dict = np.load(rir_this)
        sr_rir = rir_dict['fs']
        assert sr_rir == self.sample_rate, (sr_rir, self.sample_rate)

        rir = rir_dict['rir']  # shape [nsrc,nmic,time]
        assert rir.shape[0] >= num_spk, (rir.shape, num_spk)
        spk_rir_idxs = rng.choice(rir.shape[0], size=num_spk, replace=False).tolist()
        assert len(set(spk_rir_idxs)) == num_spk, spk_rir_idxs
        rir = rir[spk_rir_idxs, :, :]
        if self.target == 'direct_path':  # read simulated direct-path rir
            rir_target = rir_dict['rir_dp']  # shape [nsrc,nmic,time]
            rir_target = rir_target[spk_rir_idxs, :, :]
        elif self.target == 'revb_image':  # revb_image
            rir_target = rir  # shape [nsrc,nmic,time]
        elif self.target.startswith('RTS'):  # e.g. RTS_0.1s
            rts_time = float(self.target.replace('RTS_', '').replace('s', ''))
            win = reverberation_time_shortening_window(rir=rir, original_T60=rir_dict['RT60'], target_T60=rts_time, sr=self.sample_rate)
            rir_target = win * rir
        else:
            raise NotImplementedError('Unknown target: ' + self.target)
        num_mic = rir.shape[1]

        # step 3: decide the overlap type, overlap ratio, and the needed length of the two signals
        # randomly sample one ovlp_type if self.ovlp==fhms or hms
        ovlp_type = sample_an_overlap(rng=rng, ovlp_type=self.ovlp, num_spk=num_spk)
        # randomly sample the overlap ratio if necessary and decide the needed length of signals
        lens = [clean.shape[0] for clean in cleans]  # clean speech length of each speaker
        ovlp_ratio, lens, mix_frames = sample_ovlp_ratio_and_cal_length(
            rng=rng,
            ovlp_type=ovlp_type,
            ratio_range=self.speech_overlap_ratio,
            target_len=None if self.audio_time_len is None else int(self.audio_time_len * self.sample_rate),
            lens=lens,
        )

        # step 4: repeat signals if they are shorter than the length needed, then cut them to needed
        cleans = pad_or_cut(wavs=cleans, lens=lens, rng=rng)

        # step 5: convolve rir and clean speech, then place them at right place to satisfy the given overlap types
        rvbts, targets = zip(*[convolve(wav=wav, rir=rir_spk, rir_target=rir_spk_t, ref_channel=0, align=True) for (wav, rir_spk, rir_spk_t) in zip(cleans, rir, rir_target)])
        rvbts, targets = overlap2(rvbts=rvbts, targets=targets, ovlp_type=ovlp_type, mix_frames=mix_frames, rng=rng)

        # step 6: rescale rvbts and targets
        sir_this = None
        if self.sir != None and num_spk == 2:
            sir_this = rng.uniform(low=self.sir[0], high=self.sir[1])  # randomly sample in the given range
            assert len(cleans) == 2, len(cleans)
            coeff = cal_coeff_for_adjusting_relative_energy(wav1=rvbts[0], wav2=rvbts[1], target_dB=sir_this)
            assert coeff is not None
            # scale cleans[1] to -5 ~ 5 dB
            rvbts[1][:] *= coeff
            if targets is not rvbts:
                targets[1][:] *= coeff

        # step 7: generate diffused noise and mix with a sampled SNR
        noise_type = self.noise_type[rng.integers(low=0, high=len(self.noise_type))]
        mix = np.sum(rvbts, axis=0)
        if noise_type == 'babble':
            noises = []
            for i in range(num_mic):
                noise_i = np.zeros(shape=(mix_frames,), dtype=mix.dtype)
                for j in range(10):
                    noise_path = self.noises[rng.integers(low=0, high=len(self.noises))]
                    noise_ij, sr_noise = sf.read(noise_path, dtype='float32', always_2d=False)  # [T]
                    assert sr_noise == self.sample_rate and noise_ij.ndim == 1, (sr_noise, self.sample_rate)
                    noise_i += pad_or_cut([noise_ij], lens=[mix_frames], rng=rng)[0]
                noises.append(noise_i)
            noise = np.stack(noises, axis=0).reshape(-1)
        elif noise_type == 'white':
            noise = rng.normal(size=mix.shape[0] * mix.shape[1])
        noise = gen_diffuse_noise(noise=noise, L=mix_frames, Cs=self.Cs, nfft=256, rng=rng)  # shape [num_mic, mix_frames]

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

        paras = {
            'index': index,
            'seed': seed,
            'saveto': info['saveto'],
            'target': self.target,
            'sample_rate': 8000,
            'dataset': f'SMS-WSJ-Plus/{self.dataset}',
            'noise_type': noise_type,
            'noise': noises if self.return_noise else None,
            'rvbt': rvbts if self.return_rvbt else None,
            'sir': float(sir_this),
            'snr': float(snr_real),
            'ovlp_type': ovlp_type,
            'ovlp_ratio': float(ovlp_ratio),
            'audio_time_len': self.audio_time_len,
            'num_spk': num_spk,
            'rir': {
                'RT60': rir_dict['RT60'],
                'pos_src': rir_dict['pos_src'],
                'pos_rcv': rir_dict['pos_rcv'],
            }
        }

        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32), paras

    def __len__(self):
        return len(self.dataset_info)


class SmsWsjPlusDataModule(LightningDataModule):

    def __init__(
        self,
        sms_wsj_dir: str = '~/datasets/sms_wsj',  # a dir contains [early, noise, observation, rirs, speech_source, tail, wsj_8k_zeromean]
        rir_dir: str = '~/datasets/SMS_WSJ_Plus_rirs',  # containing train, validation, and test subdirs
        target: str = "direct_path",  # e.g. rvbt_image, direct_path
        datasets: Tuple[str, str, str, str] = ['train_si284', 'cv_dev93', 'test_eval92', 'test_eval92'],  # datasets for train/val/test/predict
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None, None],  # audio_time_len (seconds) for train/val/test/predictS
        ovlp: Union[str, Tuple[str, str, str, str]] = "mid",  # speech overlapping type for train/val/test/predict: 'mid', 'headtail', 'startend', 'full', 'hms', 'fhms'
        speech_overlap_ratio: Tuple[float, float] = [0.1, 1.0],
        sir: Optional[Tuple[float, float]] = [-5, 5],  # relative energy of speakers (dB), i.e. signal-to-interference ratio
        snr: Tuple[float, float] = [0, 20],  # SNR dB
        num_spk: int = 2,  # separation task: 2 speakers; enhancement task: 1 speaker
        noise_type: List[Literal['babble', 'white']] = ['babble', 'white'],  # the type of noise
        return_noise: bool = False,
        return_rvbt: bool = False,
        batch_size: List[int] = [1, 1],  # batch size for [train, val, {test, predict}]
        num_workers: int = 10,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int, int] = [None, 2, 3, 3],  # random seeds for train/val/test/predict sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.sms_wsj_dir = sms_wsj_dir
        self.rir_dir = rir_dir
        self.target = target
        self.datasets = datasets
        self.audio_time_len = audio_time_len
        self.ovlp = [ovlp] * 4 if isinstance(ovlp, str) else ovlp
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sir = sir
        self.snr = snr
        self.num_spk = num_spk
        self.noise_type = noise_type
        self.return_noise = return_noise
        self.return_rvbt = return_rvbt
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size
        while len(self.batch_size) < 4:
            self.batch_size.append(1)

        rank_zero_info("dataset: SMS-WSJ-Plus")
        rank_zero_info(f'train/val/test/predict: {self.datasets}')
        rank_zero_info(f'batch size: train/val/test/predict = {self.batch_size}')
        rank_zero_info(f'audio_time_length: train/val/test/predict = {self.audio_time_len}')
        rank_zero_info(f'target: {self.target}')
        # assert self.batch_size_val == 1, "batch size for validation should be 1 as the audios have different length"
        # assert audio_time_len[2] == None, "the length for test set should be None if you want to test ASR performance"

        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test, default_collate_func]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, ovlp, audio_time_len, seed, shuffle, batch_size, collate_fn):
        if dataset.endswith('cn'):
            from data_loaders.sms_wsj_plus_cn import SmsWsjPlusCNDataset
            ds = SmsWsjPlusCNDataset(
                sms_wsj_dir=self.sms_wsj_dir,
                rir_dir=self.rir_dir,
                target=self.target,
                dataset=dataset,
                ovlp=ovlp,
                speech_overlap_ratio=self.speech_overlap_ratio,
                sir=self.sir,
                snr=self.snr,
                audio_time_len=audio_time_len,
                num_spk=self.num_spk,
                noise_type=self.noise_type,
                return_noise=self.return_noise,
                return_rvbt=self.return_rvbt,
            )
        else:
            ds = SmsWsjPlusDataset(
                sms_wsj_dir=self.sms_wsj_dir,
                rir_dir=self.rir_dir,
                target=self.target,
                dataset=dataset,
                ovlp=ovlp,
                speech_overlap_ratio=self.speech_overlap_ratio,
                sir=self.sir,
                snr=self.snr,
                audio_time_len=audio_time_len,
                num_spk=self.num_spk,
                noise_type=self.noise_type,
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
            ovlp=self.ovlp[0],
            audio_time_len=self.audio_time_len[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            ovlp=self.ovlp[1],
            audio_time_len=self.audio_time_len[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[2],
            ovlp=self.ovlp[2],
            audio_time_len=self.audio_time_len[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            ovlp=self.ovlp[3],
            audio_time_len=self.audio_time_len[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
        )


if __name__ == '__main__':
    """python -m data_loaders.sms_wsj_plus"""
    from jsonargparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_class_arguments(SmsWsjPlusDataModule, nested_key='data')
    parser.add_argument('--save_dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='predict')
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    args_dict = args.data
    args_dict['num_workers'] = 1  # for debuging
    datamodule = SmsWsjPlusDataModule(**args_dict)
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
            if idx > 10:
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

            print(noisy.shape, None if args.dataset.startswith('predict') else tar.shape, paras)
