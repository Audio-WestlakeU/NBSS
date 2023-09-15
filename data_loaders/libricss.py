import os
import random
import warnings
from os.path import basename
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from numpy.linalg import norm
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)
from torch.utils.data import DataLoader, Dataset

from data_loaders.utils.array_geometry import libricss_array_geometry
from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.diffuse_noise import (gen_desired_spatial_coherence, gen_diffuse_noise)
from data_loaders.utils.mix import *
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler


class LibriCSSDataset(Dataset):

    def __init__(
        self,
        libricss_dir: str,  #  a dir contains [exp/data/7ch/utterances, exp/data/7ch/segments, ...]
        librispeech_dir: str,  # a dir contains [train-clean-100, train-clean-360]
        reverb_dir: str,  # a dir contains [reverb_tools_for_Generate_mcTrainData/NOISE, reverb_tools_for_Generate_SimData/NOISE]
        rir_dir: str,  # a dir contains [train, validation, test]
        target: str,
        dataset: str,
        ovlp: str,
        speech_overlap_ratio: Tuple[float, float] = [0.1, 1.0],
        sir: Tuple[float, float] = [-5, 5],  # signal interference ratio
        snr: Optional[Tuple[float, float]] = [5, 20],  # signal noise ratio
        audio_time_len: Optional[float] = None,
        sample_rate: int = 16000,
        spk1_prob: float = 2 / 6,
        spk2_prob: float = 0.7,
    ) -> None:
        """The LibriCSS dataset

        Args:
            target:  direct_path
            dataset: train_si284, cv_dev93, test_eval92
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert target in ['direct_path'], target
        assert dataset in ['SimTrain', 'SimVal', 'SimTest', 'utterances', 'segments', 'utterances/session0', 'segments/session0'], dataset
        assert ovlp in ['mid', 'headtail', 'startend', 'full', 'hms', 'fhms'], ovlp
        assert spk1_prob >= 0 and spk1_prob <= 1

        if ovlp == 'full' and audio_time_len == None:
            rank_zero_warn(f'dataset {dataset} could not achieve full-overlap without giving a length, the overlap type will be one of startend/headtail/mid-overlap')
            ovlp = 'hms'

        self.libricss_dir = Path(libricss_dir).expanduser()
        self.librispeech_dir = Path(librispeech_dir).expanduser()
        self.reverb_dir = Path(reverb_dir).expanduser()
        self.rir_dir = Path(rir_dir).expanduser()

        self.target = target
        self.dataset = dataset
        self.ovlp = ovlp
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sir = sir
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate
        assert sample_rate == 16000, ('Not implemented for sample rate ', sample_rate)

        if dataset in ['SimTrain', 'SimVal', 'SimTest']:
            # scan clean speeches
            spks_100 = list((self.librispeech_dir / 'train-clean-100').glob('*'))
            spks_360 = list((self.librispeech_dir / 'train-clean-360').glob('*'))
            spks_test = list((self.librispeech_dir / 'test-clean').glob('*'))
            spks_100.sort()
            spks_360.sort()

            spks_100_train, spks_100_val = spks_100[:-20], spks_100[-20:]
            spks_360_train, spks_360_val = spks_360[:-50], spks_360[-50:]
            if dataset == 'SimTrain':
                spks = spks_100_train + spks_360_train
            elif dataset == 'SimVal':
                spks = spks_100_val + spks_360_val

            if dataset in ['SimTrain', 'SimVal']:
                self.uttrs = []
                for spk in spks:
                    self.uttrs += list(spk.rglob('*.flac'))
            else:
                assert dataset == 'SimTest', dataset
                self.uttrs = list((self.librispeech_dir / 'test-clean').rglob('*.flac'))
            self.uttrs.sort()

            # scan rirs
            self.rir_dir = self.rir_dir / {'SimTrain': 'train', 'SimVal': 'validation', 'SimTest': 'test'}[dataset]
            self.rirs = [str(r) for r in list(self.rir_dir.glob('*.npz'))]
            self.rirs.sort()

            # scan noise
            self.noise_dir = self.reverb_dir / {
                'SimTrain': 'reverb_tools_for_Generate_mcTrainData/NOISE',
                'SimVal': 'reverb_tools_for_Generate_SimData/NOISE',
                'SimTest': 'reverb_tools_for_Generate_SimData/NOISE'
            }[dataset]
            self.noises = list(self.noise_dir.glob('*.wav'))
            self.noises.sort()
            self.snr = snr

            # check
            assert len(self.uttrs) > 0 and len(self.rirs) > 0 and len(self.noises) > 0, ('dir does not exist or is empty', self.librispeech_dir, self.rir_dir, self.noise_dir)
            pos_mics_1 = np.load(self.rirs[0], allow_pickle=True)['pos_rcv']
            pos_mics = libricss_array_geometry()
            dist_0 = norm(pos_mics[:, np.newaxis, :] - pos_mics[np.newaxis, :, :], axis=-1)  # shape [M, M]
            dist_1 = norm(pos_mics_1[:, np.newaxis, :] - pos_mics_1[np.newaxis, :, :], axis=-1)  # shape [M, M]
            assert np.allclose(dist_0, dist_1), "not the libricss array"
            _, self.Cs = gen_desired_spatial_coherence(pos_mics=pos_mics, fs=self.sample_rate, noise_field='spherical', c=343, nfft=256)
        else:
            assert dataset in ['utterances', 'segments', 'utterances/session0', 'segments/session0'], dataset  # session 0 is used for validation
            self.libricss_dir = self.libricss_dir / f"exp/data/7ch/{dataset.split('/')[0]}"
            self.uttrs = list(self.libricss_dir.rglob('*.wav'))
            if 'session0' in dataset:
                uttrs = []
                for uttr in self.uttrs:
                    if 'session0' in uttr.parent.name:
                        uttrs.append(uttr)
                self.uttrs = uttrs
            self.uttrs.sort()
            self.transcription = dict()
            if dataset.startswith('utterances'):
                with open(self.libricss_dir / 'utterance_transcription.txt', 'r') as f:
                    lines = f.readlines()
                    for l in lines:
                        self.transcription[l.split('\t')[0]] = l.replace(l.split('\t')[0], '').strip()

        self.spk1_prob = spk1_prob
        self.spk2_prob = spk2_prob

    def __getitem__(self, index_seed: tuple[int, int]):
        index, seed = index_seed

        # step 0: load data from disk if dataset in ['utterances', 'segments', 'utterances/session0', 'segments/session0']
        if self.dataset in ['utterances', 'segments', 'utterances/session0', 'segments/session0']:
            mix, sr = sf.read(self.uttrs[index])
            mix = mix.T
            assert self.audio_time_len == None, self.audio_time_len
            wav_short_path = str(self.uttrs[index]).replace(str(self.libricss_dir) + os.path.sep, '')
            paras = {
                'index': index,
                'sample_rate': 16000,
                'dataset': f'LibriCSS/{self.dataset}',
                'saveto': [wav_short_path.replace('.wav', f'_{x}.wav') for x in [0, 1]],
                'transcription': [self.transcription[wav_short_path.replace(os.path.sep, '_').replace('.wav', '')]] * 2 if self.dataset.startswith('utterances') else None,
            }

            return torch.as_tensor(mix, dtype=torch.float32), None, paras

        rng = np.random.default_rng(np.random.PCG64(seed))
        prob = rng.uniform()
        prob2 = rng.uniform()
        num_spk = 1 if prob < self.spk1_prob else (2 if prob2 < self.spk2_prob else 3)

        # generate audio data by convolving rir and clean speech signals
        # step 1: load single channel clean speeches
        cleans = []
        for i in range(num_spk):
            original_source, sr_src = sf.read(self.uttrs[rng.integers(low=0, high=len(self.uttrs))], dtype='float32')
            cleans.append(original_source)

        # step 2: load rirs
        rir_dict = np.load(self.rirs[rng.integers(low=0, high=len(self.rirs))])
        sr_rir = rir_dict['fs']  # shape [nsrc,nmic,time]
        assert sr_src == sr_rir, (sr_src, sr_rir)

        rir = rir_dict['rir']  # shape [nsrc,nmic,time]
        spk_idxs = rng.choice(rir.shape[0], size=num_spk, replace=False).tolist()
        assert len(set(spk_idxs)) == num_spk, spk_idxs
        rir = rir[spk_idxs, :, :]
        if self.target == 'direct_path':  # read simulated direct-path rir
            rir_target = rir_dict['rir_dp']  # shape [nsrc,nmic,time]
            rir_target = rir_target[spk_idxs, :, :]
        elif self.target == 'revb_image':  # revb_image
            rir_target = rir  # shape [nsrc,nmic,time]
        else:
            raise NotImplementedError('Unknown target: ' + self.target)

        # step 3: decide the overlap type, overlap ratio, and the needed length of the two signals
        if num_spk <= 2:
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
        else:
            assert self.audio_time_len is not None, self.audio_time_len
            mix_frames = int(self.audio_time_len * self.sample_rate)
            sil = int(rng.uniform(low=0.1, high=1.0) * self.sample_rate)  # silence between two interference speakers
            lens = [mix_frames, (mix_frames - sil) // 2, mix_frames - sil - (mix_frames - sil) // 2]  # clean speech length of each speaker
            ovlp_ratio = (mix_frames - sil) / mix_frames
            ovlp_type = 'startend3'

        # step 4: repeat signals if they are shorter than the length needed, then cut them to needed
        cleans = pad_or_cut(wavs=cleans, lens=lens, rng=rng)

        # step 5: convolve rir and clean speech, then place them at right place to satisfy the given overlap types
        rvbts, targets = zip(*[convolve(wav=wav, rir=rir_spk, rir_target=rir_spk_t, ref_channel=0, align=True) for (wav, rir_spk, rir_spk_t) in zip(cleans, rir, rir_target)])
        if num_spk <= 2:
            rvbts, targets = overlap2(rvbts=rvbts, targets=targets, ovlp_type=ovlp_type, mix_frames=mix_frames, rng=rng)
        else:
            rvbts, targets = overlap3(rvbts=rvbts, targets=targets, mix_frames=mix_frames, rng=rng)

        # step 6: rescale rvbts and targets
        sir_this = None
        if self.sir != None and (num_spk == 2 or num_spk == 3):
            sir_this = rng.uniform(low=self.sir[0], high=self.sir[1])  # randomly sample in the given range
            assert rvbts.shape[0] == 2, rvbts.shape
            coeff = cal_coeff_for_adjusting_relative_energy(wav1=rvbts[0], wav2=rvbts[1], target_dB=sir_this)
            if coeff is None:
                return self.__getitem__(index_seed=(rng.integers(low=0, high=len(self)), rng.integers(low=0, high=9999999999)))
            # scale cleans[1] to -5~5dB
            rvbts[1][:] *= coeff
            if targets is not rvbts:
                targets[1][:] *= coeff

        # step 7: generate diffused noise and mix with a sampled SNR
        mix = np.sum(rvbts, axis=0)
        if self.snr is not None:
            nidx = rng.integers(low=0, high=len(self.noises))
            noise_path = self.noises[nidx]
            noise, sr_noise = sf.read(noise_path, dtype='float32', always_2d=True)  # [T, num_mic]
            assert sr_noise == self.sample_rate, (sr_noise, self.sample_rate)
            # n_chn_idx = rng.integers(low=0, high=noise.shape[1])
            # noise = noise[:, n_chn_idx]
            noise = noise.T.reshape(-1)
            noise = gen_diffuse_noise(noise=noise, L=mix_frames, Cs=self.Cs, nfft=256, rng=rng)  # shape [num_mic, mix_frames]

            snr_this = rng.uniform(low=self.snr[0], high=self.snr[1])
            coeff = cal_coeff_for_adjusting_relative_energy(wav1=mix, wav2=noise, target_dB=snr_this)
            if coeff is None:
                return self.__getitem__(index_seed=(rng.integers(low=0, high=len(self)), rng.integers(low=0, high=9999999999)))
            noise[:, :] *= coeff
            snr_real = 10 * np.log10(np.sum(mix**2) / np.sum(noise**2))
            if not np.isclose(snr_this, snr_real, atol=0.1):  # something wrong happen, skip this item
                warnings.warn(f'skip LibriCSS/{self.dataset} item ({index},{seed})')
                return self.__getitem__(index_seed=(rng.integers(low=0, high=len(self)), rng.integers(low=0, high=9999999999)))
            # assert np.isclose(snr_this, snr_real, atol=0.1), (snr_this, snr_real)
            mix[:, :] = mix + noise
        else:
            snr_real = None

        # scale mix and targets to [-0.9, 0.9]
        scale_value = 0.9 / max(np.max(np.abs(mix)), np.max(np.abs(targets)))
        mix[:, :] *= scale_value
        targets[:, :] *= scale_value
        if num_spk == 1:
            targets = np.concatenate([targets, np.zeros(targets.shape)], axis=0)

        paras = {
            'index': index,
            'seed': seed,
            'sample_rate': 16000,
            'dataset': f'LibriCSS/{self.dataset}',
            'saveto': [f"{index}_1.wav", f"{index}_2.wav"],
            'snr': float(snr_real) if snr_real is not None else None,
            'ovlp_type': ovlp_type,
            'ovlp_ratio': float(ovlp_ratio),
            'ovlp(all)': self.ovlp,
            'audio_time_len': self.audio_time_len,
            'num_spk': num_spk,
            'num_stream': 1 if num_spk == 1 else 2,
        }

        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32), paras

    def __len__(self):
        if self.dataset in ['utterances', 'segments', 'utterances/session0', 'segments/session0']:
            return len(self.uttrs)
        else:
            return {'SimTrain': 20000, 'SimVal': 3000, 'SimTest': 3000}[self.dataset]


class LibriCSSDataModule(LightningDataModule):

    def __init__(
        self,
        libricss_dir: str = '~/datasets/LibriCSS',  #  a dir contains [exp/data/7ch/utterances, exp/data/7ch/segments, ...]
        librispeech_dir: str = '~/datasets/LibriSpeech',  # a dir contains [train-clean-100, train-clean-360]
        reverb_dir: str = '~/datasets/Reverb',  # a dir contains [reverb_tools_for_Generate_mcTrainData/NOISE, reverb_tools_for_Generate_SimData/NOISE]
        rir_dir: str = '~/datasets/LibriCSS_rirs',  # a dir contains [train, validation, test]
        datasets: Tuple[str, str, str, List[str]] = ('SimTrain', 'SimVal', 'SimTest', ['utterances']),  # datasets for train/val/test/predict
        target: str = "direct_path",  # e.g. revb_image, direct_path
        spk1_prob: float = 2 / 6,  # the probability
        spk2_prob: float = 0.7,  # the probability
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None],  # audio_time_len (seconds) for training, val, test.
        ovlp: str = "hms",  # speech overlapping type
        speech_overlap_ratio: Tuple[float, float] = [0.1, 1.0],
        sir: Tuple[float, float] = [-5, 5],  # relative energy of speakers (dB), i.e. signal-to-interference ratio
        snr: Optional[Tuple[float, float]] = [5, 20],  # SNR dB
        num_spk: int = 2,  # separation task: 2 speakers; enhancement task: 1 speaker
        batch_size: List[int] = [1, 1],
        test_set: str = 'test',
        num_workers: int = 10,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int] = [None, 2, 3],  # random seeds for train, val and test sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.libricss_dir = libricss_dir
        self.librispeech_dir = librispeech_dir
        self.reverb_dir = reverb_dir
        self.rir_dir = rir_dir
        self.datasets = datasets
        self.target = target
        self.spk1_prob = spk1_prob
        self.spk2_prob = spk2_prob
        self.audio_time_len = audio_time_len
        self.ovlp = ovlp
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sir = sir
        self.snr = snr
        self.num_spk = num_spk
        assert num_spk == 2, num_spk
        self.test_set = test_set
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size
        assert len(batch_size) == 2, batch_size
        if len(batch_size) <= 2:
            self.batch_size.append(1)

        rank_zero_info(f"dataset: LibriCSS \ntrain/valid/test/predict: {self.datasets}")
        rank_zero_info(f'batch size: train={self.batch_size[0]}; val={self.batch_size[1]}; test={self.batch_size[2]}')
        rank_zero_info(f'audio_time_length: train={self.audio_time_len[0]}; val={self.audio_time_len[1]}; test={self.audio_time_len[2]}')
        rank_zero_info(f'target: {self.target}')
        # assert self.batch_size_val == 1, "batch size for validation should be 1 as the audios have different length"
        if audio_time_len[2] is not None:
            rank_zero_warn("the length for test set is not None")

        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test]
        self.collate_func_train = collate_func_train
        self.collate_func_val = collate_func_val
        self.collate_func_test = collate_func_test

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, audio_time_len, seed, shuffle, batch_size, collate_fn):
        ds = LibriCSSDataset(
            libricss_dir=self.libricss_dir,
            librispeech_dir=self.librispeech_dir,
            reverb_dir=self.reverb_dir,
            rir_dir=self.rir_dir,
            target=self.target,
            dataset=dataset,  # 
            ovlp=self.ovlp,
            speech_overlap_ratio=self.speech_overlap_ratio,
            sir=self.sir,
            snr=self.snr,
            audio_time_len=audio_time_len,  # 
            spk1_prob=self.spk1_prob,
            spk2_prob=self.spk2_prob,
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
        if self.test_set == 'train':
            return self.train_dataloader()
        elif self.test_set == 'val':
            return self.val_dataloader()

        return self.construct_dataloader(
            dataset=self.datasets[2],
            audio_time_len=self.audio_time_len[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        dls = dict()
        for ds in self.datasets[3]:
            dls[ds] = self.construct_dataloader(
                dataset=ds,
                audio_time_len=None,
                seed=self.seeds[2],
                shuffle=False,
                batch_size=1,
                collate_fn=default_collate_func,
            )

        return dls


if __name__ == '__main__':
    """python -m data_loaders.libricss"""
    from jsonargparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_class_arguments(LibriCSSDataModule, nested_key='data')
    parser.add_argument('--save_dir', type=str, default='dataset/LibriCSS')
    parser.add_argument('--dataset', type=str, default='train')
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    args_dict = args.data
    args_dict['num_workers'] = 0  # for debuging
    args_dict['prefetch_factor'] = None  # for debuging
    datamodule = LibriCSSDataModule(**args_dict)
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
            if idx > 30:
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
