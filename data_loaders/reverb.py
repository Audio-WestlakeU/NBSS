import copy
import json
import math
import os
import random
from os.path import *
from pathlib import Path
from typing import *

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from scipy.signal import fftconvolve, resample_poly
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from data_loaders.utils.rand import randfloat, randint
from data_loaders.utils.window import (rectangular_window, reverberation_time_shortening_window)


def gen_obs(x: np.ndarray, RIR: np.ndarray, NOISE: np.ndarray, SNRdB: float) -> np.ndarray:
    """generate noisy reverberant data, translated from the original mathlab code

    Args:
        x: single channel clean wav
        RIR: shape [T, Chn]
        NOISE: shape [T, Chn]
        SNRdB: SNR

    Returns:
        np.ndarray: the generated wav, shape [Chn, T]
    """
    RIR = RIR.T
    # calculate direct+early reflection signal for calculating SNR
    delay = np.argmax(RIR[0, :])
    before_impulse = math.floor(16000 * 0.001)
    after_impulse = math.floor(16000 * 0.05)
    RIR_direct = RIR[0, delay - before_impulse:delay + after_impulse]
    direct_signal = fftconvolve(x, RIR_direct, mode='full', axes=-1)

    # obtain reverberant speech
    rev_y = fftconvolve(x[np.newaxis, ...], RIR, mode='full', axes=-1)
    T = rev_y.shape[-1]

    # normalize noise data according to the prefixed SNR value
    NOISE = NOISE[:T, :]
    NOISE_ref = NOISE[:, 0]

    iPn = 1. / np.mean(NOISE_ref**2, axis=0)
    Px = np.mean(direct_signal**2, axis=0)
    Msnr = np.sqrt((10**(-SNRdB / 10)) * iPn * Px)
    scaled_NOISE = NOISE * Msnr
    y = rev_y + scaled_NOISE.T
    y = y[:, delay:]
    return y


class ReverbTrainValDataset(Dataset):

    def __init__(
        self,
        wsjcam0_dir: str,
        reverb_dir: str,
        dataset: str,
        rir_dir: str = None,
        num_noises: int = 1,
        snr: Tuple[float, float] = [5, 20],
        audio_time_len: Optional[float] = None,
        sample_rate: int = 16000,
        use_real_rir_prop: float = 0.3,  # probability
        target: str = 'direct_path',
    ) -> None:
        """The Reverb dataset

        Args:
            wsjcam0_dir: a dir contains [data, docs], note: please first convert the .wv1 & .wv2 wavs to .wav 
            reverb_dir: a dir contains [REVERB_WSJCAM0, MC_WSJ_AV_Eval, MC_WSJ_AV_Dev, ...], the REVERB_WSJCAM0 contains the simulated training/validation/test sets.
            dataset: train_with_simulated_rir, val_with_simulated_rir
            num_noises: the number of noises used in one utterance
            snr: the SNR to mix the noise
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
            use_real_rir_prop: the probability to use the real-recorded RIRs
            target: direct_path or dry_source
        """
        super().__init__()
        assert dataset in ['train_with_simulated_rir', 'val_with_simulated_rir'], dataset
        assert target in ['direct_path', 'dry_source'], target
        assert use_real_rir_prop >= 0 and use_real_rir_prop <= 1, use_real_rir_prop
        assert use_real_rir_prop == 0.0, "disabled to use real rir"

        self.wsjcam0_dir = str(Path(wsjcam0_dir).expanduser())
        self.reverb_dir = str(Path(reverb_dir).expanduser())
        self.dataset = dataset
        self.use_real_rir_prop = use_real_rir_prop
        self.target = target

        if dataset in ['train_with_simulated_rir']:
            assert num_noises != None and num_noises > 0
            sources = []
            for file in ['configs/reverb/audio_si_tr.lst']:
                f = open(file, 'r')
                sources = sources + [x.strip() for x in f.readlines()]
                f.close()
            self.sources = [(str(Path(wsjcam0_dir).expanduser() / 'data') + x + '.wav') for x in sources]
            self.num_noises = num_noises
            self.rir_dir = Path(rir_dir) / 'train'
            self.simu_rirs = [str(x) for x in list(Path(self.rir_dir).expanduser().rglob('*.npz'))]
            self.real_rirs = [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_mcTrainData/RIR').glob('*.wav'))]
            self.noises = [
                [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_mcTrainData/NOISE').rglob('*SmallRoom*.wav'))],  # T60=0.25, for T60<0.35
                [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_mcTrainData/NOISE').rglob('*MediumRoom*.wav'))],  # T60=0.5, for T60<0.6
                [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_mcTrainData/NOISE').rglob('*LargeRoom*.wav'))],  # T60=0.7, for T60>=0.6
            ]
            self.noises.append(self.noises[0] + self.noises[1] + self.noises[2])  # for T60 is unknown
            self.noisy = [(str(Path(reverb_dir).expanduser() / 'REVERB_WSJCAM0/data/mc_train') + x + '.wav') for x in sources]
        elif dataset == 'val_with_simulated_rir':
            # the validation set with simulated rir
            sources = []
            for file in ['configs/reverb/audio_si_dt5a.lst', 'configs/reverb/audio_si_dt5b.lst']:
                f = open(file, 'r')
                sources = sources + [x.strip() for x in f.readlines()]
                f.close()
            self.sources = [(str(Path(wsjcam0_dir).expanduser() / 'data') + x + '.wav') for x in sources] * 2  # far and near

            self.num_noises = num_noises
            self.rir_dir = Path(rir_dir) / 'validation'
            self.simu_rirs = [str(x) for x in list(Path(self.rir_dir).expanduser().rglob('*.npz'))]
            # according to Generate_dtData.m, the *AnglA.wav is used for generating SimData/dt dataset
            self.real_rirs = [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/RIR').glob('*AnglA.wav'))]

            self.noises = [  # according to Generate_dtData.m, use Noise_SimRoom[1|2|3]*.wav
                [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/NOISE').rglob('Noise_SimRoom[1|2|3]*.wav'))],
            ]
            self.noisy = [(str(Path(reverb_dir).expanduser() / 'REVERB_WSJCAM0/data/near_test') + x + '.wav') for x in sources]
            self.noisy = self.noisy + [(str(Path(reverb_dir).expanduser() / 'REVERB_WSJCAM0/data/far_test') + x + '.wav') for x in sources]
        else:
            raise Exception(dataset)

        self.simu_rirs.sort()
        assert len(self.simu_rirs) > 0, f"the rir dir doesn't exist, or is empty: {self.rir_dir}"

        for l in self.noises:
            l.sort()
            assert len(l) > 0, f"the noise dir doesn't exist, or is empty: {str(Path(self.reverb_dir) / 'reverb_tools_for_Generate_mcTrainData/NOISE')}"

        assert len(self.noisy) == len(self.sources), (len(self.noisy), len(self.sources))

        self.audio_time_len = audio_time_len
        self.sr = sample_rate
        assert sample_rate == 16000, ('Not implemented for sample rate ', sample_rate)

        self.snr = snr

    def __getitem__(self, index_seed: Union[int, Tuple[int, int]]):
        if type(index_seed) == int:
            index = index_seed
            if self.dataset.startswith('val'):
                seed = index
            else:
                seed = random.randint(a=0, b=99999999)
        else:
            index, seed = index_seed

        g = torch.Generator()

        dataset = self.dataset
        original_index = index

        g.manual_seed(seed)
        # load source signal
        if dataset.startswith('train'):  # for train set, sample one source
            index = randint(g, low=0, high=len(self.sources))
        else:
            index = index
        source, srs = sf.read(self.sources[index])

        # load rir
        use_real_rir_prop = randfloat(g, low=0, high=1)  # sample one value from [0,1], if the value in range [0, self.use_real_rir_prop]
        if use_real_rir_prop < self.use_real_rir_prop:
            rir_index = randint(g, low=0, high=len(self.real_rirs))
            rir_path = self.real_rirs[rir_index]
            rir, sr_rir = sf.read(rir_path)
            rir = rir.T  # [nmic, time]
            channel_shift = randint(g, low=0, high=8)
            if channel_shift != 0:
                rir = np.concatenate([rir[channel_shift:, :], rir[:channel_shift, :]], axis=0)
            rir_dp = rir.copy()
            for chn in range(8):
                win = rectangular_window(rir=rir[chn, :], sr=sr_rir, time_before_after_max=0.002)
                rir_dp[chn, :] = rir[chn, :] * win
            rir_dict, spk_index = None, 0
        else:
            rir_index = randint(g, low=0, high=len(self.simu_rirs))
            rir_path = self.simu_rirs[rir_index]
            rir_dict = np.load(rir_path, allow_pickle=True)
            sr_rir = rir_dict['fs']
            rir = rir_dict['rir']  # shape [nsrc,nmic,time]
            spk_index = randint(g, low=0, high=rir.shape[0])  # sample one speaker location
            rir = rir[spk_index, ...]  # [nmic, time]
            rir_dp = rir_dict['rir_dp'][spk_index]
        assert self.sr == sr_rir and srs == self.sr, (srs, sr_rir)

        # convolve
        delay = np.argmax(rir[0, :])
        rvbt = fftconvolve(source[np.newaxis, ...], rir, mode='full', axes=-1)  # [nmic, time]
        rvbt = rvbt[:, delay:]  # time alignment with source

        # generate direct_path
        if self.target == 'direct_path':
            dp = fftconvolve(source[np.newaxis, ...], rir_dp, mode='full', axes=-1)  # take 0-th channel as the reference channel
            dp = dp[:, delay:]
        else:
            dp = source[np.newaxis, ...]

        #  pad or cut signals
        g.manual_seed(seed + 1)
        T = dp.shape[-1]
        rvbt = rvbt[:, :T]  # to the same length
        start = 0
        if self.audio_time_len:
            frames = int(self.sr * self.audio_time_len)
            if T < frames:
                rvbt = np.pad(rvbt, pad_width=((0, 0), (0, frames - T)), mode='constant', constant_values=0)
                dp = np.pad(dp, pad_width=((0, 0), (0, frames - T)), mode='constant', constant_values=0)
            elif T > frames:
                start = randint(g, low=0, high=T - frames)
                rvbt = rvbt[:, start:start + frames]
                dp = dp[:, start:start + frames]
        else:
            frames = rvbt.shape[-1]

        # add noise for train_with_simulated_rir and val_with_simulated_rir
        g.manual_seed(seed + 2)
        # sample noises
        noises = self.noises[-1]
        noise = None  # [chn, time]
        for i in range(self.num_noises):
            nidx = randint(g, low=0, high=len(noises))
            assert sf.info(noises[nidx]).frames >= frames, (sf.info(noises[nidx]).frames, frames)
            nstart = randint(g, low=0, high=sf.info(noises[nidx]).frames - frames)
            nwav, srn = sf.read(noises[nidx], frames=frames, start=nstart)
            assert srn == self.sr, srn
            scale = randfloat(g, low=0.1, high=10)
            nwav *= scale
            nwav = nwav.T
            # channel shift
            channel_shift = randint(g, low=0, high=8)
            if channel_shift != 0:
                nwav = np.concatenate([nwav[channel_shift:, :], nwav[:channel_shift, :]], axis=0)
            # add sampled noises
            if noise is None:
                noise = nwav
            else:
                noise += nwav
        # add noise
        assert self.snr is not None
        snr_this = randfloat(g, low=self.snr[0], high=self.snr[1])
        iPn = 1. / np.mean(noise[0, :]**2, axis=0)
        Px = np.mean(rvbt[0, :]**2, axis=0)
        Msnr = np.sqrt((10**(-snr_this / 10)) * iPn * Px)
        noise *= Msnr
        noisy = rvbt + noise

        paras = {
            'index': original_index,
            'seed': seed,
            'noisy': None if dataset in ['train_with_simulated_rir', 'val_with_simulated_rir'] else self.noisy[index],
            'noise': noises[nidx] if dataset in ['train_with_simulated_rir', 'val_with_simulated_rir'] else None,
            'source': self.sources[index],
            'sample_rate': 16000,
            'dataset': 'Reverb_' + dataset,
            'audio_time_len': self.audio_time_len,
            'start': start,
            'rir': None if dataset not in ['train_with_simulated_rir', 'val_with_simulated_rir'] else (str(rir_path), spk_index),
            'data': {  # put array in data
                'direct_path': dp[np.newaxis, ...].astype(np.float32),
                'reverberant_image': rvbt[np.newaxis, ...].astype(np.float32),
                'rir_dict': dict(rir_dict),
            }
        }

        return torch.as_tensor(noisy, dtype=torch.float32), torch.as_tensor(dp[np.newaxis, ...], dtype=torch.float32), paras

    def __len__(self):
        if self.dataset in ['train_with_simulated_rir']:
            return 20000
        else:
            return len(self.noisy)


class ReverbEtDataset(Dataset):
    """The Reverb Development/Evaluation dataset"""

    def __init__(
        self,
        reverb_dir: str,
        dataset: str,
        subdataset: str = 'Et',
        num_chns: int = 8,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        assert dataset in ['Real', 'Sim'], dataset
        assert num_chns in [1, 2, 8], num_chns

        self.reverb_dir = str(Path(reverb_dir).expanduser())
        self.dataset = dataset
        self.subdataset = subdataset

        self.mics = {1: ['A'], 2: ['A', 'B'], 8: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']}[num_chns]
        self.num_chns = num_chns

        taskfile_dir = f'configs/reverb/taskFiles_' + ({'Dt': 'tr_dt', 'Et': 'et'}[subdataset]) + f'/{num_chns}ch'
        takefile_names = f"{dataset}Data_{subdataset.lower()}_for_{num_chns}ch_*_A"
        tdir = Path(taskfile_dir)

        taskfiles = list(tdir.rglob(takefile_names))
        self.audio_files = []
        for taskfile in taskfiles:
            data = []
            for mic in self.mics:
                f = open(str(taskfile).replace('_A', '_' + mic), 'r')
                lines = [x.strip() for x in f.readlines()]
                f.close()
                data.append(lines)
            self.audio_files += list(zip(*data))
        self.sr = sample_rate
        assert sample_rate == 16000, ('Not implemented for sample rate ', sample_rate)

        self.subdir = {'Real': {'Et': 'MC_WSJ_AV_Eval', 'Dt': 'MC_WSJ_AV_Dev'}[subdataset], 'Sim': 'REVERB_WSJCAM0/data'}[dataset]

    def __getitem__(self, index_seed: Union[int, Tuple[int, int]]):
        if type(index_seed) == int:
            index = index_seed
        else:
            index, _ = index_seed

        all_chns = []
        for file in self.audio_files[index]:
            chn, sr = sf.read(self.reverb_dir + f'/{self.subdir}/' + file)
            assert sr == 16000, sr
            all_chns.append(chn)
        noisy = np.stack(all_chns, axis=0)  # e.g. [8, T]

        paras = {
            'index': index,
            # 'noisy': self.dataset,
            'source': self.audio_files[index],
            'sample_rate': 16000,
            'dataset': 'Reverb' + self.dataset + self.subdataset,
            'saveto': self.subdir + '/' + self.audio_files[index][0],
        }

        return torch.as_tensor(noisy, dtype=torch.float32), None, paras

    def __len__(self):
        return len(self.audio_files)


class ReverbSimDtEtMCDataset(Dataset):
    """ Difference between ReverbSimDtEtMCDataset and ReverbEtDataset: 
    - ReverbSimDtEtMCDataset returns synthesized SimEt dataset with multichannel direct-path/rvbt reference signal
    - ReverbEtDataset returns the original Reverb SimDt/SimEt/RealDt/RealEt dataset
    """

    def __init__(
        self,
        wsjcam0_dir: str,
        reverb_dir: str,
        dataset: str,
        snr: Tuple[float, float] = [20, 20],
        audio_time_len: Optional[float] = None,
        sample_rate: int = 16000,
    ) -> None:
        """
        Args:
            wsjcam0_dir: a dir contains [data, docs], note: please first convert the .wv1 & .wv2 wavs to .wav 
            reverb_dir: a dir contains [REVERB_WSJCAM0, MC_WSJ_AV_Eval, MC_WSJ_AV_Dev, ...], the REVERB_WSJCAM0 contains the simulated training/validation/test sets.
            dataset: 'SimDtMC', 'SimEtMC'
            snr: the SNR to mix the noise
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert dataset in ['SimDtMC', 'SimEtMC'], dataset
        assert audio_time_len is None, "audio_time_len is not enabled in this dataset"

        self.wsjcam0_dir = str(Path(wsjcam0_dir).expanduser())
        self.reverb_dir = str(Path(reverb_dir).expanduser())
        self.dataset = dataset

        if dataset == 'SimDtMC':
            sources = []
            for file in ['configs/reverb/audio_si_dt5a.lst', 'configs/reverb/audio_si_dt5b.lst']:
                f = open(file, 'r')
                sources = sources + [x.strip() for x in f.readlines()]
                f.close()
            self.sources = [(str(Path(wsjcam0_dir).expanduser() / 'data') + x + '.wav') for x in sources] * 2  # far and near

            # according to Generate_dtData.m, the *AnglA.wav is used for generating SimData/dt dataset
            rirs_near = [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/RIR').glob('*near*AnglA.wav'))]
            rirs_far = [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/RIR').glob('*far*AnglA.wav'))]
            # according to Generate_dtData.m, use Noise_SimRoom[1|2|3]*.wav
            self.noises = [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/NOISE').rglob('Noise_SimRoom[1|2|3]*.wav'))]
            self.noisy = [(str(Path(reverb_dir).expanduser() / 'REVERB_WSJCAM0/data/near_test') + x + '.wav') for x in sources]
            self.noisy = self.noisy + [(str(Path(reverb_dir).expanduser() / 'REVERB_WSJCAM0/data/far_test') + x + '.wav') for x in sources]
        elif dataset == 'SimEtMC':
            sources = []
            for file in ['configs/reverb/audio_si_et_1.lst', 'configs/reverb/audio_si_et_2.lst']:
                f = open(file, 'r')
                sources = sources + [x.strip() for x in f.readlines()]
                f.close()
            self.sources = [(str(Path(wsjcam0_dir).expanduser() / 'data') + x + '.wav') for x in sources] * 2  # far and near

            # according to Generate_etData.m, the *AnglB.wav is used for generating SimData/et dataset
            rirs_near = [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/RIR').glob('*near*AnglB.wav'))]
            rirs_far = [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/RIR').glob('*far*AnglB.wav'))]
            # according to Generate_etData.m, use Noise_SimRoom[1|2|3]*.wav
            self.noises = [str(x) for x in list((Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/NOISE').rglob('Noise_SimRoom[1|2|3]*.wav'))]

            self.noisy = [(str(Path(reverb_dir).expanduser() / 'REVERB_WSJCAM0/data/near_test') + x + '.wav') for x in sources]
            self.noisy = self.noisy + [(str(Path(reverb_dir).expanduser() / 'REVERB_WSJCAM0/data/far_test') + x + '.wav') for x in sources]
        else:
            raise Exception(dataset)

        # sort noises and rirs to make sure that the noises have the same order in different machines
        self.noises.sort()
        assert len(self.noises) > 0, f"the noise dir doesn't exist, or is empty: {str(Path(self.reverb_dir) / 'reverb_tools_for_Generate_mcTrainData/NOISE')}"
        rirs_near.sort()
        rirs_far.sort()
        assert len(rirs_near) > 0 and len(rirs_far) > 0, f"the rir dir doesn't exist, or is empty: {str(Path(self.reverb_dir) / 'reverb_tools_for_Generate_SimData/RIR')}"
        # repeat rirs
        N = len(self.noisy) / 2
        n_repeat = math.ceil(N / len(rirs_near))
        rirs_near = (rirs_near * n_repeat)[:int(N)]
        n_repeat = math.ceil(N / len(rirs_far))
        rirs_far = (rirs_far * n_repeat)[:int(N)]
        self.real_rirs = rirs_near + rirs_far

        assert len(self.noisy) == len(self.sources) and len(self.real_rirs) == len(self.noisy), (len(self.noisy), len(self.sources), len(self.real_rirs))

        self.audio_time_len = audio_time_len
        self.sr = sample_rate
        assert sample_rate == 16000, ('Not implemented for sample rate ', sample_rate)
        self.snr = snr

    def __getitem__(self, index_seed: Union[int, Tuple[int, int]]):
        if type(index_seed) == int:
            index = index_seed
            seed = index
        else:
            index, seed = index_seed

        g = torch.Generator()
        g.manual_seed(seed)
        # load source signal
        source, srs = sf.read(self.sources[index])

        # load rir
        rir_path = self.real_rirs[index]
        rir, sr_rir = sf.read(rir_path)
        rir = rir.T  # [nmic, time]
        channel_shift = randint(g, low=0, high=8)
        if channel_shift != 0:
            rir = np.concatenate([rir[channel_shift:, :], rir[:channel_shift, :]], axis=0)
        spk_index = 0
        # convolve
        rvbt = fftconvolve(source[np.newaxis, ...], rir, mode='full', axes=-1)  # [nmic, time]
        delay = np.argmax(rir[0, :])
        rvbt = rvbt[:, delay:]  # time alignment with source

        # direct path
        rir_dp = rir.copy()
        for chn in range(rir.shape[0]):
            win = rectangular_window(rir=rir[chn, :], sr=sr_rir, time_before_after_max=0.002)
            rir_dp[chn, :] = rir[chn, :] * win
        dp = fftconvolve(source[np.newaxis, ...], rir_dp, mode='full', axes=-1)  # [nmic, time]
        dp = dp[:, delay:]
        frames = rvbt.shape[-1]

        # sample noises
        g.manual_seed(seed + 2)
        noises = self.noises
        nidx = randint(g, low=0, high=len(noises))
        assert sf.info(noises[nidx]).frames >= frames, (sf.info(noises[nidx]).frames, frames)
        nstart = randint(g, low=0, high=sf.info(noises[nidx]).frames - frames)
        noise, srn = sf.read(noises[nidx], frames=frames, start=nstart)
        noise = noise.T  # [chn, time]
        assert noise.shape[0] == 8, noise.shape
        # channel shift
        channel_shift = randint(g, low=0, high=8)
        if channel_shift != 0:
            noise = np.concatenate([noise[channel_shift:, :], noise[:channel_shift, :]], axis=0)
        # add noise
        assert self.snr is not None
        snr_this = randfloat(g, low=self.snr[0], high=self.snr[1])
        iPn = 1. / np.mean(noise[0, :]**2, axis=0)
        Px = np.mean(rvbt[0, :]**2, axis=0)
        Msnr = np.sqrt((10**(-snr_this / 10)) * iPn * Px)
        noise *= Msnr
        noisy = rvbt + noise

        paras = {
            'index': index,
            'seed': seed,
            'noisy': self.noisy[index],
            'saveto': [self.noisy[index].replace(self.reverb_dir, '')[1:].replace('.wav', '_ch1.wav')],
            'noise': noises[nidx],
            'SNR': snr_this,
            'source': self.sources[index],
            'sample_rate': 16000,
            'dataset': 'Reverb_' + self.dataset,
            'audio_time_len': self.audio_time_len,
            'rir': (str(rir_path), spk_index),
            'data': {  # put array in data
                'direct_path': dp[np.newaxis, ...].astype(np.float32),
                'reverberant_image': rvbt[np.newaxis, ...].astype(np.float32),
            }
        }

        return torch.as_tensor(noisy, dtype=torch.float32), torch.as_tensor(dp[np.newaxis, ...], dtype=torch.float32), paras  # shape [C,T], [C,Spk,T],

    def __len__(self):
        return len(self.noisy)


class ReverbDataModule(LightningDataModule):

    def __init__(
        self,
        wsjcam0_dir: str,  # a dir contains [data, docs], note: please first convert the .wv1 & .wv2 wavs to .wav 
        reverb_dir: str,  # a dir contains [REVERB_WSJCAM0, MC_WSJ_AV_Eval, MC_WSJ_AV_Dev, ...], the REVERB_WSJCAM0 contains the simulated training/validation/test sets.
        datasets: Tuple[str, str, str] = ('train_with_simulated_rir', 'val_with_simulated_rir', 'SimEtMC'),  # datasets for train/val/test
        use_real_rir_prop: float = 0,  # probability
        rir_dir: str = None,  # containing train, validation, and test subdirs
        num_noises: int = 1,  # the number of noises used in 'train_with_simulated_rir' or 'val_with_simulated_rir'
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None],  # audio_time_len (seconds) for training, val, test.
        snr: Tuple[float, float] = [5, 20],  # SNR dB
        batch_size: List[int] = [1, 1],
        num_workers: int = 5,
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
        self.wsjcam0_dir = wsjcam0_dir
        self.reverb_dir = reverb_dir
        self.rir_dir = rir_dir
        self.num_noises = num_noises
        self.audio_time_len = audio_time_len
        self.use_real_rir_prop = use_real_rir_prop
        self.snr = snr
        self.persistent_workers = persistent_workers

        self.datasets = datasets
        rank_zero_info(f'dataset: Reverb \ntrain/valid/test: {self.datasets}')
        rank_zero_info(f"audio time length: {audio_time_len}")
        rank_zero_info(f'probability for using real rirs in training set: {self.use_real_rir_prop}')
        rank_zero_info(f"SNR: {str(snr)}, num_noises={num_noises}")

        # assert self.datasets[2] in ['SimEtMC', 'SimDtMC', 'SimEtMC+SimDtMC'], self.datasets[2]
        # assert audio_time_len[2] is None, audio_time_len

        self.batch_size = batch_size[0]
        self.batch_size_val = batch_size[1]
        self.batch_size_test = 1
        if len(batch_size) > 2:
            self.batch_size_test = batch_size[2]
        rank_zero_info(f'batch size: train={self.batch_size}; val={self.batch_size_val}; test={self.batch_size_test}')
        # assert self.batch_size_val == 1, "batch size for validation should be 1 as the audios have different length"

        self.num_workers = num_workers

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

    def train_dataloader(self) -> DataLoader:
        self.train = ReverbTrainValDataset(
            wsjcam0_dir=self.wsjcam0_dir,
            reverb_dir=self.reverb_dir,
            rir_dir=self.rir_dir,
            num_noises=self.num_noises,
            snr=self.snr,
            dataset=self.datasets[0],
            audio_time_len=self.audio_time_len[0] if self.current_stage != 'test' else None,
            use_real_rir_prop=self.use_real_rir_prop,
        )

        return DataLoader(
            self.train,
            sampler=MyDistributedSampler(self.train, seed=self.seeds[0], shuffle=True),
            batch_size=self.batch_size,
            collate_fn=self.collate_func_train,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        self.val = ReverbTrainValDataset(
            wsjcam0_dir=self.wsjcam0_dir,
            reverb_dir=self.reverb_dir,
            rir_dir=self.rir_dir,
            num_noises=self.num_noises,
            snr=self.snr,
            dataset=self.datasets[1],
            audio_time_len=self.audio_time_len[1] if self.current_stage != 'test' else None,
            use_real_rir_prop=self.use_real_rir_prop,
        )

        return DataLoader(
            self.val,
            sampler=MyDistributedSampler(self.val, seed=self.seeds[1], shuffle=False),
            batch_size=self.batch_size_val,
            collate_fn=self.collate_func_val,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        prefetch_factor = 2

        if self.datasets[2] in ['SimEtMC', 'SimDtMC', 'SimEtMC+SimDtMC']:
            datasets = [self.datasets[2]]
            if self.datasets[2] == 'SimEtMC+SimDtMC':
                datasets = ['SimEtMC', 'SimDtMC']
            self.test = dict()
            dls = dict()
            for ds in datasets:
                dataset = ReverbSimDtEtMCDataset(
                    wsjcam0_dir=self.wsjcam0_dir,
                    reverb_dir=self.reverb_dir,
                    dataset=ds,
                    snr=[20, 20],
                    audio_time_len=None,
                )
                dls[ds] = DataLoader(
                    dataset,
                    sampler=MyDistributedSampler(dataset, seed=self.seeds[2], shuffle=False),
                    batch_size=self.batch_size_test,
                    collate_fn=self.collate_func_test,
                    num_workers=3,
                    prefetch_factor=prefetch_factor,
                )
                self.test[ds] = dataset

            return dls
        else:
            self.test = ReverbTrainValDataset(
                wsjcam0_dir=self.wsjcam0_dir,
                reverb_dir=self.reverb_dir,
                rir_dir=self.rir_dir,
                num_noises=self.num_noises,
                snr=self.snr,
                dataset=self.datasets[2],  # val
                audio_time_len=self.audio_time_len[2],
                use_real_rir_prop=self.use_real_rir_prop,
            )
            return DataLoader(
                self.test,
                sampler=MyDistributedSampler(self.test, seed=self.seeds[2], shuffle=False),
                batch_size=self.batch_size_test,
                collate_fn=self.collate_func_test,
                num_workers=3,
                prefetch_factor=prefetch_factor,
            )

    def predict_dataloader(self) -> DataLoader:
        real_et = ReverbEtDataset(
            reverb_dir=self.reverb_dir,
            dataset='Real',
            subdataset='Et',
            num_chns=8,
            sample_rate=16000,
        )
        sim_et = ReverbEtDataset(
            reverb_dir=self.reverb_dir,
            dataset='Sim',
            subdataset='Et',
            num_chns=8,
            sample_rate=16000,
        )
        real_dt = ReverbEtDataset(
            reverb_dir=self.reverb_dir,
            dataset='Real',
            subdataset='Dt',
            num_chns=8,
            sample_rate=16000,
        )
        sim_dt = ReverbEtDataset(
            reverb_dir=self.reverb_dir,
            dataset='Sim',
            subdataset='Dt',
            num_chns=8,
            sample_rate=16000,
        )

        return {
            'RealEt': DataLoader(real_et, batch_size=1, collate_fn=default_collate_func, num_workers=3, shuffle=False),
            'SimEt': DataLoader(sim_et, batch_size=1, collate_fn=default_collate_func, num_workers=3, shuffle=False),
            'RealDt': DataLoader(real_dt, batch_size=1, collate_fn=default_collate_func, num_workers=3, shuffle=False),
            'SimDt': DataLoader(sim_dt, batch_size=1, collate_fn=default_collate_func, num_workers=3, shuffle=False),
        }


if __name__ == '__main__':
    """python -m data_loaders.reverb"""
    from argparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_argument('--wsjcam0_dir', type=str, default='~/datasets/wsjcam0')
    parser.add_argument('--reverb_dir', type=str, default='~/datasets/Reverb')
    parser.add_argument('--save_dir', type=str, default='dataset/Reverb')
    parser.add_argument('--rir_dir', type=str, default='~/datasets/Reverb_rirs')
    parser.add_argument('--dataset', type=str, default='train_with_simulated_rir')  #, choices=['train', 'train_with_simulated_rir', 'val_with_simulated_rir', 'val', 'SimEt', 'SimEtMC+SimDtMC'])
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    if args.dataset == 'SimEtMC+SimDtMC':
        datamodule = ReverbDataModule(
            args.wsjcam0_dir,
            args.reverb_dir,
            rir_dir=args.rir_dir,
            batch_size=[1, 1],
            datasets=['train_with_simulated_rir', 'val_with_simulated_rir', args.dataset],
            num_workers=1,
        )
    else:
        datamodule = ReverbDataModule(args.wsjcam0_dir, args.reverb_dir, rir_dir=args.rir_dir, batch_size=[1, 1], datasets=[args.dataset, 'val_with_simulated_rir', 'SimEtMC'], num_workers=1)
    if args.dataset not in ['RealEt', 'SimEt', 'RealDt', 'SimDt']:
        datamodule.setup()

    if args.dataset.startswith('train') or args.dataset.startswith('val_') or args.dataset == 'SimEtMC+SimDtMC':
        dataloader = datamodule.train_dataloader() if args.dataset != 'SimEtMC+SimDtMC' else datamodule.test_dataloader()
    elif args.dataset == 'val':
        dataloader = datamodule.val_dataloader()
    elif args.dataset in ['RealEt', 'SimEt', 'RealDt', 'SimDt']:
        dataloader = datamodule.predict_dataloader()[args.dataset]
    else:
        assert args.dataset == 'val', args.dataset
        dataloader = datamodule.test_dataloader()

    if type(dataloader) != dict:
        dataloaders = {args.dataset: dataloader}
    else:
        dataloaders = dataloader

    for ds, dataloader in dataloaders.items():
        for idx, (noisy, tar, paras) in enumerate(dataloader):
            print(f'{idx}/{len(dataloader)}', end=' ')
            if idx > 10:
                continue

            # dp = paras[0]['data']['direct_path']
            # rvbt = paras[0]['data']['reverberant_image']
            # rir_dict = paras[0]['data']['rir_dict']

            # sf.write(f'dataset/IPD/{idx}_noisy.wav', noisy[0].T, samplerate=16000)
            # sf.write(f'dataset/IPD/{idx}_dp.wav', dp[0].T, samplerate=16000)
            # sf.write(f'dataset/IPD/{idx}_rvbt.wav', rvbt[0].T, samplerate=16000)
            # np.savez_compressed(f'dataset/IPD/{idx}_rir.npz', **rir_dict)

            # write target to dir
            if args.gen_target and args.dataset not in ['RealEt', 'SimEt', 'RealDt', 'SimDt']:
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/target").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(tar[0, :, 0, :].numpy())) <= 1
                assert tar.shape[1] == 1, tar.shape
                sp = tar_path / basename(paras[0]['source'])
                if not sp.exists():
                    sf.write(sp, tar[0, 0, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            # write unprocessed's 0-th channel
            if args.gen_unprocessed:
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noisy").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(noisy[0, 0, :].numpy())) <= 1
                if args.dataset in ['RealEt', 'SimEt', 'RealDt', 'SimDt']:
                    sp = tar_path / basename(paras[0]['source'][0])
                else:
                    sp = tar_path / basename(paras[0]['source'])
                if not sp.exists():
                    sf.write(sp, noisy[0, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            print(noisy.shape, None if args.dataset in ['RealEt', 'SimEt', 'RealDt', 'SimDt'] else tar.shape, paras)
