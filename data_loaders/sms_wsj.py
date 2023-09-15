import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from scipy.signal import correlate
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import copy
from scipy.signal import resample_poly
from os.path import *


def get_shift_samples(a, b):
    assert a.shape == b.shape

    if len(a.shape) == 1:
        xcorr = correlate(a, b)
        max_abs_val_idx = np.argmax(np.abs(xcorr))
        shift = max_abs_val_idx - a.shape[0] + 1
        # return shift
        return [shift, xcorr[max_abs_val_idx], xcorr[a.shape[0] - 1]]
    elif len(a.shape) == 2:
        return [get_shift_samples(a[i,], b[i,]) for i in range(a.shape[0])]
    elif len(a.shape) == 3:
        return [get_shift_samples(a[i, j,], b[i, j,]) for j in range(a.shape[1]) for i in range(a.shape[0])]


def default_collate_func(batches: List[Tuple[Tensor, Tensor, Dict[str, Any]]]) -> List[Any]:
    mini_batch = []
    for x in zip(*batches):
        if isinstance(x[0], np.ndarray):
            x = [torch.tensor(x[i]) for i in range(len(x))]
        if isinstance(x[0], Tensor):
            x = torch.stack(x)
        mini_batch.append(x)
    return mini_batch


def reverberation_time_shortening_window(rir: np.ndarray, original_T60: float, target_T60: float, sr: int = 8000, time_after_max: float = 0.002, time_before_max: float = None) -> np.ndarray:
    """shorten the T60 of a given rir"""
    assert rir.ndim == 1, rir.ndim

    if original_T60 <= target_T60:
        return np.ones(shape=rir.shape)
    q = 3 / (target_T60 * sr) - 3 / (original_T60 * sr)
    idx_max = int(np.argmax(np.abs(rir)))
    N1 = int(idx_max + time_after_max * sr)
    win = np.empty(shape=rir.shape, dtype=np.float64)
    win[:N1] = 1
    win[N1:] = 10**(-q * np.arange(rir.shape[0] - N1))
    if time_before_max:
        N0 = int(idx_max - time_before_max * sr)
        if N0 > 0:
            win[:N0] = 0
    return win


def rectangular_window(rir: np.ndarray, sr: int = 8000, time_before_after_max: float = 0.002) -> np.ndarray:
    assert rir.ndim == 1, rir.ndim
    idx = int(np.argmax(np.abs(rir)))
    win = np.zeros(shape=rir.shape)
    N = int(sr * time_before_after_max)
    win[max(0, idx - N):idx + N + 1] = 1
    return win


class SmsWsjDataset(Dataset):

    def __init__(self, sms_wsj_dir: str, target: str, dataset: str, audio_time_len: Optional[float] = None, ref_channel: int = 0, num_spk: int = 2, fuss_dir: str = None) -> None:
        """The SMS-WSJ dataset

        Args:
            sms_wsj_dir: a dir contains [early, noise, observation, rirs, speech_source, tail, wsj_8k_zeromean]
            target:  early, image (i.e. early+tail), speech_source, direct_path
            dataset: train_si284, cv_dev93, test_eval92
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
            num_spk: the number of speakers, if num_spk==2, then speech + speech. if num_spk==1, then speech + non-speech. if num_spk==2, then non-speech + non-speech.
            fuss_dir: a dir contains [fsd_data,...]. Is used to test the spectral generalization ability of models.
        """
        super().__init__()
        assert target in ['early', 'image', 'speech_source', 'direct_path'] or target.startswith('direct_path') or target.startswith('RTS'), target
        assert dataset in ['train_si284', 'cv_dev93', 'test_eval92'], dataset
        self.sms_wsj_dir = Path(sms_wsj_dir).expanduser()
        self.target = target
        self.dataset = dataset
        self.audio_time_len = audio_time_len
        self.ref_channel = ref_channel

        with open(self.sms_wsj_dir / 'sms_wsj.json', 'r') as f:
            d = json.load(f)
            self.dataset_info = d['datasets'][dataset]

        obs_path = self.sms_wsj_dir / 'observation' / dataset
        self.observations = list(obs_path.rglob('*.wav'))
        self.observations.sort()

        self.num_spk = num_spk
        assert num_spk in [0, 1, 2], num_spk
        if num_spk != 2:
            assert self.target == 'direct_path', "target should be direct_path, but " + self.target + " is given"
            assert fuss_dir is not None
            foreground_file = open(Path(fuss_dir).expanduser() / 'fsd_data' / ({'train_si284': 'train', 'cv_dev93': 'validation', 'test_eval92': 'eval'}[dataset] + '_foreground.txt'), 'r')
            lines = [l.strip() for l in foreground_file.readlines()]
            foreground_file.close()
            self.non_speeches = [Path(fuss_dir).expanduser() / 'fsd_data' / l for l in lines]
            if len(self.observations) == len(self.non_speeches):
                self.non_speeches = self.non_speeches[:-1]  # remove one non-speech to make the indexes used not equal
            assert len(self.non_speeches) > 0, f"dir is empty, {fuss_dir}"

        self.non_speech_dir = fuss_dir

    def __getitem__(self, index):
        name = self.observations[index % len(self.observations)].name
        info = copy.deepcopy(self.dataset_info[name.removesuffix('.wav')])
        target_name = [name.replace('.wav', '_0.wav'), name.replace('.wav', '_1.wav')]

        if self.target in ['early', 'speech_source']:
            mix, sr = sf.read(self.observations[index])
            mix = mix.T  # [T, C]
            target_1, sr_1 = sf.read(self.sms_wsj_dir / self.target / self.dataset / name.replace('.wav', '_0.wav'), always_2d=True, dtype='float32')
            target_2, sr_2 = sf.read(self.sms_wsj_dir / self.target / self.dataset / name.replace('.wav', '_1.wav'), always_2d=True, dtype='float32')

            target = np.stack([target_1.T, target_2.T], axis=0)  # [Spk, C, T]
        elif self.target == 'image-from-file':  # image-from-file
            mix, sr = sf.read(self.observations[index])
            mix = mix.T  # [T, C]
            target_1_early, sr_1 = sf.read(self.sms_wsj_dir / 'early' / self.dataset / name.replace('.wav', '_0.wav'), always_2d=True, dtype='float32')
            target_1_tail, sr_1 = sf.read(self.sms_wsj_dir / 'tail' / self.dataset / name.replace('.wav', '_0.wav'), always_2d=True, dtype='float32')
            target_1 = target_1_early + target_1_tail

            target_2_early, sr_2 = sf.read(self.sms_wsj_dir / 'early' / self.dataset / name.replace('.wav', '_1.wav'), always_2d=True, dtype='float32')
            target_2_tail, sr_2 = sf.read(self.sms_wsj_dir / 'tail' / self.dataset / name.replace('.wav', '_1.wav'), always_2d=True, dtype='float32')
            target_2 = target_2_early + target_2_tail

            target = np.stack([target_1.T, target_2.T], axis=0)  # [Spk, C, T]
        else:
            # generate audio data by convolving rir and clean speech signals
            original_source_list = []
            rir_list = []
            for os_path, rir_path in zip(info['audio_path']['original_source'], info['audio_path']['rir']):
                os_path = self.sms_wsj_dir / ('wsj_8k_zeromean' + os_path.split('wsj_8k_zeromean')[-1])
                original_source, sr_os = sf.read(os_path, dtype='float64')
                original_source_list.append(original_source.T)

                rir_path = self.sms_wsj_dir / ('rirs' + rir_path.split('rirs')[-1])
                rir, sr_rir = sf.read(rir_path, dtype='float64')
                rir_list.append(rir.T)

            # replace speech with non-speech
            if self.num_spk != 2:
                original_source_list, target_name = self.replace_with_non_speech(index, original_source_list, sr_os, target_name, info)

            info['audio_data'] = {}
            info['audio_data']['original_source'] = original_source_list
            rirs = np.stack(rir_list)
            info['audio_data']['rir'] = rirs

            if self.target == 'direct_path':
                # read simulated direct-path rir
                rir_list = []
                for rir_path in info['audio_path']['rir']:
                    rir_path = self.sms_wsj_dir / ('rirs_direct_path' + rir_path.split('rirs')[-1])
                    rir, sr_rir = sf.read(rir_path, dtype='float64')
                    rir_list.append(rir.T)
                dp_rir = np.stack(rir_list)

                info = scenario_map_fn(example=info, add_speech_image=False, rir_target=dp_rir)
                target = info['audio_data']['speech_target']

            elif self.target.startswith('direct_path') and self.target.endswith('ms'):
                # e.g. direct_path_2ms, uses a rectangular-windowed version rir to compute the direct-path signal.
                time_before_after_max = float(self.target.split('_')[-1].replace('ms', '').strip()) / 1000
                win = np.stack([rectangular_window(r, sr=8000, time_before_after_max=time_before_after_max) for r in rirs[:, self.ref_channel, :]])  # takes the 0-th channel as reference channel
                rir_winded = rirs[:, self.ref_channel, :] * win
                info = scenario_map_fn(example=info, add_speech_image=False, rir_target=rir_winded)
                target = info['audio_data']['speech_target']

            elif self.target.startswith('RTS') and self.target.endswith('s'):
                # e.g. RTS_0.15s, uses a reverberation time shortening rir to compute the target signal.
                target_T60 = float(self.target.split('_')[-1].replace('s', '').strip())
                win = np.stack([reverberation_time_shortening_window(r, original_T60=info['sound_decay_time'], target_T60=target_T60, sr=8000) for r in rirs[:, self.ref_channel, :]])
                rir_winded = rirs[:, self.ref_channel, :] * win
                info = scenario_map_fn(example=info, add_speech_image=False, rir_target=rir_winded)
                target = info['audio_data']['speech_target']

            elif self.target == 'image':  # image
                info = scenario_map_fn(example=info, add_speech_image=True)
                target = info['audio_data']['speech_image']
            else:
                raise NotImplementedError('Unknown target: ' + self.target)
            mix = info['audio_data']['observation']
            if self.num_spk == 2:  # check if the generated wav equals to the SMS-WSJ or not
                assert np.allclose(sf.read(self.observations[index])[0].T, mix), "the generated observation doesn't equal to the SMS-WSJ's"

        if self.audio_time_len:
            needed_frames = int(self.audio_time_len * 8000)
            if mix.shape[-1] < needed_frames:
                left = np.random.randint(0, needed_frames - mix.shape[-1])
                right = needed_frames - mix.shape[-1] - left

                mix = np.pad(mix, pad_width=((0, 0), (left, right)), mode='constant', constant_values=0)
                target = np.pad(target, pad_width=((0, 0), (0, 0), (left, right)), mode='constant', constant_values=0)
            elif mix.shape[-1] > needed_frames:
                audio_start = info['offset']
                audio_end = [info['offset'][i] + info['num_samples']['original_source'][i] for i in [0, 1]]

                for sec in [1, 1.5, 2, 2.5, 3]:  # allow sec seconds non-overlapping in left/right side of mixture
                    rand_min = max(0, max(audio_start) - int(sec * 8000))
                    rand_max = min(mix.shape[-1] - needed_frames, min(audio_end) - int((self.audio_time_len - sec) * 8000))
                    if rand_min < rand_max:
                        break
                if rand_max <= rand_min:
                    rand_min, rand_max = 0, mix.shape[-1] - needed_frames
                left = np.random.randint(rand_min, rand_max)
                right = needed_frames + left

                mix = mix[:, left:right]
                target = target[:, :, left:right]

        paras = {
            'index': index,
            'wavname': name,
            'mix_path': str(self.observations[index % len(self.observations)]),
            'saveto': target_name,
            'target': self.target,
            'sample_rate': 8000,
            'dataset': 'SMS-WSJ',
            'audio_path': info['audio_path'],
        }
        del info
        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(target, dtype=torch.float32), paras

    def __len__(self):
        if self.num_spk == 2:
            return len(self.observations)
        elif self.num_spk == 1:
            return len(self.observations) * 2  # doubles the dataset to iterate over all the utterances in SMS-WSJ
        else:
            assert self.num_spk == 0, self.num_spk
            return len(self.observations) * 2

    def replace_with_non_speech(
        self,
        index: int,
        original_source_list: List[np.ndarray],
        sr_os: int,
        target_name: List[str],
        info: Dict,
    ) -> Tuple[List[np.ndarray], List[str]]:
        if self.dataset == 'train_si284':
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(index)

        if self.num_spk == 1:
            i = rng.integers(0, len(self.non_speeches)) if self.dataset == 'train_si284' else (index % len(self.non_speeches))
            if index // len(self.observations) == 0:  # keep first speech
                non_speech_index = [None, i]
            else:  # keep second speech
                non_speech_index = [i, None]
        else:
            assert self.num_spk == 0, self.num_spk
            if self.dataset == 'train_si284':
                non_speech_index = [rng.integers(0, len(self.non_speeches)), rng.integers(0, len(self.non_speeches))]
            else:
                non_speech_index = [index % len(self.non_speeches), (len(self.observations) + index) % len(self.non_speeches)]

        # read non_speeches
        original_source_list_2 = []
        target_name_2 = []
        for idx, nsi in enumerate(non_speech_index):
            if nsi == None:
                original_source_list_2.append(original_source_list[idx])
                target_name_2.append(target_name[idx])
                continue

            non_speech, sr_ns = sf.read(self.non_speeches[nsi], dtype='float64', always_2d=True)
            non_speech = non_speech[:, 0]
            if sr_ns != sr_os:
                non_speech = resample_poly(non_speech, up=sr_os, down=sr_ns)

            frames = info['num_samples']['original_source'][idx]
            if len(non_speech) < frames:
                non_speech = np.concatenate([non_speech] * (frames // len(non_speech) + 1))
            start = rng.integers(0, len(non_speech) - frames + 1)
            non_speech = non_speech[start:start + frames]
            non_speech = non_speech * (np.mean(np.abs(original_source_list[idx])) / np.mean(np.abs(non_speech)))
            if np.max(np.abs(non_speech)) > 1:
                non_speech /= np.max(np.abs(non_speech))
            original_source_list_2.append(non_speech)
            target_name_2.append(basename(str(self.non_speeches[nsi])))

        return original_source_list_2, target_name_2


class SmsWsjDataModule(LightningDataModule):

    def __init__(
        self,
        sms_wsj_dir: str,  # a dir contains [early, noise, observation, rirs, speech_source, tail, wsj_8k_zeromean]
        target: str,  # e.g. early, image (i.e. early+tail), speech_source, RTS_0.15s, direct_path_2ms, direct_path
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None],  # audio_time_len (seconds) for training, val, test.
        ref_channel: int = 0,
        batch_size: List[int] = [1, 1],
        num_spk: int = 2,  # 2: SMS-WSJ, 1: speech + non-speech, 0: non-speech + non-speech
        fuss_dir: str = None,  # a dir contains [fsd_data,...]
        num_workers: int = 5,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        test_set: str = 'test',
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.sms_wsj_dir = sms_wsj_dir
        self.target = target
        self.audio_time_len = audio_time_len
        self.test_set = test_set
        self.ref_channel = ref_channel
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size[0]
        self.batch_size_val = batch_size[1]
        self.batch_size_test = 1
        if len(batch_size) > 2:
            self.batch_size_test = batch_size[2]

        self.num_spk = num_spk
        self.fuss_dir = fuss_dir

        rank_zero_info("dataset:" + {2: "SMS-WSJ", 1: "SMS-WSJ + non-speech", 0: "non-speech + non-speech"}[num_spk])
        rank_zero_info(f'batch size: train={self.batch_size}; val={self.batch_size_val}; test={self.batch_size_test}')
        rank_zero_info(f'audio_time_length: train={self.audio_time_len[0]}; val={self.audio_time_len[1]}; test={self.audio_time_len[2]}')
        rank_zero_info(f'target: {self.target}')
        assert audio_time_len[2] == None, "the length for test set should be None"
        # assert self.batch_size_val == 1, "batch size for validation should be 1 as the audios have different length"
        assert self.num_spk == 2 or self.fuss_dir is not None, "the dir of FUSS dataset should be specified when num_spk!=2"

        self.num_workers = num_workers

        self.collate_func_train = collate_func_train
        self.collate_func_val = collate_func_val
        self.collate_func_test = collate_func_test

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def prepare_data(self):
        ...

    def setup(self, stage=None):
        self.train = SmsWsjDataset(
            sms_wsj_dir=self.sms_wsj_dir,
            target=self.target,
            dataset='train_si284',
            audio_time_len=self.audio_time_len[0] if stage != 'test' else None,
            ref_channel=self.ref_channel,
            num_spk=self.num_spk,
            fuss_dir=self.fuss_dir,
        )
        self.val = SmsWsjDataset(
            sms_wsj_dir=self.sms_wsj_dir,
            target=self.target,
            dataset='cv_dev93',
            audio_time_len=self.audio_time_len[1] if stage != 'test' else None,
            ref_channel=self.ref_channel,
            num_spk=self.num_spk,
            fuss_dir=self.fuss_dir,
        )
        self.test = SmsWsjDataset(
            sms_wsj_dir=self.sms_wsj_dir,
            target=self.target,
            dataset='test_eval92',
            audio_time_len=None,
            ref_channel=self.ref_channel,
            num_spk=self.num_spk,
            fuss_dir=self.fuss_dir,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_func_train,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            shuffle=False,
            batch_size=self.batch_size_val,
            collate_fn=self.collate_func_val,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        prefetch_factor = 2

        if self.test_set == 'test':
            dataset = self.test
        elif self.test_set == 'val':
            dataset = self.val
        else:  # train
            dataset = self.train

        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size_test,
            collate_fn=self.collate_func_test,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor,
        )

    def predict_dataloader(self) -> DataLoader:
        prefetch_factor = 2

        if self.test_set == 'test':
            dataset = self.test
        elif self.test_set == 'val':
            dataset = self.val
        else:  # train
            dataset = self.train

        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            collate_fn=default_collate_func,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor,
        )


########## The following code is taken from SMS-WSJ and modified to generate direct-path signal using rir ###############
from hashlib import md5

import numpy as np
from scipy.signal import fftconvolve


def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response. 
    """
    assert level_ratio < 1, level_ratio
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        return np.min([get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h])

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    return rir_start_sample


def _example_id_to_rng(example_id):
    """
    >>> _example_id_to_rng('example_id').get_state()[1][0]
    2915193065
    """
    hash_value = md5(example_id.encode())
    hash_value = int(hash_value.hexdigest(), 16)
    hash_value -= 1  # legacy operation
    hash_value = hash_value % 2**32
    return np.random.RandomState(hash_value)


def extract_piece(x, offset, target_length):
    """
    Args:
        x:
        offset:
            If negative, cut left side.
            If positive: pad left side.
        target_length:

    Returns:

    """

    def pad_axis(array, pad_width, axis=-1):
        array = np.asarray(array)

        npad = np.zeros([array.ndim, 2], dtype=np.int32)
        npad[axis, :] = pad_width
        return np.pad(array, pad_width=npad, mode='constant')

    if offset < 0:
        x = x[..., -offset:]
    else:
        x = pad_axis(x, (offset, 0), axis=-1)

    if x.shape[-1] < target_length:
        x = pad_axis(x, (0, target_length - x.shape[-1]), axis=-1)
    else:
        x = x[..., :target_length]

    return x


def get_white_noise_for_signal(time_signal, *, snr, rng_state: np.random.RandomState = np.random):
    """
        Args:
            time_signal:
            snr: SNR or single speaker SNR.
            rng_state: A random number generator object or np.random
    """
    noise_signal = rng_state.normal(size=time_signal.shape)

    power_time_signal = np.mean(time_signal**2, keepdims=True)
    power_noise_signal = np.mean(noise_signal**2, keepdims=True)
    current_snr = 10 * np.log10(power_time_signal / power_noise_signal)

    factor = 10**(-(snr - current_snr) / 20)

    noise_signal *= factor
    return noise_signal


def synchronize_speech_source(original_source, offset, T):
    return np.array([extract_piece(x_, offset_, T) for x_, offset_ in zip(
        original_source,
        offset,
    )])


def scenario_map_fn(
        example: Dict[str, Any],
        *,
        snr_range: tuple = (20, 30),
        sync_speech_source=False,
        add_speech_image=True,
        add_speech_reverberation_early=False,
        add_speech_reverberation_tail=False,
        add_noise_image=False,
        rir_target=None,
        early_rir_samples: int = int(8000 * 0.05),  # 50 milli seconds
        channel_slice: Union[None, slice, tuple, list] = None,
        details=False,
):
    """
    This will care for convolution with RIR and also generate noise.
    The random noise generator is fixed based on example ID. It will
    therefore generate the same SNR and same noise sequence the next time
    you use this DB.

    Args:
        example: Example dictionary.
        snr_range: required for noise generation
        sync_speech_source: Legacy option. The new convention is, that
            original_source is the unpadded signal, while speech_source is the
            padded signal.
            pad and/or cut the source signal to match the length of the
            observations. Considers the offset.
        add_speech_image:
            The speech_image is always computed, but when it is not needed,
            this option can reduce the memory consumption.
        add_speech_reverberation_early:
            Calculate the speech_reverberation_early signal, i.e., the speech
            source (padded original source) convolved with the early part of
            the RIR.
        add_speech_reverberation_tail:
            Calculate the speech_reverberation_tail signal, i.e., the speech
            source (padded original source) convolved with the tail part of
            the RIR.
        add_noise_image:
            If True, add the noise_image the returned example.
            This option has no effect to the computation time or the peak
            memory consumption.
        early_rir_samples:
            The number of samples that we count as the early RIR, default 50ms.
            The remaining part of the RIR we call tail.
            Note: The length of the early RIR is the time of flight plus this
            value.
        channel_slice: e.g. None (All channels), [4] (Single channel), ...
            Warning: Use this only for training. It will change the scale of
            the data and the added white noise.
            For the scale the standard deviation is estimated and the generated
            noise shape changes, hence also the values.
            With this option you can select the interested channels.
            All RIRs are used to estimate the time of flight, but only the
            interested channels are convolved with the original/speech source.
            This reduces computation time and memory consumption.

    Returns:

    """
    h = example['audio_data']['rir']  # Shape (speaker, channel, sample)

    # Estimate start sample first, to make it independent of channel_mode
    # Calculate one rir_start_sample (i.e. time of flight) for each speaker.
    rir_start_sample = np.array([get_rir_start_sample(h_k) for h_k in h])

    if channel_slice is not None:
        assert h.ndim == 3, h.shape
        h = h[:, channel_slice, :]
        assert h.ndim == 3, h.shape

    _, D, rir_length = h.shape

    # TODO: SAMPLE_RATE not defined
    # rir_stop_sample = rir_start_sample + int(SAMPLE_RATE * 0.05)
    # Use 50 milliseconds as early rir part, excluding the propagation delay
    #    (i.e. "rir_start_sample")
    assert isinstance(early_rir_samples, int), (type(early_rir_samples), early_rir_samples)
    rir_stop_sample = rir_start_sample + early_rir_samples

    log_weights = example['log_weights']

    # The two sources have to be cut to same length
    K = example['num_speakers']
    T = example['num_samples']['observation']
    if 'original_source' not in example['audio_data']:
        # legacy code
        example['audio_data']['original_source'] = example['audio_data']['speech_source']
    if 'original_source' not in example['num_samples']:
        # legacy code
        example['num_samples']['original_source'] = example['num_samples']['speech_source']
    s = example['audio_data']['original_source']

    def get_convolved_signals(h):
        assert len(s) == h.shape[0], (len(s), h.shape)
        x = [fftconvolve(s_[..., None, :], h_, axes=-1) for s_, h_ in zip(s, h)]

        assert len(x) == len(example['num_samples']['original_source']), (len(x), len(example['num_samples']['original_source']))
        for x_, T_ in zip(x, example['num_samples']['original_source']):
            assert x_.shape == (h.shape[1], T_ + rir_length - 1), (x_.shape, h.shape[1], T_ + rir_length - 1)

        # This is Jahn's heuristic to be able to still use WSJ alignments.
        offset = [offset_ - rir_start_sample_ for offset_, rir_start_sample_ in zip(example['offset'], rir_start_sample)]

        assert len(x) == len(offset)
        x = [extract_piece(x_, offset_, T) for x_, offset_ in zip(x, offset)]
        x = np.stack(x, axis=0)
        assert x.shape == (K, h.shape[1], T), x.shape
        return x

    x = get_convolved_signals(h)

    # Note: scale depends on channel mode
    std = np.maximum(
        np.std(x, axis=(-2, -1), keepdims=True),
        np.finfo(x.dtype).tiny,
    )

    # Rescale such that invasive SIR is as close as possible to `log_weights`.
    scale = (10**(np.asarray(log_weights)[:, None, None] / 20)) / std
    # divide by 71 to ensure that all values are between -1 and 1
    scale /= 71

    x *= scale
    if add_speech_image:
        example['audio_data']['speech_image'] = x

    clean_mix = np.sum(x, axis=0)
    if not add_speech_image:
        del x  # Reduce memory consumption for the case of `not add_speech_image`

    if add_speech_reverberation_early:
        h_early = h.copy()
        # Replace this with advanced indexing
        for i in range(h_early.shape[0]):
            h_early[i, ..., rir_stop_sample[i]:] = 0
        x_early = get_convolved_signals(h_early)
        x_early *= scale
        example['audio_data']['speech_reverberation_early'] = x_early

        if details:
            example['audio_data']['rir_early'] = h_early

    if add_speech_reverberation_tail:
        h_tail = h.copy()
        for i in range(h_tail.shape[0]):
            h_tail[i, ..., :rir_stop_sample[i]] = 0
        x_tail = get_convolved_signals(h_tail)
        x_tail *= scale
        example['audio_data']['speech_reverberation_tail'] = x_tail

        if details:
            example['audio_data']['rir_tail'] = h_tail

    if rir_target is not None:
        if rir_target.ndim == 2:
            assert rir_target.shape == (h.shape[0], h.shape[2])
            rir_target = rir_target[:, None, :]
        elif rir_target.ndim == 3:
            assert rir_target.shape == h.shape, (rir_target.shape, h.shape)
        x_target = get_convolved_signals(rir_target)
        x_target *= scale
        example['audio_data']['speech_target'] = x_target

        if details:
            example['audio_data']['rir_target'] = rir_target

    if sync_speech_source:
        example['audio_data']['speech_source'] = synchronize_speech_source(
            example['audio_data']['original_source'],
            offset=example['offset'],
            T=T,
        )
    else:
        # legacy code
        example['audio_data']['speech_source'] = \
            example['audio_data']['original_source']

    rng = _example_id_to_rng(example['example_id'])
    snr = rng.uniform(*snr_range)
    example["snr"] = snr

    rng = _example_id_to_rng(example['example_id'])

    n = get_white_noise_for_signal(clean_mix, snr=snr, rng_state=rng)
    if add_noise_image:
        example['audio_data']['noise_image'] = n

    observation = clean_mix
    observation += n  # Inplace to reduce memory consumption
    example['audio_data']['observation'] = observation

    return example


####### taken from SMS-WSJ  end ###########
def generate_and_save_rir(
    example,
    database_path,
    dataset_name,
    sound_decay_time=None,
    sample_rate=16000,
    filter_length=2**13,
    sensor_orientations=None,
    sensor_directivity=None,
    sound_velocity=343,
):
    import rir_generator

    room_dimensions = np.array(example['room_dimensions'])
    source_positions = np.array(example['source_position'])
    sensor_positions = np.array(example['sensor_position'])
    if sound_decay_time is None:
        sound_decay_time = example['sound_decay_time']

    if np.ndim(source_positions) == 1:
        source_positions = np.reshape(source_positions, (-1, 1))
    if np.ndim(room_dimensions) == 1:
        room_dimensions = np.reshape(room_dimensions, (-1, 1))
    if np.ndim(sensor_positions) == 1:
        sensor_positions = np.reshape(sensor_positions, (-1, 1))

    assert room_dimensions.shape == (3, 1)
    assert source_positions.shape[0] == 3
    assert sensor_positions.shape[0] == 3

    number_of_sources = source_positions.shape[1]
    number_of_sensors = sensor_positions.shape[1]

    if sensor_orientations is None:
        sensor_orientations = np.zeros((2, number_of_sources))
    else:
        raise NotImplementedError(sensor_orientations)

    if sensor_directivity is None:
        sensor_directivity = 'omnidirectional'
    else:
        raise NotImplementedError(sensor_directivity)

    assert filter_length is not None
    rir = np.zeros((number_of_sources, number_of_sensors, filter_length), dtype=np.float64)
    for k in range(number_of_sources):
        temp = rir_generator.generate(
            c=sound_velocity,
            fs=sample_rate,
            r=np.ascontiguousarray(sensor_positions.T),
            s=np.ascontiguousarray(source_positions[:, k]),
            L=np.ascontiguousarray(room_dimensions[:, 0]),
            reverberation_time=sound_decay_time,
            nsample=filter_length,
            mtype=rir_generator.mtype.omnidirectional,
        )
        rir[k, :, :] = np.asarray(temp.T)

    assert rir.shape[0] == number_of_sources
    assert rir.shape[1] == number_of_sensors
    assert rir.shape[2] == filter_length

    assert not np.any(np.isnan(rir)), f"{np.sum(np.isnan(rir))} values of {rir.size} are NaN."

    K, D, T = rir.shape
    directory = database_path / dataset_name / example['example_id']
    directory.mkdir(parents=True, exist_ok=True)

    for k in range(K):
        # Although storing as np.float64 does not allow every reader
        # to access the files, it does not require normalization and
        # we are unsure how much precision is needed for RIRs.
        with sf.SoundFile(str(directory / f"h_{k}.wav"), subtype='DOUBLE', samplerate=sample_rate, mode='w', channels=rir.shape[1]) as f:
            f.write(rir[k, :, :].T)

    return rir


def generate_direct_path_rir(sms_wsj_dir: str, num_workers: int = 8):
    print('gen_direct_path_rir')
    import rir_generator
    from p_tqdm import p_map
    from functools import partial

    sms_wsj_path = Path(sms_wsj_dir).expanduser()
    with open(sms_wsj_path / 'rirs/scenarios.json', 'r') as f:
        d = json.load(f)
        dataset_info = d['datasets']

    database_path = sms_wsj_path / 'rirs_direct_path'
    if database_path.exists():
        print(database_path, 'exists, so not generate dp rirs')

    for dataset_name, dataset in dataset_info.items():
        print(dataset_name)
        dataset = list(dataset.values())
        p_map(
            partial(
                generate_and_save_rir,
                database_path=database_path,
                dataset_name=dataset_name,
                sample_rate=8000,
                filter_length=2**13,  # 1.024 seconds when sample_rate == 8000
                sound_decay_time=0,  # for direct path
            ),
            dataset,
            num_cpus=num_workers,
        )


if __name__ == '__main__':
    "python -m data_loaders.sms_wsj"
    from argparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_argument('--sms_wsj_dir', type=str, default='~/datasets/sms_wsj')
    parser.add_argument('--target', type=str, default='direct_path')
    parser.add_argument('--gen_direct_path_rir', type=bool, default=False)
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)
    parser.add_argument('--num_spk', type=int, default=1)
    parser.add_argument('--fuss_dir', type=str, default='~/datasets/FUSS')
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()
    if args.gen_direct_path_rir:
        generate_direct_path_rir(sms_wsj_dir=args.sms_wsj_dir, num_workers=args.num_workers)

    sms_wsj_ds = SmsWsjDataset(args.sms_wsj_dir, args.target, 'train_si284', audio_time_len=4, num_spk=args.num_spk, fuss_dir=args.fuss_dir)
    sms_wsj_ds[0]

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    sms_wsj_dm = SmsWsjDataModule(args.sms_wsj_dir, args.target, num_spk=args.num_spk, fuss_dir=args.fuss_dir)
    sms_wsj_dm.setup()
    dataloader = sms_wsj_dm.test_dataloader()

    for idx, (mix, tar, paras) in enumerate(dataloader):
        assert paras[0]['target'] not in ['rirs', 'noise', 'observation', 'speech_source', 'early', 'tail', 'wsj_8k_zeromean'], 'You should not overwrite the original data ' + str(paras[0]['target'])

        print(mix.shape, tar.shape, paras)
        if idx > 10:
            continue
        # write target to dir
        if args.gen_target:
            tar_path = Path(f"dataset/SMS-WSJ/{args.num_spk}spk/{args.target}/test_eval92").expanduser()
            tar_path.mkdir(exist_ok=True, parents=True)
            for i in [0, 1]:
                assert np.max(np.abs(tar[0, i, 0, :].numpy())) <= 1
                if not (tar_path / paras[0]['target_name'][i]).exists():
                    sf.write(tar_path / paras[0]['target_name'][i], tar[0, i, 0, :].numpy(), samplerate=8000)

        # write unprocessed's 0-th channel
        if args.gen_unprocessed:
            unp_path = Path(f"dataset/SMS-WSJ/{args.num_spk}spk/unprocessed/test_eval92").expanduser()
            unp_path.mkdir(exist_ok=True, parents=True)
            for i in [0, 1]:
                assert np.max(np.abs(mix[0, 0, :].numpy())) <= 1
                if not (unp_path / paras[0]['target_name'][i]).exists():  # write to disk with the same name as targets for the computing of unprocessed metrics
                    sf.write(unp_path / paras[0]['target_name'][i], mix[0, 0, :].numpy(), samplerate=8000)
