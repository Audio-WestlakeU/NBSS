from torch import Tensor
from torch.utils.data import Dataset
import torch
import numpy as np
import soundfile as sf
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.signal import convolve, resample


def randint(g: torch.Generator, low: int, high: int) -> int:
    """return [low, high)
    """
    r = torch.randint(low=low, high=high, size=(1,), generator=g, device='cpu')
    return r[0].item()  # type:ignore


def randfloat(g: torch.Generator, low: float, high: float) -> float:
    """return [low, high)
    """
    r = torch.rand(size=(1,), generator=g, device='cpu')[0].item()
    return float(low + r * (high - low))


class SS_SemiOnlineDataset(Dataset):
    """语音分离半在线数据集：训练的语音是动态合成的，RIR是离线生成的
    """

    @staticmethod
    def collate_fn(batches):
        mini_batch = []
        for x in zip(*batches):
            if isinstance(x[0], np.ndarray):
                x = [torch.tensor(x[i]) for i in range(len(x))]
            if isinstance(x[0], Tensor):
                x = torch.stack(x)
            mini_batch.append(x)
        return mini_batch

    def __init__(self,
                 speeches: List[List[Dict[str, str]]],
                 rirs: List[str],
                 speech_overlap_ratio: Tuple[float, float],
                 speech_scale: Tuple[float, float],
                 audio_time_len: Optional[Union[float, str]] = None,
                 sample_rate: Optional[int] = None) -> None:
        """初始化

        Args:
            speeches: 单通道原始语音信号的路径。对于分离任务，类型为List[List[Dict[str,str]]]；增强和去混响为List[Dict[str,str]]?
            rirs: rir的路径，指向NPZ格式的文件。每个NPZ内部是一个Dict，其中名字为speech_rir的部分是speech的rir，名字为noise_rir的部分是noise的rir
            speech_scale: 范围，用于重新调整语音的能量，单位为dB。分离任务为语音间相互的能量。增强和降噪？？？
            audio_time_len: 返回语音的长度，单位秒。指定该参数则会对信号做补0和截取的操作
            speech_overlap_ratio: 分离任务的参数，指定后会调整信号的语音信号的重叠比例
            sample_rate: 采样率。如果指定，则会对语音和rir进行降采样或者重采样
        """
        self.speaker_num = len(speeches)
        assert self.speaker_num == 2, f"Only support two speaker cases, now it's {self.speaker_num} speakers"

        self.speeches = speeches
        self.rirs = rirs
        self.speech_scale = speech_scale
        self.audio_time_len = audio_time_len
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sample_rate = sample_rate

        # construct a mapping: (index, spk) -> the index of next uttr of the same spk
        self.uttr_next = [[-1 for utt in range(len(self.speeches[spk]))] for spk in range(self.speaker_num)]

    def __getitem__(self, index: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:  # type: ignore
        """returns the indexed item

        Args:
            index: index

        Returns:
            Tensor: xm of shape [channel, time]
            Tensor: ys of shape [spk, channel, time]
            dict: paras used
        """
        sidx = index['speech_index']
        ridx = index['rir_index']
        g = torch.Generator()
        g.manual_seed(index['seed'])

        # step 1: load clean speeches, single channel
        cleans: List[np.ndarray] = []
        for speech in self.speeches:
            clean_i, samplerate_i = self.read(speech[sidx]['wav'])
            cleans.append(clean_i)
        if self.sample_rate == None:
            self.sample_rate = samplerate_i

        # step 2: load rirs
        rir_all = np.load(self.rirs[ridx])
        rir, rir_samplerate = rir_all['speech_rir'], rir_all['sr']  # rir shape: [speaker, channel, time]
        # resample if necessary
        if self.sample_rate != None and rir_samplerate != self.sample_rate:
            re_len = int(rir.shape[2] * self.sample_rate / rir_samplerate)
            rir = resample(rir, re_len, axis=2)

        # step 3: compute the needed length of the two signals
        ovlp_type = self.audio_time_len
        if self.audio_time_len == None or self.audio_time_len == "max":
            ovlp_type = "mid"
            # for None and max, no pad and cut, overlap_ratio=min/max
            needed_lens = [clean.shape[0] for clean in cleans]
            speech_overlap_ratio_for_this = np.min(needed_lens) / np.max(needed_lens)
            mix_frame_len = max(needed_lens)
        elif str.startswith(str(self.audio_time_len), "mix"):  # eg: mix 5
            # sample a type
            mix_type_num = 3 if str.startswith(str(self.audio_time_len), "mix3") else 2
            types = ['mid', 'headtail', 'full']
            which_type = randint(g, low=0, high=mix_type_num)
            ovlp_type = types[which_type]
            # cal needed_lens
            speech_overlap_ratio_for_this = randfloat(g, low=self.speech_overlap_ratio[0], high=self.speech_overlap_ratio[1])
            mix_frame_len = int(float(str(self.audio_time_len).split(" ")[1].strip()) * self.sample_rate)  # type:ignore
            if ovlp_type == "mid":  # part-full 5
                # for part-full 'n', pad and cut, overlap_ratio=sampled
                needed_lens = [clean.shape[0] for clean in cleans]
                max_idx = needed_lens.index(max(needed_lens))
                min_idx = needed_lens.index(min(needed_lens))
                if max_idx == min_idx:
                    max_idx = [1, 0][max_idx]
                needed_lens[max_idx] = mix_frame_len
                needed_lens[min_idx] = int(mix_frame_len * speech_overlap_ratio_for_this)  # type:ignore
            elif ovlp_type == "headtail":  # 5
                needed_lens = [int(mix_frame_len * (0.5 + speech_overlap_ratio_for_this / 2))] * self.speaker_num
            else:  # full:
                speech_overlap_ratio_for_this = 1.0
                needed_lens = [mix_frame_len] * self.speaker_num
        elif str.startswith(str(self.audio_time_len), "nmix"):  # eg: nmix 5
            # sample a type
            types = ['mid', 'headtail', ['front', 'end']]  # type:ignore
            which_type = randint(g, low=0, high=len(types))
            if isinstance(types[which_type], list):
                types = types[which_type]  # type:ignore
                which_type = randint(g, low=0, high=len(types))
            ovlp_type = types[which_type]
            # cal needed_lens
            speech_overlap_ratio_for_this = randfloat(g, low=self.speech_overlap_ratio[0], high=self.speech_overlap_ratio[1])
            mix_frame_len = int(float(str(self.audio_time_len).split(" ")[1].strip()) * self.sample_rate)  # type:ignore
            if ovlp_type == "headtail":  # 5
                needed_lens = [int(mix_frame_len * (0.5 + speech_overlap_ratio_for_this / 2))] * self.speaker_num
            else:  # mid or front or end
                # pad and cut, overlap_ratio=sampled
                needed_lens = [clean.shape[0] for clean in cleans]
                max_idx = needed_lens.index(max(needed_lens))
                min_idx = needed_lens.index(min(needed_lens))
                if max_idx == min_idx:
                    max_idx = [1, 0][max_idx]
                needed_lens[max_idx] = mix_frame_len
                needed_lens[min_idx] = int(mix_frame_len * speech_overlap_ratio_for_this)  # type:ignore
        elif str.startswith(str(self.audio_time_len), "all-mix"):  # eg: all-mix 5
            # sample a type
            types = ['full', 'mid', 'headtail', ['front', 'end']]  # type:ignore
            which_type = randint(g, low=0, high=len(types))
            if isinstance(types[which_type], list):
                types = types[which_type]  # type:ignore
                which_type = randint(g, low=0, high=len(types))
            ovlp_type = types[which_type]
            # cal needed_lens
            speech_overlap_ratio_for_this = randfloat(g, low=self.speech_overlap_ratio[0], high=self.speech_overlap_ratio[1])
            if ovlp_type == 'full':
                speech_overlap_ratio_for_this = 1.0
            mix_frame_len = int(float(str(self.audio_time_len).split(" ")[1].strip()) * self.sample_rate)  # type:ignore
            if ovlp_type == "headtail":  # 5
                needed_lens = [int(mix_frame_len * (0.5 + speech_overlap_ratio_for_this / 2))] * self.speaker_num
            else:  # full, mid or front or end
                # pad and cut, overlap_ratio=sampled
                needed_lens = [clean.shape[0] for clean in cleans]
                max_idx = needed_lens.index(max(needed_lens))
                min_idx = needed_lens.index(min(needed_lens))
                if max_idx == min_idx:
                    max_idx = [1, 0][max_idx]
                needed_lens[max_idx] = mix_frame_len
                needed_lens[min_idx] = int(mix_frame_len * speech_overlap_ratio_for_this)  # type:ignore
        elif str.startswith(str(self.audio_time_len), "frontend"):  # eg: frontend 5
            # sample a type
            types = ['front', 'end']
            which_type = randint(g, low=0, high=len(types))
            ovlp_type = types[which_type]
            # for frontend 'n', pad and cut, overlap_ratio=sampled
            speech_overlap_ratio_for_this = randfloat(g, low=self.speech_overlap_ratio[0], high=self.speech_overlap_ratio[1])
            mix_frame_len = int(float(str(self.audio_time_len).split(" ")[1].strip()) * self.sample_rate)  # type:ignore

            needed_lens = [clean.shape[0] for clean in cleans]
            max_idx = needed_lens.index(max(needed_lens))
            min_idx = needed_lens.index(min(needed_lens))
            if max_idx == min_idx:
                max_idx = [1, 0][max_idx]
            needed_lens[max_idx] = mix_frame_len
            needed_lens[min_idx] = int(mix_frame_len * speech_overlap_ratio_for_this)  # type:ignore
        elif str.startswith(str(self.audio_time_len), "mid"):  # eg: mid 5
            ovlp_type = "mid"
            # for mid 'n', pad and cut, overlap_ratio=sampled
            speech_overlap_ratio_for_this = randfloat(g, low=self.speech_overlap_ratio[0], high=self.speech_overlap_ratio[1])
            mix_frame_len = int(float(str(self.audio_time_len).split(" ")[1].strip()) * self.sample_rate)  # type:ignore

            needed_lens = [clean.shape[0] for clean in cleans]
            max_idx = needed_lens.index(max(needed_lens))
            min_idx = needed_lens.index(min(needed_lens))
            if max_idx == min_idx:
                max_idx = [1, 0][max_idx]
            needed_lens[max_idx] = mix_frame_len
            needed_lens[min_idx] = int(mix_frame_len * speech_overlap_ratio_for_this)  # type:ignore
        elif self.audio_time_len == "min":
            ovlp_type = "full"
            # for min, cut the longer audio to the shorter length, no pad, overlap_ratio=1
            lens = [clean.shape[0] for clean in cleans]
            needed_lens = [min(lens)] * self.speaker_num
            speech_overlap_ratio_for_this = 1.0
            mix_frame_len = min(lens)
        else:  # headtail 5 or 5
            ovlp_type = "headtail"
            speech_overlap_ratio_for_this = randfloat(g, low=self.speech_overlap_ratio[0], high=self.speech_overlap_ratio[1])
            # for float, pad & cut if needed
            if str.startswith(str(self.audio_time_len), "headtail"):
                mix_frame_len = int(float(str(self.audio_time_len).split(" ")[1].strip()) * self.sample_rate)  # type:ignore
            else:
                mix_frame_len = int(self.audio_time_len * self.sample_rate)  # type: ignore
            needed_lens = [int(mix_frame_len * (0.5 + speech_overlap_ratio_for_this / 2))] * self.speaker_num

        # step 4: pad signals from the same speaker if they are not long to needed, then cut them to needed
        for i, clean in enumerate(cleans):
            # search pad from index+1
            speaker_this = self.speeches[i][sidx]['speaker']
            idx = sidx
            while len(clean) < needed_lens[i]:
                if self.uttr_next[i][idx] < 0:
                    idx_old = idx
                    # search for the next speech from the same speaker
                    idx = (idx + 1) % len(self.speeches[i])
                    while self.speeches[i][idx]['speaker'] != speaker_this:
                        idx = (idx + 1) % len(self.speeches[i])
                    self.uttr_next[i][idx_old] = idx
                else:  # read cached search result
                    idx = self.uttr_next[i][idx]
                # read and pad
                clean_for_pad, _ = self.read(self.speeches[i][idx]['wav'])
                clean = np.concatenate([clean, clean_for_pad])
            # cut to needed_lens[i]
            if len(clean) > needed_lens[i]:
                start = randint(g, low=0, high=len(clean) - needed_lens[i])
                clean = clean[start:start + needed_lens[i]]
            cleans[i] = clean

        # step 5: rescale cleans
        scale_ratio_dB = None
        if self.speech_scale != None:
            scale_ratio_dB = randfloat(g, self.speech_scale[0], self.speech_scale[1])  # randomly sample in the given range
            # because cleans[0] and cleans[1] may have two different length,
            # thus here first scale them to a state where the average power per second equals one
            cleans[0] = cleans[0] / np.sqrt(np.sum(cleans[0]**2) + 1e-8) * (len(cleans[0]) / self.sample_rate)  # type: ignore
            cleans[1] = cleans[1] / np.sqrt(np.sum(cleans[1]**2) + 1e-8) * (len(cleans[1]) / self.sample_rate)  # type: ignore
            # scale cleans[1] to -5~5dB for example
            cleans[1] = cleans[1] * np.power(10, scale_ratio_dB / 20.)

        # step 6: convolve rir and clean speech, then mix
        chn_num = rir.shape[1]
        echoics = np.zeros((self.speaker_num, chn_num, mix_frame_len))
        for i, y in enumerate(cleans):
            start = None  # type: ignore
            for ch in range(chn_num):
                if len(y) == 0:  # ignore y if needed_lens[i] == 0, i.e. len(y) == 0
                    continue
                echoic_i = convolve(y, rir[i, ch, :])
                other = 1 - i  # i=1, other=0; i=0, other=1
                if needed_lens[other] == mix_frame_len:  # if other speech is full of the length
                    if start == None:  # use the same start for all the channels of one speaker
                        start = randint(g, low=0, high=mix_frame_len - needed_lens[i] + 1)  # [0, mix_frame_len - needed_lens[i]]
                        if ovlp_type == "front":
                            start = 0
                        elif ovlp_type == "end":
                            start = mix_frame_len - needed_lens[i]
                    echoics[i, ch, start:start + needed_lens[i]] = echoic_i[:needed_lens[i]]  # type: ignore
                elif i == 0:  # speaker 1
                    echoics[i, ch, :needed_lens[i]] = echoic_i[:needed_lens[i]]
                else:  # speaker 2
                    echoics[i, ch, -needed_lens[i]:] = echoic_i[:needed_lens[i]]
        mix = np.sum(echoics, axis=0)

        # put the parameters you want to pass to network in paras
        rir_info = {}
        for k, v in dict(rir_all).items():
            if k != 'speech_rir' and k != 'noise_rir':
                rir_info[k] = v

        paras = {
            "index": sidx,
            "spk1": self.speeches[0][sidx],
            "spk2": self.speeches[1][sidx],
            'seed': index['seed'],
            'rir_file': self.rirs[ridx],
            'rir': rir_info,
            'audio_time_len': self.audio_time_len if self.audio_time_len != None else -1,  # -1 means not specified
            'mix_frame_len': mix_frame_len,
            'echoic_frame_len': needed_lens,
            'sample_rate': self.sample_rate,
            'speech_overlap_ratio': speech_overlap_ratio_for_this,
            'ovlp_type': ovlp_type,
        }

        # norm amplitude
        max_amp = max(np.max(np.abs(mix)), np.max(np.abs(echoics)))
        amp_scaling = 0.9 / max_amp
        mix *= amp_scaling
        echoics *= amp_scaling

        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(echoics, dtype=torch.float32), paras

    def __len__(self):
        return self.speech_num()

    def speech_num(self):
        return len(self.speeches[0])

    def rir_num(self):
        return len(self.rirs)

    def read(self, wav_path):
        clean, samplerate = sf.read(wav_path, dtype='float32')
        assert len(clean.shape) == 1, "clean speech should be single channel"
        # resample if necessary
        if self.sample_rate != None and samplerate != self.sample_rate:
            re_len = int(clean.shape[0] * self.sample_rate / samplerate)
            clean = resample(clean, re_len)
        return clean, samplerate