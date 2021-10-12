from typing import Callable, Dict, List, Optional, Tuple, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import os

from .ss_semi_online_dataset import SS_SemiOnlineDataset
from .ss_semi_online_sampler import SS_SemiOnlineSampler
import pandas as pd
import json
import random


class SS_SemiOnlineDataModule(LightningDataModule):
    """半在线方式的DataModule实现，DataModule里面提供了如何准备数据、构造Dataset、构造DataLoader的方法
    """

    @staticmethod
    def add_argparse_args(parent_parser):

        def float_or_str(value):
            try:
                return float(value)
            except:
                return value

        parser = parent_parser.add_argument_group("SS_SemiOnlineDataModule")
        parser.add_argument('--clean_speech_dataset', type=str, required=True, choices=['wsj0-mix'])
        parser.add_argument('--clean_speech_dir', type=str, required=True)
        parser.add_argument('--rir_dir', type=str, required=True)
        parser.add_argument('--speech_overlap_ratio', type=float, nargs='+', required=True)
        parser.add_argument('--speech_scale', type=float, nargs='+', required=True)
        parser.add_argument(
            '--audio_time_len',
            type=float_or_str,
            nargs='+',
            required=True,
            help="can be float, max, min or None. given one parameter means for train only, others are None; given two means for train and val; given three means for train val and test")
        parser.add_argument('--num_workers', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=8, nargs='+', help='batch_size for train and validation')

        return parent_parser

    rirs: Dict[str, List[str]]
    spk1_cfgs: Dict[str, List[Dict[str, str]]]
    overlap_det: Optional[Dict[str, List[float]]]
    scale_det: Dict[str, List[float]]

    def __init__(
        self,
        clean_speech_dataset: str,
        clean_speech_dir: str,
        rir_dir: str,
        speech_overlap_ratio: List[float] = [0.1, 1.],
        speech_scale: List[float] = [-5., 5.],
        batch_size: List[int] = [5, 5],
        speaker_num: int = 2,
        audio_time_len: List[Union[float, str]] = ["headtail 4", "headtail 4", "headtail 4"],
        num_workers: int = 5,
        collate_func_train: Callable = None,
        collate_func_val: Callable = None,
        collate_func_test: Callable = None,
        test_set: str = 'test',
        shuffle_train_rir: bool = True,
        seeds: Dict[str, Optional[int]] = {
            'train': None,
            'val': 2,  # fix val and test seeds to make sure they won't change in any time
            'test': 3
        },
    ):

        super().__init__()

        self.test_set = test_set

        self.shuffle_train_rir = shuffle_train_rir
        self.seeds: Dict[str, int] = dict()
        for k, v in seeds.items():
            if v is None:
                v = random.randint(0, 1000000)
            self.seeds[k] = v
        print('seeds for datasets:', self.seeds)
        # generate seeds for train, validation and test
        # self.seed = seed
        # self.g = torch.Generator()
        # self.g.manual_seed(seed)
        # self.seeds = {'train': randint(self.g, 0, 100000), 'val': randint(self.g, 0, 100000), 'test': randint(self.g, 0, 100000)}

        self.clean_speech_dataset = clean_speech_dataset
        self.clean_speech_dir = clean_speech_dir
        self.rir_dir = rir_dir
        assert len(speech_overlap_ratio) == 2, "should give a range"
        self.speech_overlap_ratio = (speech_overlap_ratio[0], speech_overlap_ratio[1])
        print(f'speech overlap: train={self.speech_overlap_ratio}; val={self.speech_overlap_ratio}; test={self.speech_overlap_ratio}')

        assert len(speech_scale) == 2, "should give a range"
        self.speech_scale = (speech_scale[0], speech_scale[1])
        print(f'speech scale: train={self.speech_scale}; val={self.speech_scale}; test={self.speech_scale}')

        self.batch_size = batch_size[0]
        self.batch_size_val = batch_size[1]
        print(f'batch size: train={self.batch_size}; val={self.batch_size_val}; test=1')

        self.speaker_num = speaker_num
        self.num_workers = num_workers

        self.audio_time_len = audio_time_len[0]
        self.audio_time_len_for_val = None if len(audio_time_len) < 2 else audio_time_len[1]
        self.audio_time_len_for_test = None if len(audio_time_len) < 3 else audio_time_len[2]
        print(f'audio_time_len: train={self.audio_time_len}; val={self.audio_time_len_for_val}; test={self.audio_time_len_for_test}')

        self.collate_func_train = collate_func_train
        self.collate_func_val = collate_func_val
        self.collate_func_test = collate_func_test

        self.prepare_data()

    def prepare_data(self):
        """根据clean_speech_dataset、clean_speech_dir、rir_dir，构造SS_SemiOnlineDataset需要的参数

        configs文件夹:
            文件夹内应当有名字为clean_speech_dataset值的文件夹，里面存放着对应的config文件，里面的内容是自定义的或别人给定的，如果是别人给定的，那么应该是别人的原始文件
        
        clean_speech_dir文件夹：
            应该保持干净语音的原始文件夹布局

        rir_dir文件夹：
            文件夹内应该有三个子文件夹：train、validation、test三个文件夹，里面存的全是rir文件
        """

        # self.rirs: Dict[str, List[str]]
        self.rirs = {}
        for sub_dir in ['train', 'validation', 'test']:
            files = os.listdir(os.path.join(self.rir_dir, sub_dir))
            files_full_path = []
            for f in files:
                path = os.path.join(self.rir_dir, sub_dir, f)
                files_full_path.append(path)
            self.rirs[sub_dir] = files_full_path

        # wsj0-mix
        if self.clean_speech_dataset == 'wsj0-mix':
            spk1_cfgs, spk2_cfgs = SS_SemiOnlineDataModule.prepare_data_wsj0_mix(f'configs/{self.clean_speech_dataset}')
        # 相对路径转绝对路径
        for ds in ['train', 'validation', 'test']:
            ds_cfg = spk1_cfgs[ds]
            for cfg in ds_cfg:
                cfg['wav'] = os.path.join(self.clean_speech_dir, cfg['wav'])
            ds_cfg = spk2_cfgs[ds]
            for cfg in ds_cfg:
                cfg['wav'] = os.path.join(self.clean_speech_dir, cfg['wav'])
        self.spk1_cfgs = spk1_cfgs
        self.spk2_cfgs = spk2_cfgs

    @staticmethod
    def prepare_data_wsj0_mix(config_dir: str,) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
        """translate wsj0-mix dataset to the data structure supported by SS_SemiOnlineDataModule

        Args:
            config_dir: the dir contains wsj0-mix dataset configuration files
            speech_overlap_ratio: generate deterministic parameters for train, validation, test
            speech_scale: generate deterministic parameters for train, validation, test

        Example:
            >>> from data_loaders.ss_semi_online_data_module import SS_SemiOnlineDataModule
            >>> spk1_cfg, spk2_cfg, sord, ssd = SS_SemiOnlineDataModule.prepare_data_wsj0_mix(config_dir='configs/wsj0-mix', speech_overlap_ratio=(0, 1), speech_scale=(-5, 5))
            >>> spk1_cfg['train']
            >>> spk1_cfg['validation']
            >>> spk1_cfg['test']

        Returns:
            Dict[str, List[Dict[str, str]]]: First speaker dataset. Eg. Dict['train'] is a list of Dict, which contains the information of speaker 1, like 'wav', 'speaker', 'gender', ...
            Dict[str, List[Dict[str, str]]]: Second speaker dataset. Eg. Dict['train'] is a list of Dict, which contains the information of speaker 1, like 'wav', 'speaker', 'gender', ...
        """
        wav_cfg_path = f'{config_dir}/wsj0_mix-SS_SODM-wav.json'

        # read wav cfgs
        if os.path.exists(wav_cfg_path):
            print(f'read wav configuration from {wav_cfg_path}')
            f = open(wav_cfg_path, 'r', encoding='utf-8')
            wav_cfg = json.load(f)
            f.close()
            spk1_cfgs = wav_cfg['spk1_cfgs']
            spk2_cfgs = wav_cfg['spk2_cfgs']
        else:
            # cal wav cfgs
            spk1_cfgs = {}
            spk2_cfgs = {}

            genders = pd.read_csv(f'{config_dir}/speaker_gender.csv', delim_whitespace=True)
            genders.sort_values(by='ID', inplace=True)

            for key, dataset in {'train': 'tr', 'validation': 'cv', 'test': 'tt'}.items():
                df = pd.read_csv(f'{config_dir}/mix_2_spk_' + dataset + '.txt', delim_whitespace=True, names=['wav_file1', 'scale1', 'wav_file2', 'scale2'])
                num_files = df.shape[0]

                spk1_cfg = []
                spk2_cfg = []
                for i in range(num_files):
                    dfi = df.iloc[i]

                    inwav1_name, inwav2_name = dfi[0], dfi[2]  #'wsj0/si_tr_s/40n/40na010x.wav'
                    inwav1_snr, inwav2_snr = dfi[1], dfi[3]
                    spk_name1 = inwav1_name.split('/')[3][0:3]
                    spk_name2 = inwav2_name.split('/')[3][0:3]

                    gender_i1 = genders.ID.values.searchsorted(spk_name1, side='left')
                    gender_i2 = genders.ID.values.searchsorted(spk_name2, side='left')

                    gender1 = genders.iloc[gender_i1, 1]
                    gender2 = genders.iloc[gender_i2, 1]

                    assert spk_name1 == genders.iloc[gender_i1, 0], 'error1'
                    assert spk_name2 == genders.iloc[gender_i2, 0], 'error1'

                    spk1_cfg.append({'wav': inwav1_name, 'speaker': spk_name1, 'gender': gender1, 'dataset': f"wsj0-mix/{key}"})
                    spk2_cfg.append({'wav': inwav2_name, 'speaker': spk_name2, 'gender': gender2, 'dataset': f"wsj0-mix/{key}"})
                spk1_cfgs[key] = spk1_cfg
                spk2_cfgs[key] = spk2_cfg
            print(f'write wav configuration to {wav_cfg_path}')
            f = open(wav_cfg_path, 'w', encoding='utf-8')
            json.dump({'spk1_cfgs': spk1_cfgs, 'spk2_cfgs': spk2_cfgs}, f, indent=4)
            f.close()

        return spk1_cfgs, spk2_cfgs

    def setup(self, stage=None):
        if stage is not None and stage == 'test':  # 用于训练和测试的dataset的行为是有区别的：训练时的语音，为了构造成一个batch，需要将语音截短、补0成固定长度
            # 此处按照测试的行为来生成对应的数据集：即确定性的数据集，长度为
            self.train = SS_SemiOnlineDataset(
                speeches=[self.spk1_cfgs['train'], self.spk2_cfgs['train']],
                rirs=self.rirs['train'],
                speech_overlap_ratio=self.speech_overlap_ratio,
                speech_scale=self.speech_scale,
                audio_time_len=self.audio_time_len_for_test,
            )
            self.val = SS_SemiOnlineDataset(
                speeches=[self.spk1_cfgs['validation'], self.spk2_cfgs['validation']],
                rirs=self.rirs['validation'],
                speech_overlap_ratio=self.speech_overlap_ratio,
                speech_scale=self.speech_scale,
                audio_time_len=self.audio_time_len_for_test,
            )
            self.test = SS_SemiOnlineDataset(
                speeches=[self.spk1_cfgs['test'], self.spk2_cfgs['test']],
                rirs=self.rirs['test'],
                speech_overlap_ratio=self.speech_overlap_ratio,
                speech_scale=self.speech_scale,
                audio_time_len=self.audio_time_len_for_test,
            )
        else:  # fit
            self.train = SS_SemiOnlineDataset(
                speeches=[self.spk1_cfgs['train'], self.spk2_cfgs['train']],
                rirs=self.rirs['train'],
                speech_overlap_ratio=self.speech_overlap_ratio,
                speech_scale=self.speech_scale,
                audio_time_len=self.audio_time_len,
            )
            self.val = SS_SemiOnlineDataset(
                speeches=[self.spk1_cfgs['validation'], self.spk2_cfgs['validation']],
                rirs=self.rirs['validation'],
                speech_overlap_ratio=self.speech_overlap_ratio,
                speech_scale=self.speech_scale,
                audio_time_len=self.audio_time_len_for_val,
            )
            self.test = SS_SemiOnlineDataset(
                speeches=[self.spk1_cfgs['test'], self.spk2_cfgs['test']],
                rirs=self.rirs['test'],
                speech_overlap_ratio=self.speech_overlap_ratio,
                speech_scale=self.speech_scale,
                audio_time_len=self.audio_time_len_for_test,
            )

    def train_dataloader(self) -> DataLoader:
        prefetch_factor = self.batch_size
        persistent_workers = False
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_func_train,
                          sampler=SS_SemiOnlineSampler(self.train, seed=self.seeds['train'], shuffle=True, shuffle_rir=self.shuffle_train_rir),
                          num_workers=self.num_workers,
                          prefetch_factor=prefetch_factor,
                          pin_memory=True,
                          persistent_workers=persistent_workers)

    def val_dataloader(self) -> DataLoader:
        prefetch_factor = self.batch_size_val
        persistent_workers = False
        return DataLoader(self.val,
                          batch_size=self.batch_size_val,
                          collate_fn=self.collate_func_val,
                          sampler=SS_SemiOnlineSampler(self.val, seed=self.seeds['val'], shuffle=False, shuffle_rir=False),
                          num_workers=self.num_workers,
                          prefetch_factor=prefetch_factor,
                          pin_memory=True,
                          persistent_workers=persistent_workers)

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
            batch_size=1,
            collate_fn=self.collate_func_test,
            sampler=SS_SemiOnlineSampler(dataset, seed=self.seeds['test'], shuffle=False, shuffle_rir=True),
            num_workers=1,
            prefetch_factor=prefetch_factor,
        )
