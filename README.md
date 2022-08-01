# Multi-channel Narrow-band Deep Speech Separation

A narrow-band speech separation method.
The official repo of "[Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training](https://arxiv.org/abs/2110.05966)" accepted by ICASSP 2022 and "[Multichannel Speech Separation with Narrow-band Conformer](https://arxiv.org/abs/2204.04464)" accepted by InterSpeech 2022.
Detailed introduction (with images and examples) about this work can be found [in English](https://audio.westlake.edu.cn/Research/nbss.htm) or [in Chinese](https://quancs.github.io/zh-cn/blog/nbss/).
More information about our group can be found at [https://audio.westlake.edu.cn](https://audio.westlake.edu.cn/Publications.htm).


## Results

Speech Separation Performance Comparision with SOTA Speech Separation Methods for 8-Channel 2-Speaker Mixtures (reported in [5])

Model			| #param	| NB-PESQ 	| WB-PESQ 	| SI-SDR	| RTF
------			|------:	|------:	|------:	|------:	|------:
Mixture 		| - 		| 2.05 		| 1.59 		| 0.0		| -
Oracle MVDR [1] | - 		| 3.16	 	| 2.65 		| 11.0		| -
FaSNet-TAC [2] 	| 2.8 M 	| 2.96 		| 2.53 		| 12.6		| 0.67
SepFormer [3]	| 25.7 M	| 3.17		| 2.72		| 13.2		| 1.69
SepFormerMC		| 25.7 M	| 3.42		| 3.01		| 14.9		| 1.70
NB-BLSTM [4] 	| 1.2 M		| 3.28 		| 2.81	 	| 12.8		| 0.37
NBC [5]			| 2.0 M		| **4.00**	| **3.78**	| **20.3**	| 1.32

[1] https://github.com/Enny1991/beamformers

[2] Yi Luo, Zhuo Chen, Nima Mesgarani, and Takuya Yoshioka. End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation. In ICASSP 2020.

[3] C. Subakan, M. Ravanelli, S. Cornell, M. Bronzi, and J. Zhong. Attention Is All You Need In Speech Separation. In ICASSP 2021.

[4] Changsheng Quan, Xiaofei Li. **Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training**. In ICASSP 2022.

[5] Changsheng Quan, Xiaofei Li. **Multichannel Speech Separation with Narrow-band Conformer**. arXiv preprint arXiv:2204.04464.


## Requirements
```
pip install -r requirements.txt

# gpuRIR: check https://github.com/DavidDiazGuerra/gpuRIR
```

## Generate rirs
Generate rirs using `configs/rir_cfg_4.json`, and the generated rirs are placed in `dataset/rir_cfg_4`.
```
python generate_rirs.py
```

## Train & Test
**Train** Narrow-band Conformer (NBC) on the 0-th GPU with config file `configs/NBC-fit.yaml` (replace the rir & clean speech dir before training, and NB-BLSTM `configs/NB-BLSTM-fit.yaml` can be trained and tested in the same way but mind to change the valid batch size). **The valid batch size = num of gpus used * batch_size for dataloader * accumulate_grad_batches.** In the following case, we have a valid batch size of 16= 1* 2 * 8.
```
python NBSSCLI.py fit --config=configs/NBC-fit.yaml --data.batch_size=[2,2] --trainer.accumulate_grad_batches=8 --trainer.gpus=0,
```
More gpus can be used by appending the gpu indexes to `trainer.gpus`, e.g. `--trainer.gpus=0,1,2,3,`.


**Resume** training from a checkpoint:
```
python NBSSCLI.py fit --config=logs/NBSS/version_x/config.yaml --data.batch_size=[2,2] --trainer.accumulate_grad_batches=8 --trainer.gpus=0, --ckpt_path=logs/NBSS/version_x/checkpoints/last.ckpt
```

**Test** the model trained (Dataset with different seeds will generate different wavs):
```
python NBSSCLI.py test --config=logs/NBSS/version_x/config.yaml --ckpt_path=logs/NBSS/version_x/checkpoints/epochY_neg_si_sdrZ.ckpt --trainer.gpus=0, --data.seeds="{'train':null,'val':2,'test':3}"
```


## Module Version
see models/arch/NBSS.py

## Citation
If you like this work and want to cite us, please cite [4, 5].
