# Multi-channel Speech Separation, Denoising and Dereverberation

A multichannel speech separation method.
The official repo of:  
[1] Changsheng Quan, Xiaofei Li. [Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training](https://arxiv.org/abs/2110.05966). In ICASSP 2022.  
[2] Changsheng Quan, Xiaofei Li. [Multichannel Speech Separation with Narrow-band Conformer](https://arxiv.org/abs/2204.04464). In Interspeech 2022.  
[3] Changsheng Quan, Xiaofei Li. [NBC2: Multichannel Speech Separation with Revised Narrow-band Conformer](https://arxiv.org/abs/2212.02076). arXiv:2212.02076.  
[4] Changsheng Quan, Xiaofei Li. [SpatialNet: Extensively Learning Spatial Information for Multichannel Joint Speech Separation, Denoising and Dereverberation](https://arxiv.org/abs/2307.16516). arXiv:2307.16516. **Code is coming soon.**

Audio examples can be found at [https://audio.westlake.edu.cn/Research/nbss.htm](https://audio.westlake.edu.cn/Research/nbss.htm) and [https://audio.westlake.edu.cn/Research/SpatialNet.htm](https://audio.westlake.edu.cn/Research/SpatialNet.htm).
More information about our group can be found at [https://audio.westlake.edu.cn](https://audio.westlake.edu.cn/Publications.htm).

## Requirements

```bash
pip install -r requirements.txt

# gpuRIR: check https://github.com/DavidDiazGuerra/gpuRIR
```

## Generate rirs

Generate rirs using `configs/rir_cfg_4.json`, and the generated rirs are placed in `dataset/rir_cfg_4`.

```bash
python generate_rirs.py
```

## Train & Test

This project is built on the `pytorch-lightning` package, in particular its [command line interface (CLI)](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_intermediate.html). Thus we recommond you to have some knowledge about the CLI in lightning.

**Train** NBC2 on the 0-th GPU with config file `configs/NBC2_small.yaml` or `configs/NBC2_large.yaml` (replace the rir & clean speech dir before training).

```bash
python NBSSCLI.py fit --config=configs/NBC2_small.yaml \
 --data.batch_size=[2,2] \ # batch size for train and val
 --trainer.accumulate_grad_batches=1 \
 --trainer.devices=0,
```

More gpus can be used by appending the gpu indexes to `trainer.devices`, e.g. `--trainer.devices=0,1,2,3,`.

Configs `configs/NBC-fit.yaml` and `configs/NB-BLSTM-fit.yaml` can be used to train and test NBC and NB-BLSTM in the same way respectively. But mind to change the number of utterances for training in one mini-batch. As we use ddp for distributed training, **the number of utterances in one mini-batch = num of gpus used * the number of utterances for dataloader * accumulate_grad_batches.** In the above command, we have 2 utterances in one mini-batch, i.e. 1 *2* 1.

**Resume** training from a checkpoint:

```bash
python NBSSCLI.py fit --config=logs/NBSS/version_x/config.yaml \
 --data.batch_size=[2,2] \
 --trainer.accumulate_grad_batches=1 \ 
 --trainer.devices=0, \ 
 --ckpt_path=logs/NBSS/version_x/checkpoints/last.ckpt
```

where `version_x` should be replaced with the version you want to resume.

**Test** the model trained (Dataset with different seeds will generate different wavs):

```bash
python NBSSCLI.py test --config=logs/NBSS/version_x/config.yaml \ 
 --ckpt_path=logs/NBSS/version_x/checkpoints/epochY_neg_si_sdrZ.ckpt \ 
 --trainer.devices=0, \ 
 --data.seeds="{'train':null,'val':2,'test':3}" \ 
 --data.audio_time_len="['headtail 4', 'headtail 4', 'headtail 4']"
```

where ```headtail``` is the speech overlap way and it can be ```mid```, ```full```, or ```startend``` (please refer to [3]).

## Module Version

see models/arch/NBSS.py
