# Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training

Detailed introduction (with images and examples) about this work can be found [in English](https://quancs.github.io/blog/nbss/) or [in Chinese](https://quancs.github.io/zh-cn/blog/nbss/)

## Requirements
```
# TorchMetrics
pip install git+https://github.com/quancs/metrics.git@personal
# PytorchLightning (version 1.5)
pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip
```

## Generate rirs
Generate rirs using `configs/rir_cfg_3.json`, and the generated rirs are placed in `dataset/rir_cfg_3`.
```
python generate_rirs.py
```

## Train
Train NBSS on the 0-th GPU with config file `configs/ifp/fit-WavForm.yaml` (replace the rir & clean speech dir before training).
```
python cli_ifp.py --config configs/ifp/fit-WavForm.yaml fit --trainer.gpu 0, --seed_everything 2 --model.exp_name "train"
```

## Test
Test on the 0-th GPU.
Different seeds for dataset will generate different wavs.
```
python cli_ifp.py --config logs/NBSS_ifp/version_66/config-test.yaml test --model logs/NBSS_ifp/version_66/hparams.yaml --ckpt_path logs/NBSS_ifp/version_66/checkpoints/epoch707_neg_si_sdr-13.7777.ckpt --trainer.gpus=0, --data.seeds="{'train':null,'val':2,'test':5}" --model.exp_name="test"
```

## Citation
If you like this work and want to cite us, please use:
```
@article{quan_multi-channel_2021,
	title = {Multi-channel {Narrow}-{Band} {Deep} {Speech} {Separation} with {Full}-band {Permutation} {Invariant} {Training}},
	journal = {arXiv preprint arXiv:2110.05966},
	author = {Quan, Changsheng and Li, Xiaofei},
	year = {2021},
}
```
