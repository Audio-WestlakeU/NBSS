from typing import Dict, List, Tuple, Union
from torchmetrics.audio import *
from torchmetrics.functional.audio import *
from torch import Tensor

ALL_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'SNR', 'NB_PESQ', 'WB_PESQ', 'STOI']


def cal_metrics_functional(
    metric_list: List[str],
    preds: Tensor,
    target: Tensor,
    original: Union[Tensor, Dict[str, Tensor]],
    fs: int,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    preds_cpu = preds.detach().cpu()
    target_cpu = target.detach().cpu()

    if isinstance(original, Tensor):
        input_metrics = {}
        original_cpu = original.detach().cpu()
    else:
        input_metrics = original
        original_cpu = None

    metrics = {}
    imp_metrics = {}

    for m in metric_list:
        mname = m.lower()
        if m.upper() == 'SDR':
            metric_func = lambda: signal_distortion_ratio(preds, target).mean().detach().cpu()
            input_metric_func = lambda: signal_distortion_ratio(original, target).mean().detach().cpu()
        elif m.upper() == 'SI_SDR':
            metric_func = lambda: scale_invariant_signal_distortion_ratio(preds, target).mean().detach().cpu()
            input_metric_func = lambda: scale_invariant_signal_distortion_ratio(original, target).mean().detach().cpu()
        elif m.upper() == 'SI_SNR':
            metric_func = lambda: scale_invariant_signal_noise_ratio(preds, target).mean().detach().cpu()
            input_metric_func = lambda: scale_invariant_signal_noise_ratio(original, target).mean().detach().cpu()
        elif m.upper() == 'SNR':
            metric_func = lambda: signal_noise_ratio(preds, target).mean().detach().cpu()
            input_metric_func = lambda: signal_noise_ratio(original, target).mean().detach().cpu()
        elif m.upper() == 'NB_PESQ':
            metric_func = lambda: perceptual_evaluation_speech_quality(preds_cpu, target_cpu, fs, 'nb').mean()
            input_metric_func = lambda: perceptual_evaluation_speech_quality(original_cpu, target_cpu, fs, 'nb').mean()
        elif m.upper() == 'WB_PESQ':
            metric_func = lambda: perceptual_evaluation_speech_quality(preds_cpu, target_cpu, fs, 'wb').mean()
            input_metric_func = lambda: perceptual_evaluation_speech_quality(original_cpu, target_cpu, fs, 'wb').mean()
        elif m.upper() == 'STOI':
            metric_func = lambda: short_time_objective_intelligibility(preds_cpu, target_cpu, fs).mean()
            input_metric_func = lambda: short_time_objective_intelligibility(original_cpu, target_cpu, fs).mean()
        else:
            raise ValueError('Unkown audio metric ' + m)

        metrics[m.lower()] = metric_func()
        if 'input_' + mname not in input_metrics.keys():
            input_metrics['input_' + mname] = input_metric_func()
        imp_metrics[mname + '_i'] = metrics[mname] - input_metrics['input_' + mname]

    return metrics, input_metrics, imp_metrics
