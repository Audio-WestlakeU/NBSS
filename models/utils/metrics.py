from typing import Dict, List, Optional, Tuple, Union
import warnings
from torchmetrics import Metric
from torchmetrics.collections import MetricCollection
from torchmetrics.audio import *
from torchmetrics.functional.audio import *
from torch import Tensor

ALL_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'SNR', 'NB_PESQ', 'WB_PESQ', 'STOI']


def construct_audio_MetricCollection(
    metrics: List[str],
    prefix: str = '',
    postfix: str = '',
    fs: int = None,
) -> MetricCollection:
    md: Dict[str, Metric] = {}
    for m in metrics:
        mname = prefix + m.lower() + postfix
        if m.upper() == 'SDR':
            md[mname] = SignalDistortionRatio()
        elif m.upper() == 'SI_SDR':
            md[mname] = ScaleInvariantSignalDistortionRatio()
        elif m.upper() == 'SI_SNR':
            md[mname] = ScaleInvariantSignalNoiseRatio()
        elif m.upper() == 'SNR':
            md[mname] = SignalNoiseRatio()
        elif m.upper() == 'NB_PESQ':
            md[mname] = PerceptualEvaluationSpeechQuality(fs, mode='nb').cpu()
        elif m.upper() == 'WB_PESQ':
            md[mname] = PerceptualEvaluationSpeechQuality(fs, mode='wb').cpu()
        elif m.upper() == 'STOI':
            md[mname] = ShortTimeObjectiveIntelligibility(fs).cpu()
        else:
            raise ValueError('Unkown audio metric ' + m)

    return MetricCollection(md)


def cal_metrics(
    preds: Tensor,
    target: Tensor,
    original: Union[Tensor, Dict[str, Tensor]],
    mc: MetricCollection,
    input_mc: Optional[MetricCollection] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    """calculate metrics, input_metrics, imp_metrics

    Args:
        preds: prediction
        target: target
        original: original signal or input_metrics
        mc: MetricCollection 
        input_mc: Input MetricCollection if original signal is given, else None

    Returns:
        metrics, input_metrics, imp_metrics
    """
    metrics = mc(preds, target)
    if isinstance(original, Tensor):
        if input_mc == None:
            raise ValueError('input_mc cannot be None when original signal is given, i.e. original is a Tensor')
        input_metrics = input_mc(original, target)
    else:
        input_metrics = original
    imp_metrics = {}
    for k, v in metrics.items():
        v = v.detach().cpu()
        iv = input_metrics['input_' + k].detach().cpu()
        metrics[k] = v
        input_metrics['input_' + k] = iv
        imp_metrics[k + '_i'] = v - iv

    return metrics, input_metrics, imp_metrics


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
            ## not use signal_distortion_ratio for it gives NaN sometimes
            metric_func = lambda: signal_distortion_ratio(preds, target).mean().detach().cpu()
            input_metric_func = lambda: signal_distortion_ratio(original, target).mean().detach().cpu()
            # assert preds.dim() == 2 and target.dim() == 2 and original.dim() == 2, '(spk, time)!'
            # metric_func = lambda: torch.tensor(bss_eval_sources(target_cpu.numpy(), preds_cpu.numpy(), False)[0]).mean().detach().cpu()
            # input_metric_func = lambda: torch.tensor(bss_eval_sources(target_cpu.numpy(), original_cpu.numpy(), False)[0]).mean().detach().cpu()
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

        if m.upper() == 'WB_PESQ' and fs == 8000:
            warnings.warn("There is narrow band (nb) mode only when sampling rate is 8000Hz")
            continue  # Note there is narrow band (nb) mode only when sampling rate is 8000Hz

        metrics[m.lower()] = metric_func()
        if 'input_' + mname not in input_metrics.keys():
            input_metrics['input_' + mname] = input_metric_func()
        imp_metrics[mname + '_i'] = metrics[mname] - input_metrics['input_' + mname]

    return metrics, input_metrics, imp_metrics
