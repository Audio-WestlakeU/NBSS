from typing import Dict, List, Optional, Tuple, Union
import warnings
from torchmetrics import Metric
from torchmetrics.collections import MetricCollection
from torchmetrics.audio import *
from torchmetrics.functional.audio import *
from torch import Tensor
import torch
import pesq as pesq_backend
import numpy as np
from typing import *
from models.utils.dnsmos import deep_noise_suppression_mean_opinion_score

ALL_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'SNR', 'NB_PESQ', 'WB_PESQ', 'STOI', 'DNSMOS', 'pDNSMOS']


def get_metric_list_on_device(device: Optional[str]):
    metric_device = {
        None: ['SDR', 'SI_SDR', 'SNR', 'SI_SNR', 'NB_PESQ', 'WB_PESQ', 'STOI', 'ESTOI', 'DNSMOS', 'pDNSMOS'],
        "cpu": ['NB_PESQ', 'WB_PESQ', 'STOI', 'ESTOI'],
        "gpu": ['SDR', 'SI_SDR', 'SNR', 'SI_SNR', 'DNSMOS', 'pDNSMOS'],
    }
    return metric_device[device]


def cal_metrics_functional(
    metric_list: List[str],
    preds: Tensor,
    target: Tensor,
    original: Optional[Tensor],
    fs: int,
    device_only: Literal['cpu', 'gpu', None] = None,  # cpu-only: pesq, stoi;
    chunk: Tuple[float, float] = None,  # (chunk length, hop length) in seconds for chunk-wise metric evaluation
    suffix: str = "",
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    metrics, input_metrics, imp_metrics = {}, {}, {}
    if chunk is not None:
        clen, chop = int(fs * chunk[0]), int(fs * chunk[1])
        for i in range(int((preds.shape[-1] / fs - chunk[0]) / chunk[1]) + 1):
            metrics_chunk, input_metrics_chunk, imp_metrics_chunk = cal_metrics_functional(
                metric_list,
                preds[..., i * chop:i * chop + clen],
                target[..., i * chop:i * chop + clen],
                original[..., i * chop:i * chop + clen] if original is not None else None,
                fs,
                device_only,
                chunk=None,
                suffix=f"_{i*chunk[1]+1}s-{i*chunk[1]+chunk[0]}s",
            )
            metrics.update(metrics_chunk), input_metrics.update(input_metrics_chunk), imp_metrics.update(imp_metrics_chunk)

    if device_only is None or device_only == 'cpu':
        preds_cpu = preds.detach().cpu()
        target_cpu = target.detach().cpu()
        original_cpu = original.detach().cpu() if original is not None else None
    else:
        preds_cpu = None
        target_cpu = None
        original_cpu = None

    for m in metric_list:
        mname = m.lower()
        if m.upper() not in get_metric_list_on_device(device=device_only):
            continue

        if m.upper() == 'SDR':
            ## not use signal_distortion_ratio for it gives NaN sometimes
            metric_func = lambda: signal_distortion_ratio(preds, target).detach().cpu()
            input_metric_func = lambda: signal_distortion_ratio(original, target).detach().cpu()
            # assert preds.dim() == 2 and target.dim() == 2 and original.dim() == 2, '(spk, time)!'
            # metric_func = lambda: torch.tensor(bss_eval_sources(target_cpu.numpy(), preds_cpu.numpy(), False)[0]).mean().detach().cpu()
            # input_metric_func = lambda: torch.tensor(bss_eval_sources(target_cpu.numpy(), original_cpu.numpy(), False)[0]).mean().detach().cpu()
        elif m.upper() == 'SI_SDR':
            metric_func = lambda: scale_invariant_signal_distortion_ratio(preds, target).detach().cpu()
            input_metric_func = lambda: scale_invariant_signal_distortion_ratio(original, target).detach().cpu()
        elif m.upper() == 'SI_SNR':
            metric_func = lambda: scale_invariant_signal_noise_ratio(preds, target).detach().cpu()
            input_metric_func = lambda: scale_invariant_signal_noise_ratio(original, target).detach().cpu()
        elif m.upper() == 'SNR':
            metric_func = lambda: signal_noise_ratio(preds, target).detach().cpu()
            input_metric_func = lambda: signal_noise_ratio(original, target).detach().cpu()
        elif m.upper() == 'NB_PESQ':
            metric_func = lambda: perceptual_evaluation_speech_quality(preds_cpu, target_cpu, fs, 'nb', n_processes=0)
            input_metric_func = lambda: perceptual_evaluation_speech_quality(original_cpu, target_cpu, fs, 'nb', n_processes=0)
        elif m.upper() == 'WB_PESQ':
            metric_func = lambda: perceptual_evaluation_speech_quality(preds_cpu, target_cpu, fs, 'wb', n_processes=0)
            input_metric_func = lambda: perceptual_evaluation_speech_quality(original_cpu, target_cpu, fs, 'wb', n_processes=0)
        elif m.upper() == 'STOI':
            metric_func = lambda: short_time_objective_intelligibility(preds_cpu, target_cpu, fs)
            input_metric_func = lambda: short_time_objective_intelligibility(original_cpu, target_cpu, fs)
        elif m.upper() == 'ESTOI':
            metric_func = lambda: short_time_objective_intelligibility(preds_cpu, target_cpu, fs, extended=True)
            input_metric_func = lambda: short_time_objective_intelligibility(original_cpu, target_cpu, fs, extended=True)
        elif m.upper() == 'DNSMOS':
            metric_func = lambda: deep_noise_suppression_mean_opinion_score(preds, fs, False)
            input_metric_func = lambda: deep_noise_suppression_mean_opinion_score(original, fs, False)
        elif m.upper() == 'PDNSMOS':  # personalized DNSMOS
            metric_func = lambda: deep_noise_suppression_mean_opinion_score(preds, fs, True)
            input_metric_func = lambda: deep_noise_suppression_mean_opinion_score(original, fs, True)
        else:
            raise ValueError('Unkown audio metric ' + m)

        if m.upper() == 'WB_PESQ' and fs == 8000:
            # warnings.warn("There is narrow band (nb) mode only when sampling rate is 8000Hz")
            continue  # Note there is narrow band (nb) mode only when sampling rate is 8000Hz

        try:
            if mname == 'dnsmos':
                # p808_mos, mos_sig, mos_bak, mos_ovr
                m_val = metric_func().cpu().numpy()

                for idx, mid in enumerate(['p808', 'sig', 'bak', 'ovr']):
                    mname_i = mname + '_' + mid + suffix
                    metrics[mname_i] = np.mean(m_val[..., idx]).item()
                    metrics[mname_i + '_all'] = m_val[..., idx].tolist()
                    if original is None:
                        continue

                    if 'input_' + mname_i not in input_metrics.keys():
                        im_val = input_metric_func().cpu().numpy()
                        input_metrics['input_' + mname_i] = np.mean(im_val[..., idx]).item()
                        input_metrics['input_' + mname_i + '_all'] = im_val[..., idx].tolist()

                    imp_metrics[mname_i + '_i'] = metrics[mname_i] - input_metrics['input_' + mname_i]  # _i means improvement
                    imp_metrics[mname_i + '_all' + '_i'] = (m_val[..., idx] - im_val[..., idx]).tolist()
                continue

            mname = mname + suffix
            m_val = metric_func().cpu().numpy()
            metrics[mname] = np.mean(m_val).item()
            metrics[mname + '_all'] = m_val.tolist()  # _all means not averaged
            if original is None:
                continue

            if 'input_' + mname not in input_metrics.keys():
                im_val = input_metric_func().cpu().numpy()
                input_metrics['input_' + mname] = np.mean(im_val).item()
                input_metrics['input_' + mname + '_all'] = im_val.tolist()

            imp_metrics[mname + '_i'] = metrics[mname] - input_metrics['input_' + mname]  # _i means improvement
            imp_metrics[mname + '_all' + '_i'] = (m_val - im_val).tolist()
        except Exception as e:
            metrics[mname] = None
            metrics[mname + '_all'] = None
            if 'input_' + mname not in input_metrics.keys():
                input_metrics['input_' + mname] = None
                input_metrics['input_' + mname + '_all'] = None
            imp_metrics[mname + '_i'] = None
            imp_metrics[mname + '_i' + '_all'] = None

    return metrics, input_metrics, imp_metrics

def mypesq(preds: np.ndarray, target: np.ndarray, mode: str, fs: int) -> np.ndarray:
    # 使用ndarray是因为tensor会在linux上会导致一些多进程的错误
    ori_shape = preds.shape
    if type(preds) == Tensor:
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    else:
        assert type(preds) == np.ndarray, type(preds)
        assert type(target) == np.ndarray, type(target)

    if preds.ndim == 1:
        pesq_val = pesq_backend.pesq(fs=fs, ref=target, deg=preds, mode=mode)
    else:
        preds = preds.reshape(-1, ori_shape[-1])
        target = target.reshape(-1, ori_shape[-1])
        pesq_val = np.empty(shape=(preds.shape[0]))
        for b in range(preds.shape[0]):
            pesq_val[b] = pesq_backend.pesq(fs=fs, ref=target[b, :], deg=preds[b, :], mode=mode)
        pesq_val = pesq_val.reshape(ori_shape[:-1])
    return pesq_val


def cal_pesq(ys: np.ndarray, ys_hat: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    try:
        if sample_rate == 16000:
            wb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='wb').mean()
            nb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='nb').mean()
            return [wb_pesq_val, nb_pesq_val]
        elif sample_rate == 8000:
            nb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='nb').mean()
            return [None, nb_pesq_val]
        else:
            ...
    except Exception as e:
        ...
        # warnings.warn(str(e))
    return [None, None]


def recover_scale(preds: Tensor, mixture: Tensor, scale_src_together: bool, norm_if_exceed_1: bool = True) -> Tensor:
    """recover wav's original scale by solving min ||Y^T a - X||F, cuz sisdr will lose scale

    Args:
        preds: prediction, shape [batch, n_src, time]
        mixture: mixture or noisy or reverberant signal, shape [batch, time]
        scale_src_together: keep the relative ennergy level between sources. can be used for scale-invariant SA-SDR
        norm_max_if_exceed_1: norm the magitude if exceeds one

    Returns:
        Tensor: the scale-recovered preds
    """
    # TODO: add some kind of weighting mechanism to make the predicted scales more precise
    # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
    if scale_src_together:
        a = torch.linalg.lstsq(preds.sum(dim=-2, keepdim=True).transpose(-1, -2), mixture.unsqueeze(-1)).solution
    else:
        a = torch.linalg.lstsq(preds.transpose(-1, -2), mixture.unsqueeze(-1)).solution

    preds = preds * a

    if norm_if_exceed_1:
        # normalize the audios so that the maximum doesn't exceed 1
        max_vals = torch.max(torch.abs(preds), dim=-1).values
        norm = torch.where(max_vals > 1, max_vals, 1)
        preds = preds / norm.unsqueeze(-1)
    return preds


if __name__ == "__main__":
    x, y, m = torch.rand((2, 8000 * 8)), torch.rand((2, 8000 * 8)), torch.rand((2, 8000 * 8))
    m, im, mi = cal_metrics_functional(["si_sdr"], preds=x, target=y, original=m, fs=8000, chunk=[4, 1])
    print(m)
    print(im)
    print(mi)
