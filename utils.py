import os, torch
import random
import numpy as np


def mse_complex(a, b):
    return torch.mean(torch.abs(a - b)**2)


def mse_complex_batch(target, estimate):
    batch_size = target.shape[0]
    res = torch.abs(target - estimate)**2
    return torch.mean(res.view(batch_size, -1), dim=1)


def mse_batch(target, estimate):
    batch_size = target.shape[0]
    res = (target - estimate)**2
    return torch.mean(res.view(batch_size, -1), dim=1)


def cut_to_same_len(batches):
    """cut time domain wav signals to same time length

    Args:
        batches (list): a list of [x, ys] or [x, ys, paths]

    Returns:
        list: [x, ys] or [x, ys, [paths, ...] ]
    """
    # cut audios to the same length
    min_time_steps = batches[0][0].shape[-1]
    for batch in batches:
        if batch[0].shape[-1] < min_time_steps:
            min_time_steps = batch[0].shape[-1]

    # cut audios to the same length
    x_batch, ys_batch, paths_list = None, None, []
    for batch in batches:
        if len(batch) == 2:
            x_bi, ys_bi = batch
        else:
            x_bi, ys_bi, paths = batch
            paths_list.append(paths)

        # choose a random start
        start = 0
        if x_bi.shape[-1] > min_time_steps:
            start = torch.randint(low=0, high=x_bi.shape[-1] - min_time_steps, size=[1])[0]
        new_x = x_bi[..., start:(start + min_time_steps)]
        new_ys = ys_bi[..., start:(start + min_time_steps)]

        # cat them
        if x_batch is None:
            x_batch = new_x
            ys_batch = new_ys
        else:
            x_batch = torch.cat([x_batch, new_x], dim=0)
            ys_batch = torch.cat([ys_batch, new_ys], dim=0)
    if len(paths_list) == 0:
        return x_batch, ys_batch
    else:
        return x_batch, ys_batch, paths_list


def encode(x, ys, band_num, ref_chn_idx, ft_len, ft_overlap, time_step=None, window=None):
    batch_size, chn_num, time = x.shape
    _, spk_num, _ = ys.shape

    if window == None:
        window = torch.hann_window(ft_len, device=x.device)

    # stft x
    x = x.reshape((batch_size * chn_num, time))
    # TODO torch的stft似乎比较耗时，或许可以考虑换成scipy的stft，以及比较两个stft的速度和结果是否相同，同时也需要检查是否使用了torch的istft作为配对的操作
    X = torch.stft(x, n_fft=ft_len, hop_length=ft_overlap, window=window, win_length=ft_len, return_complex=True)
    X = X.reshape((batch_size, chn_num, X.shape[-2], X.shape[-1]))  # (batch, channel, freq, time)
    X = X.permute(0, 2, 3, 1)  # (batch, freq, time, channel)

    # stft ys
    ys = ys.reshape((batch_size * spk_num, -1))
    Ys = torch.stft(ys, n_fft=ft_len, hop_length=ft_overlap, window=window, win_length=ft_len, return_complex=True)
    Ys = Ys.reshape((batch_size, spk_num, Ys.shape[-2], Ys.shape[-1]))  # (batch, spk, freq, time)
    Ys = Ys.permute(0, 2, 1, 3)  # (batch, freq, spk, time) for the reason of narrow band pit (each freq can be regarded as a part of batch)

    batch_size, freq_num, time, chn_num = X.shape

    # concatenate neighbour freq bands
    if time_step is None:
        time_step = time
    step_inc = time_step // 2

    if time < time_step:  # 补0
        nX = torch.zeros((batch_size, freq_num, time_step, chn_num), dtype=X.dtype, device=x.device)
        nYs = torch.zeros((batch_size, freq_num, spk_num, time_step), dtype=Ys.dtype, device=x.device)
        nX[:, :, 0:time, :] = X[:, :, :, :]
        nYs[:, :, :, 0:time] = Ys[:, :, :, :]
        X = nX
        Ys = nYs
    elif time > time_step:  # 加入新的batch
        bX = []
        bYs = []
        for t in range(0, time - time_step, step_inc):
            bX.append(X[:, :, t:t + time_step, :].clone())
            bYs.append(Ys[:, :, :, t:t + time_step].clone())
        X = torch.cat(bX, dim=0)
        Ys = torch.cat(bYs, dim=0)

    batch_size = X.shape[0]

    # normalization by using ref_channel
    # TODO 测试不使用均值的norm
    Xr = X[..., ref_chn_idx].clone()  # copy
    XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of Xm
    X[:, :, :, :] /= (XrMM.reshape(batch_size, freq_num, 1, 1) + 1e-8)

    # concatenate neighbour freq bands
    X_bands = torch.zeros((batch_size, freq_num - band_num + 1, time_step, band_num, chn_num, 2), device=x.device)
    Ys_bands = torch.zeros((batch_size, freq_num - band_num + 1, spk_num, time_step, band_num), device=x.device, dtype=torch.complex64)
    Xr_bands = torch.zeros((batch_size, freq_num - band_num + 1, time_step, band_num), device=x.device, dtype=torch.complex64)
    XrMM_bands = torch.zeros((batch_size, freq_num - band_num + 1, band_num), device=x.device)
    for f in range(freq_num - band_num + 1):
        for band in range(band_num):
            X_bands[:, f, :, band, :, 0] = torch.real(X[:, f + band, :, :])
            X_bands[:, f, :, band, :, 1] = torch.imag(X[:, f + band, :, :])
            Ys_bands[:, f, :, :, band] = Ys[:, f + band, :, :]
            Xr_bands[:, f, :, band] = Xr[:, f + band, :]
            XrMM_bands[:, f, band] = XrMM[:, f + band]

    # TODO remove this assert in run time
    # for band in range(band_num):
    #     assert torch.allclose(Ys[:, 0:freq_num - band, :, :], Ys_bands[:, band:freq_num, :, :, band]), 'ERROR'
    #     assert torch.allclose(Xr[:, 0:freq_num - band, :], Xr_bands[:, band:freq_num, :, band]), 'ERROR'
    #     assert torch.allclose(XrMM[:, 0:freq_num - band], XrMM_bands[:, band:freq_num, band]), 'ERROR'
    #     assert torch.allclose(X[:, 0:freq_num - band, :, :], (X_bands[:, band:freq_num, :, band, :, 0] + X_bands[:, band:freq_num, :, band, :, 1] * 1j)), 'ERROR'
    #     assert torch.allclose(XrMM_bands[:, :, band], torch.abs(Xr_bands[:, :, :, band]).mean(dim=2)), 'ERROR'
    return X_bands, Ys_bands, Xr_bands, XrMM_bands


def decode(Ys_bands_hat_per, ft_len, ft_overlap, original_len, window=None):
    """输入解完Permutation的版本，输出时域信号

    Args:
        Ys_bands_hat_per (torch.Tensor): permuted version of Ys_bands_hat

    Returns:
        torch.Tensor: time domain signal of shape [batch_size, spk_num, time]
    """
    batch_size, freq_num_band, spk_num, time, band_num = Ys_bands_hat_per.shape
    freq_num = freq_num_band + band_num - 1
    # align
    Ys_bands_hat_per_aligned = torch.empty((batch_size, freq_num, spk_num, time, band_num), device=Ys_bands_hat_per.device, dtype=torch.complex64)
    for f in range(freq_num):
        for b in range(band_num):
            if f - b >= 0 and f - b < freq_num_band:
                Ys_bands_hat_per_aligned[:, f, :, :, b] = Ys_bands_hat_per[:, f - b, :, :, b]
            else:
                valid_bands = []
                if not (f - b >= 0):
                    for bb in range(f + 1):
                        valid_bands.append(Ys_bands_hat_per[:, f - bb, :, :, bb])
                else:  # not (f - b < freq_num_band)
                    for bb in range(f - freq_num_band + 1, band_num):
                        valid_bands.append(Ys_bands_hat_per[:, f - bb, :, :, bb])

                vbcat = torch.cat(valid_bands, 0).view(len(valid_bands), *valid_bands[0].shape)
                Ys_bands_hat_per_aligned[:, f, :, :, b] = torch.mean(vbcat, dim=0)[:, :, :]

    # average
    Ys_hat = torch.mean(Ys_bands_hat_per_aligned, dim=4).permute(0, 2, 1, 3)  # (batch, spk_num, freq, time) from (batch, freq, spk_num, time)

    if window == None:
        window = torch.hann_window(ft_len, device=Ys_hat.device)
    ys_hat = torch.istft(Ys_hat.reshape(batch_size * spk_num, freq_num, time), n_fft=ft_len, hop_length=ft_overlap, window=window, win_length=ft_len, length=original_len)
    ys_hat = ys_hat.reshape(batch_size, spk_num, ys_hat.shape[1])
    return ys_hat


def normalized_cov_batch(targets, estimates):
    batch_size = targets.shape[0]
    targets = targets.view(batch_size, -1)
    estimates = estimates.view(batch_size, -1)

    std_t, mean_t = torch.std_mean(targets, dim=1)
    std_e, mean_e = torch.std_mean(estimates, dim=1)

    mean_t = mean_t.view(batch_size, 1)
    mean_e = mean_e.view(batch_size, 1)

    cov = torch.mean((targets - mean_t) * (estimates - mean_e), dim=1)
    norm_cov = cov / (std_t * std_e)  # norm_cov is close to np.corrcoef(targets, estimates)
    return norm_cov


def perm_by_ps(X, ps, spk_dim=2):
    """permute X by ps

    Args:
        X (Tensor): has a shape of [batch_size, freq_num, ..., spk_num, ...]
        ps (List): permutations
        spk_dim (int): the index of spk_num

    Returns:
        X_per: the permuted tensor
    """

    batch_size, freq_num = X.shape[0:2]
    spk_num = X.shape[spk_dim]

    # 创建出用于合并之后的batch_freq维度的ps对应的index
    bfidx = torch.arange(0, batch_size * freq_num, device=X.device, dtype=ps.dtype)
    bfidx = bfidx.view(-1, 1)
    bfidx = bfidx.repeat(1, spk_num)
    bfidx = bfidx.view(-1)  # now, bfidx has batch_size * freq_num * spk_num elements
    idx = bfidx * 2 + ps.view(-1)
    if spk_dim == 2:
        X = X.view(batch_size * freq_num * spk_num, *X.shape[3:])
        X_per = X[idx, ...].contiguous()
        X_per = X_per.view(batch_size, freq_num, spk_num, *X.shape[1:])
    elif spk_dim == 3:
        X = X.transpose(2, spk_dim).contiguous()
        X = X.view(batch_size * freq_num * spk_num, *X.shape[3:])
        X_per = X[idx, ...].contiguous()
        X_per = X_per.view(batch_size, freq_num, spk_num, *X.shape[1:])
        X_per = X_per.transpose(2, spk_dim).contiguous()

        # TODO remove below when tested ok
        if len(ps) != batch_size:
            ps = [ps[v * freq_num:(v + 1) * freq_num] for v in range(len(ps) // freq_num)]
        assert len(ps) == batch_size, "X doean't match with ps"
        assert len(ps[0]) == freq_num, "X doean't match with ps"

        X_per3 = torch.empty(X.shape, device=X.device, dtype=X.dtype)

        for b in range(batch_size):
            for f in range(freq_num):
                if spk_dim == 2:
                    X_per3[b, f, ...] = X[b, f, ps[b][f], ...]
                elif spk_dim == 3:
                    X_per3[b, f, ...] = X[b, f, :, ps[b][f], ...]
                else:
                    raise Exception('Unspported spk_dim={}'.format(spk_dim))

        equal = torch.equal(X_per3, X_per)
        if equal == False:
            raise Exception("torch.equal(X_per, X_per2) == False!!!!")
    else:
        raise Exception('Unspported spk_dim={}'.format(spk_dim))

    return X_per


def perm_by_truth(Ys_bands_hat, Ys_bands, loss_func=mse_complex_batch):
    """使用真值来解permutation

    Args:
        Ys_bands_hat (torch.Tensor): dtype = complex or real
        Ys_bands (torch.Tensor): dtype = complex or real, ground truth
        loss_func: a function passed to pit_loss

    Returns:
        torch.Tensor: the permuted version of Ys_bands_hat of shape [batch_size, freq_num, spk_num, time, band_num]
        bps: best permutations
        loss: permutation loss
    """
    batch_size, freq_num, spk_num, time, band_num = Ys_bands_hat.shape

    Ys_bands_hat = Ys_bands_hat.view(batch_size * freq_num, spk_num, time, band_num)
    Ys_bands = Ys_bands.view(batch_size * freq_num, spk_num, time, band_num)

    # permute
    losses, bps = pit_loss_batch(targets=Ys_bands, estimates=Ys_bands_hat, loss_func=loss_func)
    bps = [bps[v * freq_num:(v + 1) * freq_num] for v in range(len(bps) // freq_num)]
    avg_loss = torch.mean(losses)

    Ys_bands_hat = Ys_bands_hat.view(batch_size, freq_num, spk_num, time, band_num)
    Ys_bands_hat_per = perm_by_ps(Ys_bands_hat, bps)

    return Ys_bands_hat_per, bps, avg_loss


def perm_by_overlap(Ys_bands_hat, loss_func=mse_complex_batch, return_losses=False):
    """使用重叠频带解permutation

    Args:
        Ys_bands_hat (torch.Tensor): dtype = complex or real
        loss_func: a function passed to pit_loss
        return_losses: whether to return the losses of all permutations, default False

    Returns:
        torch.Tensor: the permuted version of Ys_bands_hat of shape [batch_size, freq_num, spk_num, time, band_num]
        bps: best permutations
        loss: permutation loss
        losses_best [optional]: the best losses of all batches
        losses_all [optional]: the losses of all permutations
    """
    batch_size, freq_num, spk_num, time, band_num = Ys_bands_hat.shape

    # full part
    freq_num_fullpart = freq_num - 1
    Ys_bands_hat_f1 = Ys_bands_hat[:, 0:freq_num - 1, :, :, 1:band_num].contiguous()
    Ys_bands_hat_f2 = Ys_bands_hat[:, 1:freq_num, :, :, 0:band_num - 1].contiguous()

    Ys_bands_hat_f1 = Ys_bands_hat_f1.view(batch_size * freq_num_fullpart, spk_num, time, band_num - 1)
    Ys_bands_hat_f2 = Ys_bands_hat_f2.view(batch_size * freq_num_fullpart, spk_num, time, band_num - 1)
    losses_best_fullpart, bps_fullpart, losses_all_fullpart = pit_loss_batch(targets=Ys_bands_hat_f1, estimates=Ys_bands_hat_f2, loss_func=loss_func, return_losses=True)

    # combine
    losses_best = losses_best_fullpart.view(batch_size, freq_num - 1)
    losses_all = losses_all_fullpart.view(batch_size, freq_num - 1, losses_all_fullpart.shape[1])

    bps_fullpart = bps_fullpart.view(batch_size, freq_num_fullpart, spk_num)
    bps_relative = torch.empty((batch_size, freq_num, spk_num), device=bps_fullpart.device, dtype=bps_fullpart.dtype)
    bps_otherpart = torch.arange(0, spk_num, 1, device=bps_fullpart.device).repeat(batch_size, 1)
    bps_relative[:, 0, :] = bps_otherpart[:, :]
    bps_relative[:, 1:, :] = bps_fullpart[:, :, :]

    # to absolute permutation
    bps = torch.empty(bps_relative.shape, device=bps_fullpart.device, dtype=bps_fullpart.dtype)
    for b in range(batch_size):
        bps[b, 0, :] = bps_relative[b][0]
        for f in range(1, freq_num):
            ps_f_relative = bps_relative[b][f]
            bps[b, f, :] = bps[b, f - 1, ps_f_relative]

    # result
    Ys_bands_hat_per = perm_by_ps(Ys_bands_hat, bps)
    avg_loss = torch.mean(losses_best)

    if return_losses == False:
        return Ys_bands_hat_per, bps, avg_loss
    else:
        return Ys_bands_hat_per, bps, avg_loss, losses_best, losses_all


def robust_neg_cor(f1, f2):
    # f1, f2 [batch, time, band]
    std1, u1 = torch.std_mean(f1, dim=1)
    std2, u2 = torch.std_mean(f2, dim=1)
    p12 = f1[...] * f2[...]
    u12 = torch.mean(p12, dim=1)
    cor = (u12 - u1 * u2) / (std1 * std2)
    cor = torch.mean(cor, dim=1)
    return -cor


def perm_by_correlation(Ys_bands_hat, loss_func=robust_neg_cor, return_losses=False):
    # TODO change to return tensor bps
    """使用相邻频带相关性解permutation

    Args:
        Ys_bands_hat (torch.Tensor): dtype = complex or real
        loss_func: a function passed to pit_loss
        return_losses: whether to return the losses of all permutations, default False

    Returns:
        torch.Tensor: the permuted version of Ys_bands_hat of shape [batch_size, freq_num, spk_num, time, band_num]
        bps: best permutations
        loss: permutation loss
        losses_best [optional]: the best losses of all batches
        losses_all [optional]: the losses of all permutations
    """
    Ys_bands_hat_given = Ys_bands_hat
    Ys_bands_hat = torch.abs(Ys_bands_hat)
    batch_size, freq_num, spk_num, time, band_num = Ys_bands_hat.shape

    # full part
    freq_num_fullpart = freq_num - 1
    Ys_bands_hat_f1 = Ys_bands_hat[:, 0:freq_num - 1, :, :, :].contiguous()
    Ys_bands_hat_f2 = Ys_bands_hat[:, 1:freq_num, :, :, :].contiguous()

    Ys_bands_hat_f1 = Ys_bands_hat_f1.view(batch_size * freq_num_fullpart, spk_num, time, band_num)
    Ys_bands_hat_f2 = Ys_bands_hat_f2.view(batch_size * freq_num_fullpart, spk_num, time, band_num)
    losses_best_fullpart, bps_fullpart, losses_all_fullpart = pit_loss_batch(targets=Ys_bands_hat_f1, estimates=Ys_bands_hat_f2, loss_func=loss_func, return_losses=True)

    # combine
    losses_best = losses_best_fullpart.view(batch_size, freq_num - 1)
    losses_all = losses_all_fullpart.view(batch_size, freq_num - 1, losses_all_fullpart.shape[1])

    bps_fullpart = bps_fullpart.view(batch_size, freq_num_fullpart, spk_num)
    bps_relative = torch.empty((batch_size, freq_num, spk_num), device=bps_fullpart.device, dtype=bps_fullpart.dtype)
    bps_otherpart = torch.arange(0, spk_num, 1, device=bps_fullpart.device).repeat(batch_size, 1)
    bps_relative[:, 0, :] = bps_otherpart[:, :]
    bps_relative[:, 1:, :] = bps_fullpart[:, :, :]

    # relative (local) permutation to absolute (global) permutation
    bps = torch.empty(bps_relative.shape, device=bps_fullpart.device, dtype=bps_fullpart.dtype)
    for b in range(batch_size):
        bps[b, 0, :] = bps_relative[b][0]
        for f in range(1, freq_num):
            ps_f_relative = bps_relative[b][f]
            bps[b, f, :] = bps[b, f - 1, ps_f_relative]

    # result
    Ys_bands_hat_per = perm_by_ps(Ys_bands_hat_given, bps)
    avg_loss = torch.mean(losses_best)

    if return_losses == False:
        return Ys_bands_hat_per, bps, avg_loss
    else:
        return Ys_bands_hat_per, bps, avg_loss, losses_best, losses_all


def permutation_analysis(bps_t: torch.Tensor, bps_e: torch.Tensor):
    bps_t = bps_t.detach()
    bps_e = bps_e.detach()

    with torch.no_grad():
        B, F, SPK = bps_t.shape
        # compute the global rightness
        eq_ele = bps_t == bps_e
        eqs = eq_ele[:, :, 0] * eq_ele[:, :, 1]  # 每个spk都正确时才正确
        rights = torch.sum(eqs, dim=1)
        wrongs = F - rights
        right_avg = torch.mean(torch.where(rights >= wrongs, rights, wrongs).float())

        # 相邻频带间的解P的正确的情况：每对相邻频带算一次是否正确
        adjacent = torch.stack([eqs, eqs], dim=2)  # [B,F,2]
        adjacent = adjacent.reshape(B, F * 2)[:, 1:-1].reshape(B, F - 1, 2).contiguous()  # [B,F-1,2]
        wrong_adjacent = adjacent[:, :, 0] != adjacent[:, :, 1]  # 相邻频带一个正确、一个错误时，认为该相邻频带是错误的
        wrong_avg = torch.mean(wrong_adjacent.sum(dim=1).float())
        wrongfs = torch.nonzero(wrong_adjacent, as_tuple=False)

    return right_avg, wrong_avg, wrongfs
