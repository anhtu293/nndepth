import torch


def linear_sampler(corr, coords_lvl):
    """Sample from correlation volume with linear interpolation

    Parameters
    ----------
    corr :
        corr volume, shape (b*h1*w1, w2)
    coords_lvl :
        coord of points to sample, (b*h1*w1, 2*r+1)

    Returns
    -------
    sampled :
        sampled values, (b*h1*w1, 2*r+1)
    """
    _, w2 = corr.shape
    # trick to make clip operation compatible with tensorRT
    coords_lvl = coords_lvl / (w2 - 1)
    coords_lvl = torch.clamp(coords_lvl, 0, 1)
    coords_lvl = coords_lvl * (w2 - 1)
    # indices of points around coords_lvl
    idx0 = coords_lvl.floor().type(torch.int64)
    idx1 = coords_lvl.ceil().type(torch.int64)
    # values at those indices
    val0 = corr.gather(dim=1, index=idx0)
    val1 = corr.gather(dim=1, index=idx1)
    # weighted average
    coef = idx1 - coords_lvl
    return coef * val0 + (1 - coef) * val1
