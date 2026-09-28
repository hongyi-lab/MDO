"""Frozen-feature ridge with train-only transforms and nested leave-one-out CV."""
import numpy as np

DEFAULT_ALPHAS = np.logspace(-6, 6, 13)


def fit_ridge(x, y, alpha):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Expected paired 2D matrices with at least two rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Finite data and positive ridge alpha required")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    ym, ys = y.mean(0), y.std(0)
    ys = np.where(ys > 1e-12, ys, 1.)
    u, s, vt = np.linalg.svd((x-mean)/scale, full_matrices=False)
    weight = (vt.T * (s/(s*s+alpha))) @ (u.T @ ((y-ym)/ys))
    return dict(mean=mean, scale=scale, target_mean=ym, target_scale=ys,
                weight=weight, alpha=float(alpha))


def predict_ridge(fit, x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != len(fit['mean']) or not np.isfinite(x).all():
        raise ValueError("Invalid ridge features")
    return (((x-fit['mean'])/fit['scale']) @ fit['weight']) * fit['target_scale'] + fit['target_mean']


def select_ridge(x_train, y_train, alphas=DEFAULT_ALPHAS):
    """Only accepts training arrays: eval labels cannot influence lambda/scales."""
    x, y = np.asarray(x_train, float), np.asarray(y_train, float)
    if len(x) < 4 or len(x) != len(y):
        raise ValueError("LOOCV requires at least four paired training rows")
    scores = []
    score_scale = np.where(y.std(0) > 1e-12, y.std(0), 1.)
    for alpha in alphas:
        predicted = []
        for j in range(len(x)):
            keep = np.arange(len(x)) != j
            fitted = fit_ridge(x[keep], y[keep], float(alpha))
            predicted.append(predict_ridge(fitted, x[j:j+1])[0])
        scores.append(float(np.mean(((np.asarray(predicted)-y)/score_scale)**2)))
    best = int(np.argmin(scores))
    fitted = fit_ridge(x, y, float(alphas[best]))
    return fitted, {"alphas": np.asarray(alphas).tolist(), "train_LOOCV_standardized_MSE": scores,
                    "selected_alpha": float(alphas[best]), "selection_uses_eval_labels": False}
