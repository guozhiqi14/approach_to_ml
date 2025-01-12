import numpy as np
from scipy import stats


def mean_predictions(probas):
    """
    Create mean predictions
    :param probas: 2-d array of probability values :return: mean probability
    """
    return np.mean(probas, axis=1)


def max_voting(preds):
    """
    Create mean predictions
    :param probas: 2-d array of prediction values
    :return: max voted predictions
    """
    idxs = np.argmax(preds, axis=1)
    return np.take_along_axis(preds, idxs[:, None], axis=1)


def rank_mean(probas):
    """
    Create mean predictions using ranks
    :param probas: 2-d array of probability values
    :return: mean ranks
    """
    ranked = []
    for i in range(probas.shape[1]):
        rank_data = stats.rankdata(probas[:, i])
        ranked.append(rank_data)
    ranked = np.column_stack(ranked)
    return np.mean(ranked, axis=1)
