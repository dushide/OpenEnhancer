import numpy as np
import torch
from sklearn import metrics
def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    F1_macro = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_micro = metrics.f1_score(labels_true, labels_pred, average='micro')

    return ACC, F1_macro, F1_micro


def CCR_at_FPR(FPR, CCR, fpr_values=[0.001,0.005,0.01,0.05,0.1,0.5,1]):
    """Computes CCR values for the desired FPR values, if such FPR values can be reached."""

    # compute FPR and CCR values from scores

    ccr_values = []
    zero = torch.zeros(FPR.shape)
    for desired_fpr in fpr_values:
        # get the FPR value that is closest, but above the current threshold
        candidates = torch.nonzero(FPR <= desired_fpr)[:, 0]
        if len(candidates) > 0:
            # there are values above threshold
            ccr_values.append(round(CCR[candidates[-1]].cpu().detach().numpy() * 100, 2))
        else:
            # the desired FPR cannot be reached
            ccr_values.append(None)

    return fpr_values, ccr_values


from sklearn.metrics import  roc_auc_score
def compute_openauc(x1, x2, pred, labels):
    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2, correct = x1.tolist(), x2.tolist(), (pred == labels).tolist()
    m_x2 = max(x2) + 1e-5
    y_score = [value if hit else m_x2 for value, hit in zip(x1, correct)] + x2
    y_true = [0] * len(x1) + [1] * len(x2)
    open_auc = roc_auc_score(y_true, y_score)
    print('OpenAUC:', open_auc)
    return open_auc

def prepare_openauc(scores, targets, unknown_label):
    """
    Prepare inputs for compute_openauc.
    """
    # Step 1: 预测标签（每行最大得分的索引）
    pred_labels = torch.argmax(scores, dim=1)

    # Step 2: 得分中最大值作为 open set score
    max_scores, _ = torch.max(scores, dim=1)

    # Step 3: 区分 known / unknown 样本
    is_known = targets != unknown_label
    is_unknown = targets == unknown_label

    # Step 4: open set score 定义为 -max_score，越小越“未知”
    open_set_scores = -max_scores

    # 分别获取 x1, x2
    x1 = open_set_scores[is_known]
    x2 = open_set_scores[is_unknown]
    pred = pred_labels[is_known]
    labels = targets[is_known]
    openauc = compute_openauc(x1, x2, pred, labels)
    return openauc



def _known_target_scores(scores, targets):
    """Splits off the known part of the scores, i.e., where targets are positive. It returns:

    1. The complete set of scores (one for each class) that belong to the known samples
    2. The target classes for those scores
    3. The scores for the target classes only
    """
    known_indexes = targets >= 0
    known_scores = scores[known_indexes]
    known_targets = targets[known_indexes]
    known_target_scores = known_scores[range(known_scores.shape[0]), known_targets]
    return known_scores, known_targets, known_target_scores


def _known_maximum_scores(scores, targets):
    """Splits off the known part of the scores, i.e., where targets are positive. It returns:

    1. The complete set of scores (one for each class) that belong to the known samples
    2. The maximum score for each sample, maximized over all classes
    """
    known_indexes = targets >= 0
    known_scores = scores[known_indexes]
    known_maximum_scores = torch.max(known_scores, axis=1)[0]
    return known_scores, known_maximum_scores


def _unknown_maximum_scores(scores, targets, unknown_label):
    """Splits off the unknown part of the scores, i.e., where targets have the given unknown_label. It returns:

    1. The complete set of scores (one for each class) that belong to the unknown samples
    2. The maximum score for each sample, maximized over all classes
    """
    unknown_indexes = targets == unknown_label
    unknown_scores = scores[unknown_indexes]
    unknown_maximum_scores = torch.max(unknown_scores, axis=1)[0]
    return unknown_scores, unknown_maximum_scores


def OSCR(scores, targets, unknown_label):
    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (FPR, CCR) for every score.

    :param scores: Array of scores per sample, one for each known class. Shape: (n_samples, n_classes)
    :type scores: :py:class:`torch.Tensor`:

    :param targets: Array of class indexes per sample. Shape: (n_samples)
    :type targets: :py:class:`torch.Tensor`

    :param unknown_label: Label of unknown classes, to separate known and unknown
    :type unknown_label: int, optional

    :return: Tuple of FPR (False Positive Rates) and CCR (Correct Classification Rates) values for each score threshold
    :rtype: Tuple[:py:class:`torch.Tensor`, :py:class:`torch.Tensor`]
    """
    # extract information from known samples
    # targets, scores = device(targets), device(scores)
    known_scores, known_targets, known_target_scores = _known_target_scores(scores, targets)

    # get predicted class for known samples
    predicted_classes = torch.argmax(known_scores, axis=1)
    correctly_predicted = predicted_classes == known_targets

    # extract information from unknown samples
    unknown_scores, unknown_maximum_scores = _unknown_maximum_scores(scores, targets, unknown_label)

    if len(unknown_maximum_scores) == 0:
        raise ValueError(f"The given set of targets does not contain any sample with label '{unknown_label}'")

    # Any unknown score can be a threshold
    thresholds = torch.flip(torch.unique(unknown_maximum_scores), dims=(0,))

    # compute FPR
    expanded_scores = unknown_maximum_scores[None].expand(len(thresholds), -1)
    expanded_thresholds = thresholds[:, None].expand(-1, len(unknown_scores))
    FPR = (expanded_scores >= expanded_thresholds).sum(axis=1) / len(unknown_scores)

    # compute CCR
    expanded_scores = known_target_scores[None].expand(len(thresholds), -1)
    expanded_prediction = correctly_predicted.expand(len(thresholds), -1)
    expanded_thresholds = thresholds[:, None].expand(-1, len(known_scores))

    CCR = torch.logical_and(
        expanded_scores >= expanded_thresholds,
        expanded_prediction
    ).sum(axis=1) / len(known_scores)

    return FPR, CCR


def OSA(scores, targets, unknown_label, alpha='auto'):
    """ Calculates the OSA values, iterating over the score of the target class of every sample,
    produces a pair (URR, OSA) for every score.

    :param scores: Array of scores per sample, one for each known class. Shape: (n_samples, n_classes)
    :type scores: :py:class:`torch.Tensor`:

    :param targets: Array of class indexes per sample. Shape: (n_samples)
    :type targets: :py:class:`torch.Tensor`

    :param alpha: Trade-off parameter between CCR and URR. Can be either:
                 - 'auto': Automatically calculated based on ratio of known to total samples (default)
                 - float: Manual value between 0 and 1, where lower values emphasize unknown rejection
                   and higher values emphasize correct classification
                        1- alpha = 0.67: Prioritizes correct classification
                        2- alpha = 0.50: Equal weight between known and unknown
                        3- alpha = 0.29: Emphasis on unknown rejection
                        4- alpha = 0.20: High emphasis on unknown rejection
                        5- alpha = 0.17: Strongest emphasis on unknown
    :type alpha: Union[str, float]

    :param unknown_label: Label of unknown classes, to separate known and unknown
    :type unknown_label: int, optional

    :return: Tuple of URR (Unknown Rejection Rates) and OSA values
    :rtype: Tuple[:py:class:`torch.Tensor`, :py:class:`torch.Tensor`]
    """

    # targets, scores = device(targets), device(scores)
    # extract information from known samples
    known_scores, known_targets, known_target_scores = _known_target_scores(scores, targets)

    # extract information from unknown samples
    unknown_scores, unknown_maximum_scores = _unknown_maximum_scores(scores, targets, unknown_label)

    # extract information from known samples
    known_count = len(known_scores)
    unknown_count = len(unknown_scores)

    # calculate automatic alpha if requested
    if alpha == 'auto':
        alpha = known_count / (known_count + unknown_count)

    # compute OSCR
    FPR, CCR = OSCR(scores, targets, unknown_label)
    URR = 1 - FPR

    # compute OSA
    OSA = alpha * CCR + (1 - alpha) * URR

    return URR, OSA


def OOSA(scores, targets, unknown_label, threshold=None, alpha='auto'):
    """
    Calculates the Operational Open-Set Accuracy (OOSA) at a given threshold or finds the optimal threshold.

    Parameters:
    -----------
    scores : torch.Tensor
        Array of scores per sample, one for each known class. Shape: (n_samples, n_classes)
    targets : torch.Tensor
        Array of class indexes per sample. Shape: (n_samples)
    threshold : float, optional
        Operational threshold for classification. If None, finds optimal threshold.
    alpha : Union[str, float]
        Trade-off parameter between CCR and URR. Can be either:
        - 'auto': Automatically calculated based on ratio of known to total samples
        - float: Manual value between 0 and 1
    unknown_label : int, optional
        Label used to identify unknown classes


    Returns:
    --------
    tuple: (oosa_value, threshold)
        The OOSA value and the threshold used (either provided or optimal)
    """

    # Get URR and OSA curves using existing implementation
    URR, OSA_Val = OSA(scores, targets, alpha=alpha, unknown_label=unknown_label)

    # Get unknown maximum scores to match OSCR's threshold calculation
    unknown_scores, unknown_maximum_scores = _unknown_maximum_scores(scores, targets, unknown_label)

    # Get thresholds in same order as OSCR
    thresholds = torch.flip(torch.unique(unknown_maximum_scores), dims=(0,))

    if threshold is None:
        # Find the threshold that maximizes OSA
        max_idx = torch.argmax(OSA_Val)
        return URR, OSA_Val, OSA_Val[max_idx].cpu().detach().numpy(), thresholds[max_idx].cpu().detach().numpy()
    else:
        # Convert threshold to tensor if it's not already
        threshold_tensor = threshold if isinstance(threshold, torch.Tensor) else torch.tensor(threshold)

        # Find the closest threshold value that's less than or equal to our target
        valid_thresholds = thresholds[thresholds <= threshold_tensor]
        if len(valid_thresholds) == 0:
            # If no valid threshold found, use the smallest available threshold
            idx = len(thresholds) - 1
        else:
            # Get the index of the largest valid threshold
            idx = len(thresholds) - len(valid_thresholds)

        return URR, OSA_Val ,OSA_Val[idx].cpu().detach().numpy(), thresholds[idx].cpu().detach().numpy()


def CCR_at_FPR(scores, targets, unknown_label, fpr_values=[0.001,0.005,0.01,0.05,0.1,0.5,1]):
    """Computes CCR values for the desired FPR values, if such FPR values can be reached."""

    # compute FPR and CCR values from scores
    FPR, CCR = OSCR(scores, targets, unknown_label)

    ccr_values = []
    zero = torch.zeros(FPR.shape)
    for desired_fpr in fpr_values:
        # get the FPR value that is closest, but above the current threshold
        candidates = torch.nonzero(FPR <= desired_fpr)[:, 0]
        if len(candidates) > 0:
            # there are values above threshold
            ccr_values.append(np.round(CCR[candidates[-1]].item()*100,2))
        else:
            # the desired FPR cannot be reached
            ccr_values.append(None)

    return  ccr_values


# def AUOSCR(scores, targets, unknown_label=dataset.Dataset.UNKNOWN_LABEL, min_fpr_value=1e-4, log_scale=False):
def AUOSCR(scores, targets, unknown_label, min_fpr_value=1e-4, log_scale=False):
    """Computes the Area Under the OSCR curve.

    This implementation uses the trapezoidal rule to compute the area under the OSCR curve.
    Since the OSCR might not be defined for low FPR values, and especially FPR=0 is typically not reached, we define a minimum FPR score here, from which we start the curve.
    If FPR values smaller than this ``min_fpr_value`` are provided, they will be ignored.
    If no such FPR can be found, we assume ``CCR@min_fpr_value = 0``.

    OSCR curves are typically computed in logarithmic scale.
    If ``log_scale`` is enabled, this function computed the area under the OSCR in log scale (using the basis of 10 via :py:func:`torch.log10`) as well, i.e., putting more emphasis on low FPR values.
    This is normalized by the logarithm of the ``min_fpr_value``.
    Otherwise, the linear AUC is computed, putting more emphasis on high FPR values and, therewith, closed-set accuracy.

    The equations for the trapezoidal rule are given as follows:

    .. math::

       \\frac1{2(1-FPR_0)} \\sum_{i=1}^{N-1} \\left(CCR_{i+1} + CCR_{i}\\right) \\left(FPR_{i+1} - FPR_{i}\\right)

        -\\frac1{2\\log FPR_1} \\sum_{i=1}^{N-1} \\left(CCR_{i+1} + CCR_{i}\\right) \\left(\\log FPR_{i+1} - \\log FPR_{i}\\right)
    """

    # first, compute CCR and FPR values
    FPR, CCR = OSCR(scores, targets, unknown_label)
    # fpr, ccr = EvaluationMetrics.calculate_oscr(targets.cpu().numpy(),scores.cpu().numpy(),unknown_label)
    # now remove all scores that are below FPR of min_fpr_value
    smaller_values = torch.sum(FPR <= min_fpr_value)
    if smaller_values > 0:
        # the FPR at the min_FPR is taken from the supremum sample that will be reomved
        min_ccr_value = CCR[smaller_values - 1]
        # remove all FPR and CCR values that are smaller
        FPR = FPR[smaller_values:]
        CCR = CCR[smaller_values:]
    else:
        min_ccr_value = 0.

    # pre-pend minimum value
    FPR = torch.cat((torch.tensor([min_fpr_value]).to(FPR.device), FPR))
    CCR = torch.cat((torch.tensor([min_ccr_value]).to(FPR.device), CCR))

    # enable log scale
    if log_scale:
        FPR = torch.log10(FPR)
        factor = -0.5 / torch.log10(torch.tensor(min_fpr_value))
    else:
        factor = 0.5 / (1. - min_fpr_value)

    # compute trapezoidal rule
    first_indexes = torch.arange(len(FPR) - 1, dtype=torch.int64)
    second_indexes = first_indexes + 1
    AUOSCR = torch.sum(
        (CCR[second_indexes] + CCR[first_indexes])
        *
        (FPR[second_indexes] - FPR[first_indexes])
    )

    return FPR, CCR, factor * AUOSCR.cpu().detach().numpy()

