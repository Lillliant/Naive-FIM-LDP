# Calculate the precision of the frequent itemset mining algorithms
# Formula: Correct Positive predictions / All Positive Predictions
# Definition of Correct Positive prediction: same frequent itemse OR same frequent itemset AND same support
def calculate_precision(fim_true: list[tuple[int], int], fim_pred: list[tuple[int], int], s: bool=False) -> float:
    # assumes fims contains no duplicates
    if not s:
        fi_true = [fi for fi, _ in fim_true]
        fi_pred = [fi for fi, _ in fim_pred]
    else:
        fi_true = fim_true
        fi_pred = fim_pred
    correct_predictions = 0
    for fi in fi_pred:
        if fi in fi_true:
            correct_predictions += 1
    # Calculate precision
    precision = correct_predictions / len(fi_pred) if fi_pred else 0
    return precision

# Calculate the recall of the frequent itemset mining algorithms
# Formula: Correct Positive predictions / All True Positive Predictions
# Definition of Correct Positive prediction: same frequent itemset OR same frequent itemset AND same support
def calculate_recall(fim_true: list[tuple[int], int], fim_pred: list[tuple[int], int], s: bool=False) -> float:
    # assumes fims contains no duplicates
    if not s:
        fi_true = [fi for fi, _ in fim_true]
        fi_pred = [fi for fi, _ in fim_pred]
    else:
        fi_true = fim_true
        fi_pred = fim_pred
    correct_predictions = 0
    for fi in fi_pred:
        if fi in fi_true:
            correct_predictions += 1
    recall = correct_predictions / len(fi_true) if fi_true else 0
    return recall

# For False Positive and False Negative rates, we do not examine the support ===
def calculate_fp(fim_true: list[tuple[int], int], fim_pred: list[tuple[int], int]) -> float:
    # assumes fims contains no duplicates
    fi_true = [fi for fi, _ in fim_true]
    fi_pred = [fi for fi, _ in fim_pred]
    false_positive = 0
    for fi in fi_pred:
        if fi not in fi_true:
            false_positive += 1
    fp_rate = false_positive / len(fi_pred) if fi_pred else 0
    return fp_rate

def calculate_fn(fim_true: list[tuple[int], int], fim_pred: list[tuple[int], int]) -> float:
    # assumes fims contains no duplicates
    fi_true = [fi for fi, _ in fim_true]
    fi_pred = [fi for fi, _ in fim_pred]
    false_negative = 0
    for fi in fi_true:
        if fi not in fi_pred:
            false_negative += 1
    fn_rate = false_negative / len(fi_true) if fi_true else 0
    return fn_rate

# F1 score when only the frequent itemsets are considered, OR
# F1 score when the support is also considered
def calculate_f1(fim_true: list[tuple[int], int], fim_pred: list[tuple[int], int], s: bool=False) -> float:
    precision = calculate_precision(fim_true, fim_pred, s)
    recall = calculate_recall(fim_true, fim_pred, s)
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def average_support_deviation(fim_true: list[tuple[int], int], fim_pred: list[tuple[int], int]) -> float:
    total_support = 0
    total_fi = 0
    fi_true = [fi for fi, _ in fim_true]
    support_true = [support for _, support in fim_true]
    for fi, s_pred in fim_pred:
        if fi in fi_true:
            s_true = support_true[fi_true.index(fi)]
            total_support += abs(s_true - s_pred)
            total_fi += 1
    if total_fi == 0:
        return 0
    average_deviation = total_support / total_fi
    return average_deviation

def average_item_change_per_transaction(original_data: list[list[int]], perturbed_data: list[list[int]]) -> float:
    total_changes = 0
    for original, perturbed in zip(original_data, perturbed_data):
        changes = len(set(original) ^ set(perturbed))
        total_changes += changes
    average_change = total_changes / len(original_data) if original_data else 0
    return average_change