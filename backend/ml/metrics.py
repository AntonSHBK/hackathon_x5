from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)


def compute_metrics(eval_pred, idx2label):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    true_labels = []
    true_predictions = []

    for pred_seq, label_seq in zip(predictions, labels):
        seq_true = []
        seq_pred = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_true.append(idx2label[l])
            seq_pred.append(idx2label[p])
        true_labels.append(seq_true)
        true_predictions.append(seq_pred)

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1_micro = f1_score(true_labels, true_predictions, average="micro")
    f1_macro = f1_score(true_labels, true_predictions, average="macro")
    accuracy = accuracy_score(true_labels, true_predictions)

    report = classification_report(true_labels, true_predictions, digits=4)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        # "report": report,
    }
    return metrics