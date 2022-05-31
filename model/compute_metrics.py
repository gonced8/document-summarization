from datasets import load_metric


def compute_metrics(pred):
    metrics = {
        "f1": load_metric("f1"),
        "precision": load_metric("precision"),
        "recall": load_metric("recall"),
        "rouge": load_metric("rouge"),
    }

    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    results = {
        name: metric.compute(predictions=predictions, references=labels)["name"]
        for name, metric in metrics
    }

    print(results)

    return results
