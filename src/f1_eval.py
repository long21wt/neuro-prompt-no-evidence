import argparse
import json
import os

from tqdm import tqdm


def process_file(filepath, true_label, force_extract=False):
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]

    results = []
    modified = False
    for entry in tqdm(data, desc=f"Processing {os.path.basename(filepath)}"):
        if not force_extract and "extracted_label" in entry:
            pred = entry["extracted_label"]
        else:
            output = entry["output"]
            if (
                '"category": "Major Depressive Disorder' in output
                or '"Major Depressive Disorder' in output
                or "{'category': 'Major Depressive Disorder" in output
            ):
                pred = "MDD"
            elif (
                '"category": "Control' in output
                or '"Control (no disorder detected)' in output
                or "{'category': 'Control" in output
            ):
                pred = "Control"
            else:
                print(f"\nTRUE LABEL: {true_label}")
                print(f"Filename: {entry.get('filename', 'Unknown')}")
                print(f"Output: {output}")
                answer = input("Enter 1 if correct, 0 if incorrect: ")
                if answer == "1":
                    pred = true_label
                else:
                    pred = "MDD" if true_label == "Control" else "Control"
            entry["extracted_label"] = pred
            entry["true_label"] = true_label
            modified = True

        results.append((true_label, pred))

    if modified:
        with open(filepath, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    return results


def calculate_metrics(results):
    tp = tn = fp = fn = 0
    for true_label, pred in results:
        if true_label == "MDD" and pred == "MDD":
            tp += 1
        elif true_label == "Control" and pred == "Control":
            tn += 1
        elif true_label == "Control" and pred == "MDD":
            fp += 1
        elif true_label == "MDD" and pred == "Control":
            fn += 1

    total = len(results)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
    }


def parse_args():
    p = argparse.ArgumentParser(description="MDD/Control classification metrics from two JSONL files.")
    p.add_argument("--control_file", required=True)
    p.add_argument("--mdd_file", required=True)
    p.add_argument("--force_extract", action="store_true",
                   help="Re-extract labels even if already cached in the file.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    control_results = process_file(args.control_file, "Control", force_extract=args.force_extract)
    mdd_results = process_file(args.mdd_file, "MDD", force_extract=args.force_extract)
    m = calculate_metrics(control_results + mdd_results)

    print(f"Accuracy    {m['accuracy']:.4f}  ({m['tp'] + m['tn']}/{m['total']})")
    print(f"Precision   {m['precision']:.4f}")
    print(f"Recall      {m['recall']:.4f}")
    print(f"Specificity {m['specificity']:.4f}")
    print(f"F1          {m['f1_score']:.4f}")
    print(f"TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
