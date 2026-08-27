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
                '"category": "Cognitive Decline' in output
                or '"Cognitive Decline' in output
                or "{'category': 'Cognitive Decline" in output
            ):
                pred = "Cognitive Decline"
            elif (
                '"category": "Cognitive Normal' in output
                or '"Cognitive Normal' in output
                or "{'category': 'Cognitive Normal" in output
            ):
                pred = "Cognitive Normal"
            else:
                print(f"\nTRUE LABEL: {true_label}")
                print(f"Filename: {entry.get('filename', 'Unknown')}")
                print(f"Output: {output}")
                answer = input("Enter 1 if correct, 0 if incorrect: ")
                if answer == "1":
                    pred = true_label
                else:
                    pred = "Cognitive Decline" if true_label == "Cognitive Normal" else "Cognitive Normal"
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
    # Positive class = Cognitive Decline
    tp = tn = fp = fn = 0
    for true_label, pred in results:
        if true_label == "Cognitive Decline" and pred == "Cognitive Decline":
            tp += 1
        elif true_label == "Cognitive Normal" and pred == "Cognitive Normal":
            tn += 1
        elif true_label == "Cognitive Normal" and pred == "Cognitive Decline":
            fp += 1
        elif true_label == "Cognitive Decline" and pred == "Cognitive Normal":
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
    p = argparse.ArgumentParser(description="Cognitive Decline classification metrics from two JSONL files.")
    p.add_argument("--cn_file", required=True)
    p.add_argument("--cd_file", required=True)
    p.add_argument("--force_extract", action="store_true",
                   help="Re-extract labels even if already cached in the file.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cn_results = process_file(args.cn_file, "Cognitive Normal", force_extract=args.force_extract)
    cd_results = process_file(args.cd_file, "Cognitive Decline", force_extract=args.force_extract)
    m = calculate_metrics(cn_results + cd_results)

    print(f"Accuracy    {m['accuracy']:.4f}  ({m['tp'] + m['tn']}/{m['total']})")
    print(f"Precision   {m['precision']:.4f}")
    print(f"Recall      {m['recall']:.4f}")
    print(f"Specificity {m['specificity']:.4f}")
    print(f"F1          {m['f1_score']:.4f}")
    print(f"TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
