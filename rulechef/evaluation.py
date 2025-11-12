"""NER span extraction evaluation metrics"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Span:
    text: str
    start: int
    end: int


def span_iou(span1: Dict, span2: Dict) -> float:
    """Calculate Intersection over Union for two spans"""
    s1_start = span1.get("start", 0)
    s1_end = span1.get("end", 0)
    s2_start = span2.get("start", 0)
    s2_end = span2.get("end", 0)

    # Calculate intersection
    inter_start = max(s1_start, s2_start)
    inter_end = min(s1_end, s2_end)
    intersection = max(0, inter_end - inter_start)

    # Calculate union
    union = (s1_end - s1_start) + (s2_end - s2_start) - intersection

    if union == 0:
        return 0.0

    return intersection / union


def boundary_distance(pred_span: Dict, gold_span: Dict) -> int:
    """Calculate average boundary error distance"""
    start_error = abs(pred_span.get("start", 0) - gold_span.get("start", 0))
    end_error = abs(pred_span.get("end", 0) - gold_span.get("end", 0))
    return (start_error + end_error) // 2


def find_best_match(
    pred_span: Dict, gold_spans: List[Dict], iou_threshold: float = 0.5
) -> Tuple[int, float]:
    """
    Find best matching gold span for a predicted span.
    Returns (index, iou) or (-1, 0.0) if no match above threshold.
    """
    best_idx = -1
    best_iou = 0.0

    for idx, gold_span in enumerate(gold_spans):
        iou = span_iou(pred_span, gold_span)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_iou >= iou_threshold:
        return best_idx, best_iou
    return -1, 0.0


def evaluate_spans(
    predictions: List[Dict],
    gold_standard: List[Dict],
    exact_match_only: bool = False,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Evaluate predicted spans against gold standard.

    Args:
        predictions: List of predicted spans [{"text": str, "start": int, "end": int}, ...]
        gold_standard: List of gold spans with same format
        exact_match_only: If True, only exact matches (same text) count
        iou_threshold: IoU threshold for partial match (default 0.5)

    Returns:
        Dictionary with metrics:
        - exact_matches: Count of exact boundary matches
        - partial_matches: Count of IoU-based matches
        - false_positives: Predicted but not in gold
        - false_negatives: In gold but not predicted
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1: 2 * (precision * recall) / (precision + recall)
        - boundary_errors: List of (pred, gold, distance) for error analysis
    """

    if not gold_standard:
        return {
            "exact_matches": 0,
            "partial_matches": 0,
            "false_positives": len(predictions),
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy_exact": 0.0,
            "accuracy_partial": 0.0,
            "boundary_errors": [],
        }

    matched_gold = set()
    exact_matches = 0
    partial_matches = 0
    boundary_errors = []

    for pred in predictions:
        found_exact = False
        found_partial = False

        for gold_idx, gold in enumerate(gold_standard):
            if gold_idx in matched_gold:
                continue

            # Check for exact match (same text and position)
            if (
                pred.get("text") == gold.get("text")
                and pred.get("start") == gold.get("start")
                and pred.get("end") == gold.get("end")
            ):
                exact_matches += 1
                matched_gold.add(gold_idx)
                found_exact = True
                found_partial = True
                break

        # If no exact match, check for partial match
        if not found_exact and not exact_match_only:
            best_idx, iou = find_best_match(pred, gold_standard, iou_threshold)
            if best_idx != -1 and best_idx not in matched_gold:
                partial_matches += 1
                matched_gold.add(best_idx)
                found_partial = True
                distance = boundary_distance(pred, gold_standard[best_idx])
                boundary_errors.append(
                    {
                        "predicted": pred,
                        "gold": gold_standard[best_idx],
                        "distance": distance,
                        "iou": iou,
                    }
                )

    true_positives = exact_matches + partial_matches
    false_positives = len(predictions) - true_positives
    false_negatives = len(gold_standard) - len(matched_gold)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_exact": exact_matches / len(gold_standard) if gold_standard else 0.0,
        "accuracy_partial": true_positives / len(gold_standard)
        if gold_standard
        else 0.0,
        "boundary_errors": boundary_errors,
    }


def print_eval_report(metrics: Dict, dataset_name: str = "Dataset"):
    """Pretty-print evaluation metrics"""
    print(f"\n{'=' * 70}")
    print(f"EVALUATION REPORT: {dataset_name}")
    print(f"{'=' * 70}\n")

    print("SPAN EXTRACTION METRICS")
    print("-" * 70)
    print(f"Exact Match Accuracy:    {metrics['accuracy_exact']:.1%}")
    print(f"Partial Match Accuracy:  {metrics['accuracy_partial']:.1%} (IoU > 0.5)")
    print()

    print("DETAILED METRICS")
    print("-" * 70)
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1']:.1%}")
    print()

    print("CONFUSION")
    print("-" * 70)
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print()

    if metrics.get("boundary_errors"):
        print("BOUNDARY ERRORS (Sample)")
        print("-" * 70)
        for error in metrics["boundary_errors"][:5]:
            pred = error["predicted"]
            gold = error["gold"]
            dist = error["distance"]
            print(f"Predicted: '{pred['text']}' [{pred['start']}:{pred['end']}]")
            print(f"Gold:      '{gold['text']}' [{gold['start']}:{gold['end']}]")
            print(f"Boundary offset: ±{dist} chars, IoU: {error['iou']:.2f}")
            print()

    print(f"{'=' * 70}\n")
