#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, string
from collections import Counter, defaultdict
from pathlib import Path

PUNCT_TABLE = str.maketrans("", "", string.punctuation)

YES_ALIASES = {"yes","y","yeah","yep","true","correct","affirmative"}
NO_ALIASES  = {"no","n","nope","false","incorrect","negative"}

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # remove "Assistant:" or role prefixes if present
    s = re.sub(r"^\s*(assistant\s*:)\s*", "", s, flags=re.I)
    # lower, strip punctuation, collapse spaces
    s = s.lower().translate(PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    s = normalize_text(s)
    return s.split()

def word_in_text(needle: str, hay: str) -> bool:
    """Check if normalized 'needle' appears as a whole-word span in normalized 'hay'."""
    n = normalize_text(needle)
    h = normalize_text(hay)
    # fast path: exact
    if n == h:
        return True
    # whole-word span search
    return re.search(rf"\b{re.escape(n)}\b", h) is not None

def closed_is_correct(pred_text: str, gold_answer: str, candidates=None) -> bool:
    p = normalize_text(pred_text)
    g = normalize_text(gold_answer)

    # yes/no robustness
    if g in YES_ALIASES:
        return any(tok in YES_ALIASES for tok in p.split())
    if g in NO_ALIASES:
        return any(tok in NO_ALIASES for tok in p.split())

    # if candidates given, you can optionally be stricter:
    # pick any candidate that appears; but your spec says:
    # "if the answer is in the output, then correct"
    # so we just check the gold answer span.
    return word_in_text(g, p)

def recall_score(pred_text: str, gold_answer: str) -> float:
    """Token recall: fraction of gold tokens present in prediction."""
    gt_tokens = tokenize(gold_answer)
    if not gt_tokens:
        return 0.0
    pred_tokens = set(tokenize(pred_text))
    hit = sum(1 for t in gt_tokens if t in pred_tokens)
    return hit / len(gt_tokens)

def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            items.append(json.loads(line))
    return items

def build_gt_map(test_jsonl_path: str):
    """Return:
       - gt_map: id(str) -> dict(gold_answer, answer_type, candidates(optional))
       - counts: simple totals per type
    """
    gt_map = {}
    counts = defaultdict(int)
    data = load_jsonl(test_jsonl_path)
    for ex in data:
        # id can be "id" or "question_id" in some sets; normalize to str
        qid = str(ex.get("id", ex.get("question_id")))
        # prefer explicit "answer" if present, fall back to conversations last gpt
        if "answer" in ex and ex["answer"] is not None:
            gold = ex["answer"]
        else:
            gold = ""
            if "conversations" in ex:
                # find the assistant/gpt turn
                for turn in ex["conversations"][::-1]:
                    if turn.get("from", "").lower() in {"gpt","assistant"}:
                        gold = turn.get("value","")
                        break
        atype = ex.get("answer_type", "").upper()  # "OPEN" / "CLOSED"
        cands = ex.get("candidates", None)
        gt_map[qid] = {
            "gold": gold,
            "answer_type": atype,
            "candidates": cands
        }
        if atype in {"OPEN","CLOSED"}:
            counts[atype] += 1
    return gt_map, counts

def evaluate(pred_jsonl_path: str, test_jsonl_path: str, dump_wrong: str=None):
    preds = load_jsonl(pred_jsonl_path)
    gt_map, counts = build_gt_map(test_jsonl_path)

    closed_total = 0
    closed_correct = 0
    open_total = 0
    open_recall_sum = 0.0

    wrong_closed = []
    per_open = []  # keep per-sample open recall (qid, recall)

    for pred in preds:
        qid = str(pred.get("question_id", pred.get("id")))
        ptxt = pred.get("text", "")

        if qid not in gt_map:
            # prediction for an id not in test set; skip
            continue

        gold = gt_map[qid]["gold"]
        atype = gt_map[qid]["answer_type"]
        cands = gt_map[qid]["candidates"]

        if atype == "CLOSED":
            closed_total += 1
            ok = closed_is_correct(ptxt, gold, candidates=cands)
            if ok:
                closed_correct += 1
            else:
                wrong_closed.append({
                    "question_id": qid,
                    "gold": gold,
                    "pred": ptxt,
                    "candidates": cands
                })
        elif atype == "OPEN":
            open_total += 1
            r = recall_score(ptxt, gold)
            open_recall_sum += r
            per_open.append((qid, r, gold, ptxt))

    closed_acc = (closed_correct / closed_total * 100.0) if closed_total > 0 else 0.0
    open_recall = (open_recall_sum / open_total * 100.0) if open_total > 0 else 0.0

    # Print summary
    print("\n================== Metrics ==================")
    print(f"Closed-set: total={closed_total}  correct={closed_correct}  ACC={closed_acc:.2f}%")
    print(f"Open-set  : total={open_total}  avg token recall={open_recall:.2f}%")
    print("=============================================\n")

    # Show a few wrong closed cases
    if wrong_closed:
        print("Some incorrect CLOSED predictions:")
        for ex in wrong_closed[:20]:
            print(f"- QID {ex['question_id']} | GOLD: {ex['gold']} | PRED: {ex['pred']}")
        print(f"... ({len(wrong_closed)} total wrong closed cases)\n")

    # Optionally dump details
    if dump_wrong:
        payload = {
            "closed": {
                "total": closed_total,
                "correct": closed_correct,
                "accuracy": closed_acc,
                "wrong_examples": wrong_closed
            },
            "open": {
                "total": open_total,
                "avg_recall": open_recall,
                "per_sample": [
                    {"question_id": q, "recall": r, "gold": g, "pred": p}
                    for (q, r, g, p) in per_open
                ]
            }
        }
        Path(dump_wrong).parent.mkdir(parents=True, exist_ok=True)
        with open(dump_wrong, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed report to: {dump_wrong}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Predictions JSONL (with question_id, text)")
    ap.add_argument("--test", required=True, help="Ground-truth JSONL (with id/answer/answer_type etc.)")
    ap.add_argument("--report", default=None, help="Optional path to save a detailed JSON report")
    args = ap.parse_args()
    evaluate(args.pred, args.test, args.report)

if __name__ == "__main__":
    main()