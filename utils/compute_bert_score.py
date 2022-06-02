import json
import sys

from bert_score import score

if __name__ == "__main__":
    filename = sys.argv[1]

    with open(filename, "r") as f:
        data = json.load(f)

    results, *data = data

    N = 200
    cands = [sample["model_answer"] for sample in data[:N]]
    refs = [sample["truth_output"] for sample in data[:N]]

    P, R, F1 = (x.mean() for x in score(cands, refs, lang="en"))

    print(f"precision: {P}\trecall: {R}\tF1: {F1}")
