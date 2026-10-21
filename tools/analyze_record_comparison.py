"""Print useful positive-answer evidence separately from negative abstentions."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    print(report["scope"])
    print("status:", report["status"])
    print("Positive checked answers are primary; task_exact includes negative abstentions.")
    print("seed arm split positive_answers/positive_worlds task_exact invalid_operation_rate")
    for paired in report["seeds"]:
        for arm, result in paired["arms"].items():
            for split, metrics in result["metrics"].items():
                print(paired["seed"], arm, split,
                      f"{metrics['supported_answers_correct']}/{metrics['supported_worlds']}",
                      metrics["task_exact"], metrics["invalid_operation_rate"])
            print("resources", paired["seed"], arm, "train_seconds", result["training_wall_seconds"],
                  "eval_seconds", result["evaluation_wall_seconds"], "train_input_tokens", result["training_input_tokens"],
                  "peak_cuda_allocated_bytes", result["peak_cuda_allocated_bytes"])
    print("external_fit_gate:", report.get("external_fit_gate", "Not declared for this enablement run"))
    print("Always-abstain and exact symbolic planner controls:", json.dumps(report["controls"], sort_keys=True))
    print("Encoding/active-capacity/token-work differences prevent architecture superiority claims.")


if __name__ == "__main__":
    main()
