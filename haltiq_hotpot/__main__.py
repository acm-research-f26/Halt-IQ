"""Prepare or run the small shared-draft method comparison, entirely offline."""
import argparse
import json
from pathlib import Path

from pc_strands.control import atomic_json

from .protocol import digest, prepare_cohort
from .runner import REPO, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--partner-csv", type=Path, required=True)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--cohort", type=Path, required=True)
            command.add_argument("--cohort-sha256", required=True)
            command.add_argument("--checkpoint-dir", type=Path, required=True)
            command.add_argument("--base-dir", type=Path, required=True)
            command.add_argument("--reranker-dir", type=Path, required=True)
            command.add_argument("--current-checkpoint", type=Path)
            command.add_argument("--lease-path", type=Path, required=True)
            command.add_argument("--deadline")
            command.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        output = args.output.resolve()
        if output == REPO or REPO in output.parents:
            raise ValueError("Private cohort/evaluator data belongs outside the source repository")
        cohort = prepare_cohort(args.partner_csv, args.dataset)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "cohort.json", cohort)
        result = {"scope": cohort["scope"], "questions": 20, "model_calls": 0,
            "selected_type_counts": {"bridge": 10, "comparison": 10},
            "source_pool_counts": cohort["selection"]["source_pool_counts"],
            "cohort_sha256": digest(output / "cohort.json")}
        atomic_json(output / "prepared.json", result)
    else:
        receipt = run(args)
        result = {key: receipt[key] for key in ("scope", "status")}
        if args.allow_model_calls:
            result.update({key: receipt[key] for key in ("physical_encoder_forwards", "physical_reranker_forwards", "metrics")})
        else:
            result.update(receipt)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
