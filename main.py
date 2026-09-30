from __future__ import annotations


def main() -> None:
    print("EvoTx scaffold is ready.")
    print("Suggested stage entrypoints:")
    print("  1. python experiments/cold_start.py --label reentrancy --tool-manifest configs/tool_manifest.json --model glm-5.1")
    print("  2. python experiments/train_fewshot.py --benign-csv benign.csv --malicious-csv Reentrancy.csv --label reentrancy --tool-manifest configs/tool_manifest.json --llm-model glm-5.1")
    print("  2.5 python experiments/run_inference.py --tx-hash 0x... --chain eth --label reentrancy --tool-manifest configs/tool_manifest.json --use-environment")
    print("  3. python experiments/evaluate.py --benign-csv benign.csv --malicious-csv Reentrancy.csv --label reentrancy --rule-file data/rules/<rule>__latest.json --tool-manifest configs/tool_manifest.json")
    print("  4. python experiments/sketch_guided_calibration.py --pool-file data/unlabeled_pool.json --rule-file data/rules/<rule>__latest.json --tool-manifest configs/tool_manifest.json")
    print("Use --use-environment on inference/evolution/evaluation to instantiate TransactionEnvironment.")


if __name__ == "__main__":
    main()
