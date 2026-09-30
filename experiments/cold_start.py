from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.adapters.llm_adapter import OpenAICompatibleLLM
from evotx.core.rule import generate_cold_start_rule
from evotx.storage.rule_store import RuleStore
from evotx.utils.logging_utils import configure_session_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: initialize an EvoTx cold-start rule from attack-family label/description."
    )
    parser.add_argument("--attack-description")
    parser.add_argument(
        "--label",
        help="Attack-family label, e.g. reentrancy or price_manipulation.",
    )
    parser.add_argument(
        "--human-example",
        default="",
        help="Optional sketch/example from a human. The model uses it only as orientation aid.",
    )
    parser.add_argument("--human-example-file")
    parser.add_argument("--rules-dir", default="data/rules")
    parser.add_argument("--rule-id")
    parser.add_argument("--rule-name")
    parser.add_argument("--model")
    parser.add_argument(
        "--llm-provider",
        help="Optional provider override: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument("--logs-dir", default="data/log")
    args = parser.parse_args()
    log_path = configure_session_logging(
        "cold_start",
        logs_dir=args.logs_dir,
        suffix=args.label or args.rule_id or "rule_init",
    )

    attack_description = resolve_attack_description(
        attack_description=args.attack_description,
        label=args.label,
    )
    print(
        f"[ColdStart] attack_label={args.label or '(none)'} "
        f"attack_description={attack_description}"
    )

    human_example = load_human_example(args.human_example, args.human_example_file)
    llm = (
        OpenAICompatibleLLM(
            model=args.model,
            provider=args.llm_provider,
            minimax_thinking="adaptive",
        )
        if args.model
        else None
    )
    print(
        f"[ColdStart] LLM={'enabled' if llm else 'disabled'} "
        f"model={args.model or '(fallback template)'} "
        f"provider={args.llm_provider or '(auto)'}"
    )

    rule = generate_cold_start_rule(
        attack_description=attack_description,
        llm=llm,
        human_example=human_example,
        attack_label=args.label,
        rule_id=args.rule_id,
        rule_name=args.rule_name,
    )
    path = RuleStore(args.rules_dir).save(
        rule,
        attack_label=args.label or str((rule.metadata or {}).get("attack_label", "")),
    )

    print("Stage 1 complete.")
    print(f"Saved cold-start rule v{rule.version}: {path}")
    print(f"Attack label: {rule.metadata.get('attack_label', 'attack')}")
    print(f"Rule source: {rule.metadata.get('source', 'unknown')}")
    print(f"Session log: {log_path}")


def load_human_example(inline_text: str, file_path: str | None) -> str:
    if inline_text.strip():
        return inline_text.strip()
    if not file_path:
        return ""
    return Path(file_path).read_text(encoding="utf-8").strip()


def resolve_attack_description(
    attack_description: str | None,
    label: str | None,
) -> str:
    if attack_description and attack_description.strip():
        return attack_description.strip()
    if label and label.strip():
        return label.strip().replace("_", " ").replace("-", " ")
    raise ValueError("Provide at least one of --attack-description or --label.")


if __name__ == "__main__":
    main()
