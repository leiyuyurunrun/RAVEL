"""
One-shot rule + plan generation for a given attack label and description.

模式1: 指定 --label 和 --attack-description，从零生成 EvolvingRule，再生成 EvidencePlan。
模式2: 指定 --rule-file，从已有 rule JSON 文件加载，直接生成 EvidencePlan。
不进入 train_fewshot、inference 或 evaluation 流程。
"""
# isort: off
# fmt: off
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import re
from datetime import datetime

from evotx.adapters.llm_adapter import OpenAICompatibleLLM
from evotx.core.plan import compile_rule_to_baseline_plan
from evotx.core.rule import generate_cold_start_rule, rule_complexity
from evotx.core.schemas import EvolvingRule
from evotx.planner.plan_generator import PlanGenerator
from evotx.planner.plan_validator import PlanValidator, PlanValidationError
# fmt: on
# isort: on


def resolve_attack_description(attack_description: str | None, label: str | None) -> str:
    if attack_description and attack_description.strip():
        return attack_description.strip()
    if label and label.strip():
        return label.strip().replace("_", " ").replace("-", " ")
    raise ValueError(
        "Either --attack-description or --label must be provided."
    )


def run_one(
    label: str,
    attack_description: str,
    llm=None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    timestamp: str | None = None,
    out_dir: str = "data/results",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate rule (cold-start)
    print(f"[1/4] Generating rule for label='{label}' ...")
    rule = generate_cold_start_rule(
        attack_description=attack_description,
        llm=llm,
        attack_label=label,
    )
    rule_id = rule.rule_id
    rule_version = rule.version
    print(f"       Rule: {rule_id} v{rule_version}")

    # 2. Generate plan
    print(f"[2/4] Generating plan for {rule_id} ...")
    if llm is not None:
        try:
            plan = PlanGenerator(
                llm=llm, enforce_label_plan_template=False).generate(rule)
            plan_source = "llm"
            plan_error = None
        except Exception as e:
            plan = compile_rule_to_baseline_plan(rule)
            plan_source = f"baseline_after_llm_failure({e})"
            plan_error = str(e)
    else:
        plan = compile_rule_to_baseline_plan(rule)
        plan_source = "baseline"
        plan_error = None
    print(f"       Plan: {plan.plan_id} ({plan_source})")

    # 3. Validate
    print(f"[3/4] Validating plan ...")
    validation_passed = False
    validation_error = None
    try:
        PlanValidator().validate(plan)
        validation_passed = True
        print(f"       Validation: PASS")
    except PlanValidationError as ve:
        validation_error = str(ve)
        print(f"       Validation: FAIL - {ve}")

    # 4. Write outputs
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_str = (llm_provider or "none").replace("-", "_")
    # Sanitize label for use in filename
    safe_label = re.sub(r"[^\w\-_. ]", "_", label)[:80]
    file_tag = f"{safe_label}_{provider_str}_{ts}"

    rule_path = out_dir / f"{file_tag}_rule.json"
    plan_path = out_dir / f"{file_tag}_plan.json"

    with open(rule_path, "w", encoding="utf-8") as f:
        json.dump(rule.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"       Rule saved: {rule_path}")

    plan_dict = {
        **plan.to_dict(),
        "metadata": {
            **(plan.metadata or {}),
            "rule_id": rule_id,
            "rule_version": rule_version,
            "plan_source": plan_source,
            "llm_model": llm_model,
            "llm_provider": llm_provider,
            "validation_passed": validation_passed,
            "validation_error": validation_error,
        }
    }
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_dict, f, ensure_ascii=False, indent=2)
    print(f"       Plan saved: {plan_path}")

    return {
        "label": label,
        "rule_id": rule_id,
        "rule_version": rule_version,
        "rule_path": str(rule_path),
        "plan_id": plan.plan_id,
        "plan_path": str(plan_path),
        "plan_source": plan_source,
        "validation_passed": validation_passed,
        "validation_error": validation_error,
        "plan_error": plan_error,
        "emit_logic": plan.emit_logic,
        "judge_steps": [
            {
                "id": s.id,
                "condition_id": s.condition_id,
                "expected_answer": s.expected_answer,
                "default_view_count": len(s.default_evidence_refs),
                "followup_view_count": len(s.allowed_followup_views),
                "allowed_tools": s.allowed_tools,
            }
            for s in plan.judge_steps
        ],
        "budget": rule_complexity(rule),
    }


def run_from_rule_file(
    rule_file: str,
    llm=None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    timestamp: str | None = None,
    out_dir: str = "data/results",
) -> dict:
    """从已有的 rule JSON 文件加载 rule，生成 plan。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load rule
    rule_path = Path(rule_file)
    print(f"[1/3] Loading rule from {rule_path} ...")
    with open(rule_path, encoding="utf-8") as f:
        rule_data = json.load(f)
    rule = EvolvingRule.from_dict(rule_data)
    rule_id = rule.rule_id
    rule_version = rule.version
    label = (rule.metadata or {}).get("attack_label", rule_id)
    print(f"       Rule: {rule_id} v{rule_version} (label={label})")

    # 2. Generate plan
    print(f"[2/3] Generating plan for {rule_id} ...")
    if llm is not None:
        try:
            plan = PlanGenerator(
                llm=llm, enforce_label_plan_template=False).generate(rule)
            plan_source = "llm"
            plan_error = None
        except Exception as e:
            plan = compile_rule_to_baseline_plan(rule)
            plan_source = f"baseline_after_llm_failure({e})"
            plan_error = str(e)
    else:
        plan = compile_rule_to_baseline_plan(rule)
        plan_source = "baseline"
        plan_error = None
    print(f"       Plan: {plan.plan_id} ({plan_source})")

    # 3. Validate
    print(f"[3/3] Validating plan ...")
    validation_passed = False
    validation_error = None
    try:
        PlanValidator().validate(plan)
        validation_passed = True
        print(f"       Validation: PASS")
    except PlanValidationError as ve:
        validation_error = str(ve)
        print(f"       Validation: FAIL - {ve}")

    # 4. Write outputs
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_str = (llm_provider or "none").replace("-", "_")
    safe_label = re.sub(r"[^\w\-_. ]", "_", label)[:80]
    file_tag = f"{safe_label}_{provider_str}_{ts}"

    plan_path = out_dir / f"{file_tag}_plan.json"

    plan_dict = {
        **plan.to_dict(),
        "metadata": {
            **(plan.metadata or {}),
            "rule_id": rule_id,
            "rule_version": rule_version,
            "rule_source": str(rule_path),
            "plan_source": plan_source,
            "llm_model": llm_model,
            "llm_provider": llm_provider,
            "validation_passed": validation_passed,
            "validation_error": validation_error,
        }
    }
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_dict, f, ensure_ascii=False, indent=2)
    print(f"       Plan saved: {plan_path}")

    return {
        "label": label,
        "rule_id": rule_id,
        "rule_version": rule_version,
        "rule_source": str(rule_path),
        "plan_id": plan.plan_id,
        "plan_path": str(plan_path),
        "plan_source": plan_source,
        "validation_passed": validation_passed,
        "validation_error": validation_error,
        "plan_error": plan_error,
        "emit_logic": plan.emit_logic,
        "judge_steps": [
            {
                "id": s.id,
                "condition_id": s.condition_id,
                "expected_answer": s.expected_answer,
                "default_view_count": len(s.default_evidence_refs),
                "followup_view_count": len(s.allowed_followup_views),
                "allowed_tools": s.allowed_tools,
            }
            for s in plan.judge_steps
        ],
        "budget": rule_complexity(rule),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate EvolvingRule + EvidencePlan from label/description "
                    "or from an existing rule file."
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Attack label, e.g. reentrancy, price_manipulation. "
             "Required when not using --rule-file.",
    )
    parser.add_argument(
        "--attack-description",
        help="Human-readable attack description for rule generation. "
             "If omitted, a default description for the label is used.",
    )
    parser.add_argument(
        "--rule-file",
        default=None,
        help="Path to an existing rule JSON file. "
             "When provided, --label and --attack-description are ignored; "
             "the rule is loaded directly and only the plan is generated.",
    )
    parser.add_argument(
        "--llm-model",
        default="glm-5.1",
        help="LLM model for rule and plan generation.",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="LLM provider: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/results/rule_plan_pairs",
        help="Output directory for generated rule and plan JSON files.",
    )
    args = parser.parse_args()

    # 模式检查
    if args.rule_file:
        if not Path(args.rule_file).exists():
            parser.error(f"--rule-file not found: {args.rule_file}")
    elif not args.label:
        parser.error("Either --rule-file or --label is required.")

    # Build LLM
    llm = None
    try:
        llm = OpenAICompatibleLLM(
            model=args.llm_model,
            provider=args.llm_provider,
            minimax_thinking="adaptive",
        )
        print(
            f"LLM ready: provider={args.llm_provider}, model={args.llm_model}")
    except Exception as e:
        print(
            f"Warning: LLM not available ({e}). Using deterministic fallback.")

    if args.rule_file:
        # 模式2: 从已有 rule 文件加载，只生成 plan
        result = run_from_rule_file(
            rule_file=args.rule_file,
            llm=llm,
            llm_model=args.llm_model,
            llm_provider=args.llm_provider,
            out_dir=args.out_dir,
        )
    else:
        # 模式1: 从零生成 rule + plan
        attack_desc = resolve_attack_description(
            args.attack_description, args.label)

        result = run_one(
            label=args.label,
            attack_description=attack_desc,
            llm=llm,
            llm_model=args.llm_model,
            llm_provider=args.llm_provider,
            out_dir=args.out_dir,
        )

    status = "PASS" if result["validation_passed"] else "FAIL"
    print(
        f"\n=== {status} | rule={result['rule_id']} v{result['rule_version']} | plan={result['plan_id']} ===")
    if result.get("plan_error"):
        print(f"  Plan error: {result['plan_error']}")
    if result.get("validation_error"):
        print(f"  Validation error: {result['validation_error']}")
    print(f"  Judge steps: {len(result['judge_steps'])}")
    print(f"  Emit logic: {result['emit_logic']}")

    if not result["validation_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
