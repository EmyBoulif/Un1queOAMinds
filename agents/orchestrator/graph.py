import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from agents.state import AgentState
from agents.dependency.agent import dependency_agent
from agents.database.agent import database_agent
from agents.test_agent.agent import test_agent
from agents.bob_client import get_model, chat

load_dotenv()
model = get_model()


# ═══════════════════════════════════════════════════════════════
# NODE 1 — Run all 3 agents concurrently, merge results
# ═══════════════════════════════════════════════════════════════
def parallel_agents_node(state: AgentState) -> dict:
    """Runs dependency_agent, database_agent, test_agent in parallel
    using ThreadPoolExecutor, then merges their results into state."""
    print("\n🚀 Running 3 agents in parallel...")

    agents = {
        "dependency": dependency_agent,
        "database":   database_agent,
        "test":       test_agent,
    }

    results: dict[str, dict] = {}
    errors: list[str] = list(state.get("errors") or [])

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(fn, state): name
            for name, fn in agents.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                msg = f"parallel_agents_node — {name} agent raised: {e}"
                print(f"   ❌ {msg}")
                errors.append(msg)
                results[name] = {}

    # Collect any errors the individual agents stored in their own dicts
    for name, res in results.items():
        for err in res.get("errors") or []:
            if err not in errors:
                errors.append(err)

    merged: dict = {
        "affected_files":  results.get("dependency", {}).get("affected_files", []),
        "database_impact": results.get("database",   {}).get("database_impact", []),
        "tests_to_run":    results.get("test",       {}).get("tests_to_run",    []),
    }
    if errors:
        merged["errors"] = errors

    print(f"   ✅ Parallel agents complete — "
          f"{len(merged['affected_files'])} affected files, "
          f"{len(merged['database_impact'])} DB tables, "
          f"{len(merged['tests_to_run'])} tests")
    return merged


# ═══════════════════════════════════════════════════════════════
# NODE 2 — Impact Report
# ═══════════════════════════════════════════════════════════════
def impact_report_node(state: AgentState) -> dict:
    """Asks the model to assess risk level and produce a markdown
    impact summary from the three agents' aggregated findings."""
    print("\n📊 Generating Impact Report...")

    errors: list[str] = list(state.get("errors") or [])

    prompt = f"""You are a senior tech lead reviewing a code change.

Change requested: "{state['change_description']}"

Analysis results:
- Affected files ({len(state['affected_files'])}):
  {json.dumps(state['affected_files'], indent=2)}
- Database tables impacted:
  {json.dumps(state['database_impact'], indent=2)}
- Tests to run ({len(state['tests_to_run'])}):
  {json.dumps(state['tests_to_run'], indent=2)}

Generate a clear, concise impact report in markdown and assess overall risk.

Return ONLY valid JSON with no additional explanation:
{{
    "risk_level": "HIGH",
    "impact_summary": "## Impact Report\\n\\n..."
}}

risk_level must be exactly one of: HIGH, MEDIUM, LOW.
"""

    risk_level    = "MEDIUM"
    impact_summary = ""
    try:
        text = chat(model, prompt)
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)

        risk_level     = str(result.get("risk_level", "MEDIUM")).strip().upper()
        impact_summary = str(result.get("impact_summary", "")).strip()

        if risk_level not in {"HIGH", "MEDIUM", "LOW"}:
            risk_level = "MEDIUM"

    except json.JSONDecodeError as e:
        msg = f"impact_report_node — JSON parse error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)
        impact_summary = "Could not generate impact report (JSON parse error)."

    except Exception as e:
        msg = f"impact_report_node — unexpected error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)
        impact_summary = "Could not generate impact report."

    print(f"   ⚠️  Risk Level: {risk_level}")

    merged: dict = {
        "risk_level":     risk_level,
        "impact_summary": impact_summary,
    }
    if errors:
        merged["errors"] = errors
    return merged


# ═══════════════════════════════════════════════════════════════
# NODE 3 — Implementation Plan
# ═══════════════════════════════════════════════════════════════
def implementation_plan_node(state: AgentState) -> dict:
    """Asks the model to produce a numbered, step-by-step implementation
    plan based on the impact summary and change context."""
    print("\n📋 Generating Implementation Plan...")

    errors: list[str] = list(state.get("errors") or [])

    prompt = f"""You are a senior developer creating a detailed implementation plan.

Change: "{state['change_description']}"
Risk Level: {state['risk_level']}

Impact Summary:
{state['impact_summary']}

Affected files: {json.dumps(state['affected_files'])}
DB Impact:      {json.dumps(state['database_impact'])}
Tests to run:   {json.dumps(state['tests_to_run'])}

Write a numbered, step-by-step implementation plan in markdown.

Return ONLY valid JSON with no additional explanation:
{{
    "implementation_plan": "## Implementation Plan\\n\\n1. ...\\n2. ...\\n3. ..."
}}
"""

    plan = ""
    try:
        text = chat(model, prompt)
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        plan   = str(result.get("implementation_plan", "")).strip()

    except json.JSONDecodeError as e:
        msg = f"implementation_plan_node — JSON parse error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)
        plan = "Could not generate implementation plan (JSON parse error)."

    except Exception as e:
        msg = f"implementation_plan_node — unexpected error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)
        plan = "Could not generate implementation plan."

    print("   ✅ Plan generated!")

    merged: dict = {"implementation_plan": plan}
    if errors:
        merged["errors"] = errors
    return merged


# ═══════════════════════════════════════════════════════════════
# NODE 4 — Human Approval Gate (terminal prompt)
# ═══════════════════════════════════════════════════════════════
def human_approval_node(state: AgentState) -> dict:
    """Prints the impact summary and implementation plan, then asks
    the user to approve or reject in the terminal (y/n)."""

    sep = "═" * 60
    print(f"\n{sep}")
    print("📊  IMPACT REPORT")
    print(sep)
    print(state["impact_summary"])
    print(f"\n📋  IMPLEMENTATION PLAN")
    print(sep)
    print(state["implementation_plan"])
    print(sep)
    print(f"\n⚠️   Risk Level : {state['risk_level']}")
    print(f"📁   Files      : {len(state['affected_files'])}")
    print(f"🗄️   DB Tables  : {len(state['database_impact'])}")
    print(f"🧪   Tests      : {len(state['tests_to_run'])}")
    print(sep)

    answer   = input("\n✅  Approve this change? (y/n): ")
    approved = answer.strip().lower() in {"yes", "y", "oui"}

    if approved:
        print("✅  Change APPROVED — proceeding with execution.")
    else:
        print("❌  Change REJECTED — jumping to final report.")

    return {
        "human_approved": approved,
        "retry_count":    0,          # reset counter at approval time
    }


# ═══════════════════════════════════════════════════════════════
# NODE 5 — Bob Execution (simulated)
# ═══════════════════════════════════════════════════════════════
def bob_execution_node(state: AgentState) -> dict:
    """Simulates Bob applying the change: prints what would be modified
    per affected file and sets code_modified=True."""
    print("\n⚙️   IBM Bob executing change (simulation)...")

    affected = state.get("affected_files") or []
    plan     = state.get("implementation_plan", "(no plan)")

    print(f"   📝 Change  : {state['change_description']}")
    print(f"   📋 Plan    :\n{plan}\n")
    print(f"   🔧 Files that would be modified ({len(affected)}):")

    for filepath in affected:
        # Determine what kind of change based on filename heuristics
        lower = filepath.lower()
        if "service" in lower or "payment" in lower:
            action = "Add PayPal branch / handler logic"
        elif "test" in lower or "spec" in lower:
            action = "Add / update test cases for new payment method"
        elif "schema" in lower or "model" in lower or "migration" in lower:
            action = "Add migration / schema column for PayPal fields"
        elif "route" in lower or "api" in lower or "checkout" in lower:
            action = "Expose new PayPal endpoint or update request validation"
        else:
            action = "Update to support the requested change"
        print(f"      • {filepath}  →  {action}")

    print("   ✅ Simulation complete — code_modified set to True")

    return {
        "code_modified": True,
        "test_results":  {"status": "pending"},
    }


# ═══════════════════════════════════════════════════════════════
# NODE 6 — Test Runner (simulated per-test pass/fail)
# ═══════════════════════════════════════════════════════════════
def test_runner_node(state: AgentState) -> dict:
    """Simulates running each test in tests_to_run, recording pass/fail
    per test. Increments retry_count when tests fail so the router can
    decide whether to retry bob_execution or move to final_report."""
    print("\n🧪  Running tests (simulation)...")

    tests_to_run = state.get("tests_to_run") or []
    retry_count  = state.get("retry_count", 0)

    if not tests_to_run:
        print("   ⚠️  No tests to run.")
        test_results = {
            "status":       "skipped",
            "passed":       True,    # nothing failed → allow graph to proceed
            "per_test":     {},
            "tests_run":    0,
            "tests_passed": 0,
            "tests_failed": 0,
        }
        return {"test_results": test_results}

    # Simulate: all tests pass on first attempt, one fails on retry
    per_test: dict[str, str] = {}
    if retry_count == 0:
        # First run — all pass
        for t in tests_to_run:
            per_test[t] = "pass"
    else:
        # Retry run — still mark all as passed (after fix simulation)
        for t in tests_to_run:
            per_test[t] = "pass"

    failed   = [t for t, r in per_test.items() if r == "fail"]
    all_pass = len(failed) == 0

    for test, result in per_test.items():
        icon = "✅" if result == "pass" else "❌"
        print(f"   {icon}  {test}  [{result.upper()}]")

    test_results = {
        "status":       "passed" if all_pass else "failed",
        "passed":       all_pass,
        "per_test":     per_test,
        "tests_run":    len(tests_to_run),
        "tests_passed": len(tests_to_run) - len(failed),
        "tests_failed": len(failed),
    }

    # Increment retry_count here (routers cannot mutate state in LangGraph)
    new_retry_count = retry_count + (0 if all_pass else 1)

    print(f"   {'✅' if all_pass else '❌'}  "
          f"{test_results['tests_passed']}/{test_results['tests_run']} passed")

    return {
        "test_results": test_results,
        "retry_count":  new_retry_count,
    }


# ═══════════════════════════════════════════════════════════════
# NODE 7 — Final Report (model-generated markdown)
# ═══════════════════════════════════════════════════════════════
def final_report_node(state: AgentState) -> dict:
    """Uses get_model()/chat() to generate a polished markdown final
    report summarising the whole run: impact, plan, execution, test results."""
    print("\n📄  Generating Final Report...")

    errors       = state.get("errors") or []
    test_results = state.get("test_results") or {}
    approved     = state.get("human_approved", False)
    code_mod     = state.get("code_modified", False)

    prompt = f"""You are IBM Bob, an AI coding assistant.
Generate a concise, well-structured markdown final report for this change request.

## Context
- Change requested : "{state['change_description']}"
- Risk Level       : {state.get('risk_level', 'N/A')}
- Human approved   : {approved}
- Code modified    : {code_mod}

## Impact Analysis
- Affected files   : {json.dumps(state.get('affected_files', []))}
- DB tables impact : {json.dumps(state.get('database_impact', []))}
- Tests identified : {json.dumps(state.get('tests_to_run', []))}

## Test Results
{json.dumps(test_results, indent=2)}

## Errors encountered
{json.dumps(errors)}

## Implementation Plan
{state.get('implementation_plan', 'N/A')}

Write the final report in markdown. Include:
1. Executive summary
2. Impact analysis (files, DB, tests)
3. Execution outcome (approved / modified / test results)
4. Errors (if any)
5. Final status: COMPLETED / REJECTED / FAILED

Return ONLY valid JSON with no additional explanation:
{{
    "final_report": "# Final Report\\n\\n..."
}}
"""

    report = ""
    try:
        text = chat(model, prompt)
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        report = str(result.get("final_report", "")).strip()

    except json.JSONDecodeError as e:
        print(f"   ❌ final_report_node — JSON parse error: {e}")
        # Fall back to a minimal built-in report
        report = _fallback_report(state)

    except Exception as e:
        print(f"   ❌ final_report_node — unexpected error: {e}")
        report = _fallback_report(state)

    print(report)
    return {"final_report": report}


def _fallback_report(state: AgentState) -> str:
    """Minimal markdown report used when the model call fails."""
    test_results = state.get("test_results") or {}
    passed       = test_results.get("passed", False)
    approved     = state.get("human_approved", False)
    return (
        f"# 🤖 IBM Bob — Change Analysis Report\n\n"
        f"**Change:** {state.get('change_description', 'N/A')}\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Risk Level | {state.get('risk_level', 'N/A')} |\n"
        f"| Files Affected | {len(state.get('affected_files') or [])} |\n"
        f"| DB Tables | {len(state.get('database_impact') or [])} |\n"
        f"| Tests | {len(state.get('tests_to_run') or [])} |\n"
        f"| Human Approved | {'✅ Yes' if approved else '❌ No'} |\n"
        f"| Code Modified | {'✅ Yes' if state.get('code_modified') else '❌ No'} |\n"
        f"| Tests | {'✅ PASSED' if passed else '❌ FAILED'} |\n"
        f"| Retry Count | {state.get('retry_count', 0)} |\n\n"
        f"**Status:** "
        f"{'✅ COMPLETED' if passed and approved else '❌ FAILED / REJECTED'}\n"
    )


# ═══════════════════════════════════════════════════════════════
# ROUTERS
# ═══════════════════════════════════════════════════════════════
def approval_router(
    state: AgentState,
) -> Literal["bob_execution", "final_report"]:
    """Route after human_approval_node."""
    if state.get("human_approved"):
        return "bob_execution"
    return "final_report"


def test_result_router(
    state: AgentState,
) -> Literal["final_report", "bob_execution"]:
    """Route after test_runner_node.
    - All tests passed  →  final_report
    - Tests failed AND retry_count < 1  →  bob_execution  (one retry)
    - Tests failed AND retry_count >= 1 →  final_report   (max retries)
    """
    results     = state.get("test_results") or {}
    retry_count = state.get("retry_count", 0)

    if results.get("passed", False):
        return "final_report"

    if retry_count < 1:
        print(f"   🔄 Tests failed — retrying (attempt {retry_count + 1}/1)...")
        return "bob_execution"

    print("   ⚠️  Max retries reached — moving to final report.")
    return "final_report"


# ═══════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════
def build_graph():
    builder = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────
    builder.add_node("parallel_agents",     parallel_agents_node)
    builder.add_node("impact_report",       impact_report_node)
    builder.add_node("implementation_plan", implementation_plan_node)
    builder.add_node("human_approval",      human_approval_node)
    builder.add_node("bob_execution",       bob_execution_node)
    builder.add_node("test_runner",         test_runner_node)
    builder.add_node("final_report",        final_report_node)

    # ── Linear edges ───────────────────────────────────────────
    builder.add_edge(START,                 "parallel_agents")
    builder.add_edge("parallel_agents",     "impact_report")
    builder.add_edge("impact_report",       "implementation_plan")
    builder.add_edge("implementation_plan", "human_approval")

    # ── Conditional: human approval gate ───────────────────────
    builder.add_conditional_edges(
        "human_approval",
        approval_router,
        {
            "bob_execution": "bob_execution",
            "final_report":  "final_report",
        },
    )

    # ── Bob → tests ────────────────────────────────────────────
    builder.add_edge("bob_execution", "test_runner")

    # ── Conditional: retry or finish ───────────────────────────
    builder.add_conditional_edges(
        "test_runner",
        test_result_router,
        {
            "final_report":  "final_report",
            "bob_execution": "bob_execution",
        },
    )

    builder.add_edge("final_report", END)

    return builder.compile()


agent_graph = build_graph()
