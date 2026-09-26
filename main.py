from dotenv import load_dotenv
from agents.orchestrator.graph import agent_graph

load_dotenv()

_DEFAULT_CHANGE = "Add PayPal support alongside existing Stripe integration"
_REPO_PATH      = "./payment/IBM-bob-payment-demo"


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║        🤖  Bob Change Impact Mode                ║")
    print("╚══════════════════════════════════════════════════╝")

    raw = input(
        f"\n📝  Describe your change\n"
        f"    (default: {_DEFAULT_CHANGE})\n"
        f"    > "
    ).strip()
    change = raw if raw else _DEFAULT_CHANGE

    print(f"\n🗂️   Repo path : {_REPO_PATH}")
    print(f"📝   Change    : {change}\n")

    initial_state = {
        # ── Input ──────────────────────────────────────
        "change_description":  change,
        "repo_path":           _REPO_PATH,
        # ── Agent outputs (empty until agents run) ─────
        "affected_files":      [],
        "database_impact":     [],
        "tests_to_run":        [],
        # ── Impact report ──────────────────────────────
        "risk_level":          "",
        "impact_summary":      "",
        # ── Implementation plan ────────────────────────
        "implementation_plan": "",
        # ── Human approval ─────────────────────────────
        "human_approved":      False,
        # ── Execution ──────────────────────────────────
        "code_modified":       False,
        "test_results":        {},
        "retry_count":         0,
        # ── Final report ───────────────────────────────
        "final_report":        "",
        # ── Errors ─────────────────────────────────────
        "errors":              [],
    }

    result = agent_graph.invoke(initial_state)

    print("\n" + "═" * 60)
    print("📄  FINAL REPORT")
    print("═" * 60)
    print(result.get("final_report", "(no report generated)"))
    print("═" * 60)

    errors = result.get("errors") or []
    if errors:
        print(f"\n⚠️   {len(errors)} error(s) recorded during the run:")
        for err in errors:
            print(f"   • {err}")

    print("\n✅  Pipeline completed.")


if __name__ == "__main__":
    main()
