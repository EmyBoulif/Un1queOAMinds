import os
import json
from dotenv import load_dotenv
from agents.bob_client import get_model, chat
from agents.state import AgentState

load_dotenv()
model = get_model()

# Directories to skip during the test file search
_SKIP_DIRS = {'node_modules', '.git', 'dist', 'build'}

# Suffixes that identify test files
_TEST_SUFFIXES = ('.test.js', '.spec.js', '.test.ts', '.spec.ts',
                  '.test.jsx', '.spec.jsx', '.test.tsx', '.spec.tsx')


def _find_test_files(tests_dir: str) -> list[str]:
    """
    Walk tests_dir recursively and return relative paths of all test files.
    Returns an empty list if tests_dir does not exist.
    """
    if not os.path.isdir(tests_dir):
        return []

    found: list[str] = []
    for root, dirs, files in os.walk(tests_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for file in files:
            if file.endswith(_TEST_SUFFIXES):
                filepath = os.path.join(root, file)
                relative = os.path.relpath(filepath, tests_dir).replace("\\", "/")
                found.append(relative)
    return found


def test_agent(state: AgentState) -> dict:
    """
    Agent 3 — Test Agent
    Discovers all *.test.js / *.spec.js (and TS/JSX variants) under
    repo_path/tests, then asks the model which of those tests are
    relevant to the current change_description and affected_files.
    Returns {"tests_to_run": [...]}.
    """
    print("🧪 Test Agent running...")

    change         = state["change_description"]
    affected_files = state.get("affected_files") or []
    repo_path      = state["repo_path"]
    errors         = list(state.get("errors") or [])

    # ── 1. Discover test files under repo_path/tests ──────────────────
    tests_dir = os.path.join(repo_path, "tests")
    print(f"   🔎 Scanning for test files in: {tests_dir}")

    test_files = _find_test_files(tests_dir)

    if not test_files:
        print(f"   ⚠️  No test files found in {tests_dir}")
    else:
        print(f"   📋 {len(test_files)} test file(s) discovered:")
        for t in test_files:
            print(f"      • {t}")

    # ── 2. Read a snippet of each test file for context ────────────────
    test_contents: dict[str, str] = {}
    for rel_path in test_files:
        full_path = os.path.join(tests_dir, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                test_contents[rel_path] = f.read()[:400]
        except Exception as e:
            test_contents[rel_path] = ""
            print(f"   ⚠️  Could not read {rel_path}: {e}")

    # ── 3. Build the prompt ────────────────────────────────────────────
    prompt = f"""You are a test impact analyzer.

The developer wants to: "{change}"

Source files that will be changed:
{json.dumps(affected_files, indent=2)}

Available test files and their content (truncated to 400 chars each):
{json.dumps(test_contents, indent=2)[:4000]}

Determine which test files are relevant to the change above.
A test is relevant if it directly tests any of the changed source files
or exercises code paths that would be affected by the change.

Return ONLY valid JSON with no additional explanation:
{{
    "tests_to_run": ["CheckoutTest.js", "PaymentServiceTest.js"],
    "reason": "brief explanation"
}}

The paths in tests_to_run must match exactly the keys in the test file list above.
"""

    # ── 4. Call the model ──────────────────────────────────────────────
    print("   🤖 Asking model which tests to run...")
    tests: list[str] = []
    try:
        text = chat(model, prompt)
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)
        raw    = result.get("tests_to_run", [])

        if not isinstance(raw, list):
            raise ValueError(
                f"'tests_to_run' is not a list, got: {type(raw).__name__}"
            )

        # Keep only entries that are non-empty strings
        tests = [str(t).strip() for t in raw if str(t).strip()]

    except json.JSONDecodeError as e:
        msg = f"Test Agent — JSON parse error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)

    except Exception as e:
        msg = f"Test Agent — unexpected error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)

    print(f"✅ Test Agent: {len(tests)} test(s) to run")
    for t in tests:
        print(f"   • {t}")

    result_state: dict = {"tests_to_run": tests}
    if errors:
        result_state["errors"] = errors

    return result_state
