import os
import json
from dotenv import load_dotenv
from agents.bob_client import get_model, chat
from agents.state import AgentState

load_dotenv()
model = get_model()

# Directories to skip during the repo walk
_SKIP_DIRS = {'node_modules', '.git', 'dist', 'build'}

# Only analyse frontend/backend JS-family source files
_EXTENSIONS = ('.js', '.ts', '.jsx', '.tsx')


def dependency_agent(state: AgentState) -> dict:
    """
    Agent 1 — Dependency Agent
    Walks repo_path, reads all .js/.ts/.jsx/.tsx files,
    asks the model which files are affected by change_description,
    and returns {"affected_files": [...]}.
    """
    print("🔍 Dependency Agent running...")

    change    = state["change_description"]
    repo_path = state["repo_path"]
    errors    = list(state.get("errors") or [])

    # ── Walk the repo and collect JS-family files ──────────────────────
    files_content: dict[str, str] = {}

    print(f"   📂 Scanning repo: {repo_path}")
    for root, dirs, files in os.walk(repo_path):
        # Prune unwanted directories in-place so os.walk skips them
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]

        for file in files:
            if not file.endswith(_EXTENSIONS):
                continue

            filepath = os.path.join(root, file)
            relative = os.path.relpath(filepath, repo_path).replace("\\", "/")

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    # Cap per-file content to avoid blowing the context window
                    files_content[relative] = f.read()[:500]
            except Exception as read_err:
                files_content[relative] = ""
                print(f"   ⚠️  Could not read {relative}: {read_err}")

    print(f"   📄 {len(files_content)} JS/TS files found")

    # ── Build the prompt ───────────────────────────────────────────────
    prompt = f"""You are a code dependency analyzer.

The developer wants to: "{change}"

Here are all the project files with their content (truncated to 500 chars each):
{json.dumps(files_content, indent=2)[:8000]}

Analyze which files would be directly or indirectly affected by this change.

Return ONLY valid JSON with no additional explanation:
{{
    "affected_files": ["path/to/file1.js", "path/to/file2.ts"],
    "reason": "brief explanation"
}}
"""

    # ── Call the model ─────────────────────────────────────────────────
    print("   🤖 Asking model for dependency analysis...")
    try:
        text = chat(model, prompt)

        # Strip markdown code fences the model may wrap around the JSON
        text = text.replace("```json", "").replace("```", "").strip()

        result   = json.loads(text)
        affected = result.get("affected_files", [])

        if not isinstance(affected, list):
            raise ValueError(
                f"'affected_files' is not a list, got: {type(affected).__name__}"
            )

    except json.JSONDecodeError as e:
        msg = f"Dependency Agent — JSON parse error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)
        affected = []

    except Exception as e:
        msg = f"Dependency Agent — unexpected error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)
        affected = []

    print(f"✅ Dependency Agent: {len(affected)} file(s) affected")
    for f in affected:
        print(f"   • {f}")

    result_state: dict = {"affected_files": affected}
    if errors:
        result_state["errors"] = errors

    return result_state
