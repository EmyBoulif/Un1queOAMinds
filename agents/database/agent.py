import os
import json
from dotenv import load_dotenv
from agents.bob_client import get_model, chat
from agents.state import AgentState

load_dotenv()
model = get_model()

# Directories to skip during schema/migration walks
_SKIP_DIRS = {'node_modules', '.git', 'dist', 'build'}

# Filename keywords that suggest schema / migration content
_SCHEMA_KEYWORDS = {'schema', 'migration', 'migrate', 'model', 'database', 'seed'}

# Valid risk values the model is expected to return
_VALID_RISKS = {"HIGH", "MEDIUM", "LOW"}


def _is_schema_file(filename: str, dirparts: list[str]) -> bool:
    """Return True if the file is likely a schema/migration/model file."""
    lower = filename.lower()
    # .sql files are always relevant
    if lower.endswith('.sql'):
        return True
    # Files whose names contain a schema-related keyword
    if any(kw in lower for kw in _SCHEMA_KEYWORDS):
        return True
    # Files sitting inside a 'models' directory (any depth)
    if 'models' in dirparts:
        return True
    return False


def database_agent(state: AgentState) -> dict:
    """
    Agent 2 — Database Agent
    Reads affected files + schema/migration files from the repo,
    then asks the model which database tables are impacted by the change.
    Returns {"database_impact": [{"table": str, "risk": "HIGH"|"MEDIUM"|"LOW"}, ...]}.
    """
    print("🗄️  Database Agent running...")

    change         = state["change_description"]
    affected_files = state.get("affected_files") or []
    repo_path      = state["repo_path"]
    errors         = list(state.get("errors") or [])

    # ── 1. Read content of affected files ─────────────────────────────
    print(f"   📄 Reading {len(affected_files)} affected file(s)...")
    files_content: dict[str, str] = {}
    for rel_path in affected_files:
        full_path = os.path.join(repo_path, rel_path.lstrip("/\\"))
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                files_content[rel_path] = f.read()[:1000]
        except Exception as e:
            files_content[rel_path] = ""
            print(f"   ⚠️  Could not read affected file {rel_path}: {e}")

    # ── 2. Search repo for schema / migration files ────────────────────
    print(f"   🔎 Searching for schema/migration files in: {repo_path}")
    schema_files: dict[str, str] = {}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]

        # Build the list of directory components for models/ detection
        rel_root  = os.path.relpath(root, repo_path)
        dirparts  = rel_root.replace("\\", "/").split("/")

        for file in files:
            if not _is_schema_file(file, dirparts):
                continue

            filepath    = os.path.join(root, file)
            relative    = os.path.relpath(filepath, repo_path).replace("\\", "/")

            # Don't double-read files already loaded as affected
            if relative in files_content:
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    schema_files[relative] = f.read()[:500]
            except Exception as e:
                print(f"   ⚠️  Could not read schema file {relative}: {e}")

    print(f"   📋 {len(schema_files)} schema/migration file(s) found")

    # ── 3. Build the prompt ────────────────────────────────────────────
    prompt = f"""You are a database impact analyzer.

The developer wants to: "{change}"

Affected source files (with content):
{json.dumps(files_content, indent=2)[:4000]}

Schema / migration / model files found in the repo:
{json.dumps(schema_files, indent=2)[:2000]}

Analyze which database tables are impacted by this change and assess the risk.

Return ONLY valid JSON with no additional explanation:
{{
    "database_impact": [
        {{"table": "transactions", "risk": "HIGH"}},
        {{"table": "users",        "risk": "LOW"}}
    ],
    "details": "brief explanation"
}}

Risk must be exactly one of: HIGH, MEDIUM, LOW.
"""

    # ── 4. Call the model ──────────────────────────────────────────────
    print("   🤖 Asking model for database impact analysis...")
    db_impact: list[dict] = []
    try:
        text = chat(model, prompt)
        text = text.replace("```json", "").replace("```", "").strip()

        result    = json.loads(text)
        raw_items = result.get("database_impact", [])

        if not isinstance(raw_items, list):
            raise ValueError(
                f"'database_impact' is not a list, got: {type(raw_items).__name__}"
            )

        # Normalise each entry: keep only table + risk, validate risk value
        for item in raw_items:
            table = str(item.get("table", "")).strip()
            risk  = str(item.get("risk", "")).strip().upper()

            if not table:
                continue
            if risk not in _VALID_RISKS:
                risk = "MEDIUM"   # safe default for unexpected values

            db_impact.append({"table": table, "risk": risk})

    except json.JSONDecodeError as e:
        msg = f"Database Agent — JSON parse error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)

    except Exception as e:
        msg = f"Database Agent — unexpected error: {e}"
        print(f"   ❌ {msg}")
        errors.append(msg)

    print(f"✅ Database Agent: {len(db_impact)} table(s) impacted")
    for entry in db_impact:
        print(f"   • {entry['table']}  [{entry['risk']}]")

    result_state: dict = {"database_impact": db_impact}
    if errors:
        result_state["errors"] = errors

    return result_state
