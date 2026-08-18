from pathlib import Path
import re
import json
import ast
from datetime import datetime

ROOT = Path(__file__).resolve().parent

# پوشه‌هایی که معمولاً نباید بررسی عمیق شوند
IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "venv",
    "env",
    ".venv",
    "build",
    "dist",
}

CODE_EXTENSIONS = {
    ".py", ".ps1", ".bat", ".cmd", ".spec", ".iss"
}

DATA_EXTENSIONS = {
    ".csv", ".tsv", ".txt", ".xlsx", ".xls",
    ".json", ".pkl", ".joblib", ".edf",
    ".fif", ".npy", ".npz"
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".svg", ".ico", ".webp"
}


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def scan_files():
    files = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        files.append(path)

    return sorted(files)


def project_tree(files):
    tree = {}

    for path in files:
        rel = path.relative_to(ROOT)
        current = tree

        for part in rel.parts[:-1]:
            current = current.setdefault(part, {})

        current[rel.name] = {
            "__file__": True,
            "size": path.stat().st_size,
            "extension": path.suffix.lower()
        }

    return tree


def print_tree(node, prefix=""):
    items = list(node.items())

    for i, (name, value) in enumerate(items):
        last = i == len(items) - 1
        connector = "└── " if last else "├── "

        if isinstance(value, dict) and value.get("__file__") is True:
            print(
                prefix
                + connector
                + f"{name}  "
                + f"[{value['size']} bytes]"
            )
        else:
            print(prefix + connector + name)
            print_tree(
                value,
                prefix + ("    " if last else "│   ")
            )


def find_paths_in_code(path):
    results = []

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        return results

    patterns = [
        r'["\']([^"\']+\.(?:csv|txt|pkl|joblib|json|edf|fif|npy|npz|xlsx|xls|svg|png|ico))["\']',
        r'["\']([^"\']*(?:models|data|results|figures|assets|reports)[^"\']*)["\']',
        r'["\']([^"\']*(?:\\|/)[^"\']+)["\']',
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            if match not in results:
                results.append(match)

    return results


def check_python_syntax(path):
    try:
        source = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        ast.parse(source)
        return True, None

    except SyntaxError as e:
        return False, (
            f"line={e.lineno}, "
            f"column={e.offset}, "
            f"message={e.msg}"
        )

    except Exception as e:
        return False, str(e)


def check_referenced_paths(path, references):
    missing = []

    for ref in references:
        # مسیرهای absolute را جداگانه بررسی می‌کنیم
        candidate1 = ROOT / ref
        candidate2 = path.parent / ref

        if candidate1.exists():
            continue

        if candidate2.exists():
            continue

        # مسیرهایی که صرفاً اسم پکیج/ماژول هستند را نادیده بگیر
        if not (
            "/" in ref
            or "\\" in ref
            or "." in Path(ref).name
        ):
            continue

        missing.append(ref)

    return missing


def collect_summary(files):
    summary = {
        "python": [],
        "data": [],
        "images": [],
        "other": []
    }

    for path in files:
        ext = path.suffix.lower()

        if ext == ".py":
            summary["python"].append(relative(path))

        elif ext in DATA_EXTENSIONS:
            summary["data"].append(relative(path))

        elif ext in IMAGE_EXTENSIONS:
            summary["images"].append(relative(path))

        else:
            summary["other"].append(relative(path))

    return summary


def main():
    print("=" * 80)
    print("PARKINSON AI PROJECT AUDIT")
    print("=" * 80)

    print(f"\nProject root:")
    print(ROOT)

    print(f"\nAudit time:")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    files = scan_files()

    print(f"\nTotal project files:")
    print(len(files))

    # ---------------------------------------------------------
    # TREE
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("PROJECT STRUCTURE")
    print("=" * 80)

    tree = project_tree(files)
    print(ROOT.name)
    print_tree(tree)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    summary = collect_summary(files)

    print("\n" + "=" * 80)
    print("FILE SUMMARY")
    print("=" * 80)

    print(f"Python files : {len(summary['python'])}")
    print(f"Data files   : {len(summary['data'])}")
    print(f"Images/assets: {len(summary['images'])}")
    print(f"Other files  : {len(summary['other'])}")

    # ---------------------------------------------------------
    # PYTHON SYNTAX
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("PYTHON SYNTAX CHECK")
    print("=" * 80)

    syntax_errors = []

    for rel in summary["python"]:
        path = ROOT / rel

        ok, error = check_python_syntax(path)

        if ok:
            print(f"[OK]   {rel}")

        else:
            print(f"[ERROR] {rel}")
            print(f"        {error}")

            syntax_errors.append({
                "file": rel,
                "error": error
            })

    # ---------------------------------------------------------
    # PATH REFERENCES
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("PATH / DATA REFERENCE CHECK")
    print("=" * 80)

    path_report = []

    for rel in summary["python"]:
        path = ROOT / rel

        references = find_paths_in_code(path)

        if not references:
            continue

        missing = check_referenced_paths(
            path,
            references
        )

        result = {
            "file": rel,
            "references": references,
            "missing": missing
        }

        path_report.append(result)

        if missing:
            print(f"\n[WARNING] {rel}")

            for item in missing:
                print(f"   Missing/Unresolved: {item}")

    # ---------------------------------------------------------
    # IMPORTANT FILES
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("IMPORTANT PROJECT FILES")
    print("=" * 80)

    important_patterns = [
        "main.py",
        "main_window.py",
        "biomarker_panel.py",
        "explainability_panel.py",
        "report_generator.py",
        "build_exe.spec",
        "installer.iss",
    ]

    for pattern in important_patterns:
        matches = list(ROOT.rglob(pattern))

        if matches:
            for match in matches:
                print(f"[FOUND] {relative(match)}")
        else:
            print(f"[MISSING] {pattern}")

    # ---------------------------------------------------------
    # MODEL FILES
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("MODEL / ML FILES")
    print("=" * 80)

    model_files = []

    for path in files:
        name = path.name.lower()

        if (
            path.suffix.lower() in {".pkl", ".joblib"}
            or "model" in name
            or "scaler" in name
            or "genes" in name
        ):
            model_files.append(relative(path))

    if model_files:
        for item in model_files:
            print(f"[MODEL] {item}")
    else:
        print("No model-related files found.")

    # ---------------------------------------------------------
    # SAVE JSON REPORT
    # ---------------------------------------------------------

    report = {
        "project_root": str(ROOT),
        "audit_time": datetime.now().isoformat(),
        "total_files": len(files),
        "summary": summary,
        "syntax_errors": syntax_errors,
        "path_report": path_report,
        "model_files": model_files,
    }

    output = ROOT / "project_audit_report.json"

    with output.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(f"\nReport saved to:")
    print(output)

    print("\nIMPORTANT:")
    print("No files were moved, renamed, deleted, or modified.")


if __name__ == "__main__":
    main()