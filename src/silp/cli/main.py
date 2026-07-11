"""silpc — SILP CLI entry point.

Commands::

    silpc validate <ir.json>           Validate an IR JSON file.
    silpc compile <ir.json> [-f code]  Compile IR to a frontend surface string.
    silpc decode <text> [-f code]      Decode a surface string back to IR.
    silpc frontends                     List available frontends.
    silpc whitelist                     List verb whitelist (approved/excluded).
    silpc lock <ir.json> [-f code]     Generate a compile.lock entry.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..frontend import get_frontend, list_frontends
from ..ir import validate as validate_ir
from ..ir.whitelist import list_approved, list_excluded, whitelist_report

console = Console()

# ── Lock file path ────────────────────────────────────────────────────
LOCK_DIR = Path(".silp")
LOCK_FILE = LOCK_DIR / "compile.lock"


# ── Helpers ───────────────────────────────────────────────────────────


def _load_ir(path: str) -> dict:
    """Load a JSON file and return the parsed dict."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] file not found: {p}")
        sys.exit(1)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error:[/red] invalid JSON in {p}: {exc}")
        sys.exit(1)


def _print_result(result) -> None:
    """Pretty-print a ValidationResult."""
    if result.valid:
        console.print("[green]OK Valid[/green]")
        for w in result.warnings:
            console.print(f"  [yellow]WARN {w}[/yellow]")
    else:
        console.print("[red]FAIL Invalid[/red]")
        for e in result.errors:
            console.print(f"  [red]- {e}[/red]")
        for w in result.warnings:
            console.print(f"  [yellow]WARN {w}[/yellow]")


# ── CLI group ─────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="silp")
def cli() -> None:
    """SILP -- Semantic Interlingua Layer Protocol CLI."""


@cli.command("validate")
@click.argument("ir_file", type=click.Path())
def validate_cmd(ir_file: str) -> None:
    """Validate an IR JSON file."""
    data = _load_ir(ir_file)
    result = validate_ir(data)
    _print_result(result)
    if not result.valid:
        sys.exit(1)


@cli.command("compile")
@click.argument("ir_file", type=click.Path())
@click.option("-f", "--frontend", "frontend_name", default="code",
              help="Frontend name (default: code).")
def compile_cmd(ir_file: str, frontend_name: str) -> None:
    """Compile IR to a frontend surface string."""
    data = _load_ir(ir_file)
    result = validate_ir(data)
    if not result.valid:
        _print_result(result)
        sys.exit(1)

    try:
        fe = get_frontend(frontend_name)
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    output = fe.compile(result.ir)
    console.print(output)


@cli.command()
@click.argument("text")
@click.option("-f", "--frontend", "frontend_name", default="code",
              help="Frontend name (default: code).")
def decode(text: str, frontend_name: str) -> None:
    """Decode a frontend surface string back to IR JSON.

    Phase 1: demonstrates the round-trip MVP.
    """
    try:
        fe = get_frontend(frontend_name)
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    try:
        ir = fe.decode(text)
    except NotImplementedError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print_json(ir.to_compact_json())


@cli.command()
def frontends() -> None:
    """List available frontends."""
    table = Table(title="Registered Frontends")
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="green")
    for name in list_frontends():
        fe = get_frontend(name)
        table.add_row(name, fe.__class__.__name__)
    console.print(table)


@cli.command()
def whitelist() -> None:
    """List the verb whitelist (approved and excluded verbs)."""
    table = Table(title="SILP Verb Whitelist")
    table.add_column("Verb", style="cyan")
    table.add_column("Fn Name", style="green")
    table.add_column("1-Token", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Notes", style="dim")

    for row in whitelist_report():
        status = row["status"]
        notes = row.get("exclude_reason", "") or row.get("subword_analysis", "")[:60]
        table.add_row(
            f"!{row['verb']}",
            row["fn_name"],
            str(row["single_token_all"]),
            status,
            notes,
        )

    console.print(table)
    console.print(f"\n[green]Approved:[/green] {len(list_approved())} verbs")
    console.print(f"[red]Excluded:[/red] {len(list_excluded())} verbs")


@cli.command("lock")
@click.argument("ir_file", type=click.Path())
@click.option("-f", "--frontend", "frontend_name", default="code",
              help="Frontend name (default: code).")
def lock_cmd(ir_file: str, frontend_name: str) -> None:
    """Generate a compile.lock entry (IR hash -> frontend output).

    Per spec Phase 0.5: compile.lock is the immutable audit trail.
    Recompiling requires rm .silp/compile.lock (git-tracked).
    """
    data = _load_ir(ir_file)
    result = validate_ir(data)
    if not result.valid:
        _print_result(result)
        sys.exit(1)

    try:
        fe = get_frontend(frontend_name)
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    ir_json = result.ir.to_compact_json()
    ir_hash = hashlib.sha256(ir_json.encode()).hexdigest()[:16]
    frontend_output = fe.compile(result.ir)
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = {
        "ir_hash": ir_hash,
        "frontend": frontend_name,
        "output": frontend_output,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
    }

    LOCK_DIR.mkdir(exist_ok=True)
    with open(LOCK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print(f"[green]OK Locked[/green] {ir_hash} -> {frontend_output}")
    console.print(f"  Appended to {LOCK_FILE}")


def _git_commit() -> str:
    """Best-effort current git commit hash (empty string if not in a repo)."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


if __name__ == "__main__":
    cli()
