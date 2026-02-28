#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tkinter_ai_dashboard import AIDashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Point d'entrée unique pour lancer le builder CPU et le dashboard live.",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "dashboard", "trainer"],
        default="live",
        help=(
            "live: dashboard + démarrage automatique de l'entraînement, "
            "dashboard: interface seulement, trainer: entraînement seul en console"
        ),
    )
    parser.add_argument("--script", default="cpu_circuit_builder.py", help="Script IA à exécuter")
    parser.add_argument("--out-dir", default="out_design", help="Dossier de sortie")
    parser.add_argument("--design-budget-sec", type=int, default=300, help="Durée de recherche en secondes")
    parser.add_argument("--eval-cycles", type=int, default=1000, help="Cycles simulés par évaluation")
    return parser


def launch_dashboard(script: Path, out_dir: Path, budget: int, eval_cycles: int, auto_start: bool) -> None:
    app = AIDashboard()
    app.script_var.set(str(script))
    app.out_dir_var.set(str(out_dir))
    app.budget_var.set(str(budget))
    app.eval_cycles_var.set(str(eval_cycles))

    if auto_start:
        app.after(200, app.start_training)

    app.mainloop()


def launch_trainer(script: Path, out_dir: Path, budget: int, eval_cycles: int) -> int:
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--out-dir",
        str(out_dir),
        "--design-budget-sec",
        str(budget),
        "--eval-cycles",
        str(eval_cycles),
    ]
    return subprocess.call(cmd, cwd=str(script.parent))


def main() -> None:
    args = build_parser().parse_args()
    script = Path(args.script).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not script.exists():
        raise FileNotFoundError(f"Script introuvable: {script}")

    if args.mode == "trainer":
        raise SystemExit(launch_trainer(script, out_dir, args.design_budget_sec, args.eval_cycles))

    launch_dashboard(
        script=script,
        out_dir=out_dir,
        budget=args.design_budget_sec,
        eval_cycles=args.eval_cycles,
        auto_start=args.mode == "live",
    )


if __name__ == "__main__":
    main()
