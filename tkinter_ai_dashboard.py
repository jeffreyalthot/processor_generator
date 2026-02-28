#!/usr/bin/env python3
from __future__ import annotations

import csv
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


@dataclass
class IterRow:
    it: str
    run_id: str
    valid_eda: str
    score: float
    perf_mips: float
    avg_temp: float
    avg_power: float


class AIDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CPU EDA AI - Visualiseur de progression")
        self.geometry("1200x760")

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.process_exit_code: int | None = None

        self.script_var = tk.StringVar(value="cpu_circuit_builder.py")
        self.out_dir_var = tk.StringVar(value="out_design")
        self.budget_var = tk.StringVar(value="300")
        self.eval_cycles_var = tk.StringVar(value="1000")
        self.status_var = tk.StringVar(value="Prêt")
        self.best_var = tk.StringVar(value="Best score: N/A")
        self.last_refresh_var = tk.StringVar(value="Dernière lecture: jamais")
        self.floorplan_var = tk.StringVar(value="Floorplan: en attente")
        self.floorplan_image: tk.PhotoImage | None = None
        self.floorplan_mtime: float | None = None

        self._build_ui()
        self.after(250, self._flush_log_queue)
        self.after(1000, self._refresh_from_outputs)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Script IA:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.script_var, width=60).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Parcourir", command=self._pick_script).grid(row=0, column=2, padx=2)

        ttk.Label(top, text="Dossier outputs:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.out_dir_var, width=60).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Parcourir", command=self._pick_output_dir).grid(row=1, column=2, padx=2)

        ttk.Label(top, text="Budget (s):").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.budget_var, width=12).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(top, text="Eval cycles:").grid(row=2, column=1, sticky="e", padx=(0, 100))
        ttk.Entry(top, textvariable=self.eval_cycles_var, width=12).grid(row=2, column=2, sticky="w", padx=4)

        ttk.Button(top, text="Lancer", command=self.start_training).grid(row=0, column=3, rowspan=3, padx=6)
        ttk.Button(top, text="Stop", command=self.stop_training).grid(row=0, column=4, rowspan=3)

        top.grid_columnconfigure(1, weight=1)

        info = ttk.Frame(self, padding=(8, 0))
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.status_var).pack(side="left")
        ttk.Label(info, text=" | ").pack(side="left")
        ttk.Label(info, textvariable=self.best_var).pack(side="left")
        ttk.Label(info, text=" | ").pack(side="left")
        ttk.Label(info, textvariable=self.last_refresh_var).pack(side="left")

        split = ttk.Panedwindow(self, orient="vertical")
        split.pack(fill="both", expand=True, padx=8, pady=8)

        table_frame = ttk.Labelframe(split, text="Progression (training_log.csv)")
        self.tree = ttk.Treeview(
            table_frame,
            columns=("iter", "run", "eda", "score", "mips", "temp", "power"),
            show="headings",
            height=14,
        )
        cols = [
            ("iter", "Iter", 70),
            ("run", "Run ID", 160),
            ("eda", "EDA", 60),
            ("score", "Score", 90),
            ("mips", "Perf (MIPS)", 100),
            ("temp", "Temp (°C)", 90),
            ("power", "Power (W)", 90),
        ]
        for key, text, width in cols:
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, anchor="center")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        preview_frame = ttk.Labelframe(split, text="Floorplan best (live)")
        ttk.Label(preview_frame, textvariable=self.floorplan_var).pack(anchor="w", padx=6, pady=(6, 2))
        self.floorplan_label = ttk.Label(preview_frame)
        self.floorplan_label.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        log_frame = ttk.Labelframe(split, text="Logs live")
        self.log_text = ScrolledText(log_frame, wrap="word", font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)

        split.add(table_frame, weight=2)
        split.add(preview_frame, weight=2)
        split.add(log_frame, weight=3)

    def _pick_script(self) -> None:
        p = filedialog.askopenfilename(title="Sélectionner le script python", filetypes=[("Python", "*.py")])
        if p:
            self.script_var.set(p)

    def _pick_output_dir(self) -> None:
        p = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if p:
            self.out_dir_var.set(p)

    def start_training(self) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showinfo("Déjà lancé", "Le process IA est déjà en cours.")
            return

        script = Path(self.script_var.get()).expanduser()
        if not script.exists():
            messagebox.showerror("Script introuvable", f"Impossible de trouver:\n{script}")
            return

        self.stop_event.clear()
        try:
            budget = int(self.budget_var.get())
            eval_cycles = int(self.eval_cycles_var.get())
        except ValueError:
            messagebox.showerror("Paramètres invalides", "Budget et eval cycles doivent être des entiers.")
            return

        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--out-dir",
            str(Path(self.out_dir_var.get()).expanduser()),
            "--design-budget-sec",
            str(budget),
            "--eval-cycles",
            str(eval_cycles),
        ]
        self.status_var.set(f"Exécution: {' '.join(cmd)}")
        self.log_text.insert("end", f"\n=== START {time.strftime('%H:%M:%S')} ===\n")
        self.log_text.see("end")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(script.parent),
        )
        threading.Thread(target=self._read_process_output, daemon=True).start()

    def stop_training(self) -> None:
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
                self.log_queue.put("[UI] Le process ne répondait pas, arrêt forcé (kill).\n")
            self.status_var.set("Process arrêté")
            self.log_queue.put("\n[UI] Process terminé par l'utilisateur.\n")

    def _read_process_output(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            if self.stop_event.is_set():
                break
            self.log_queue.put(line)

        rc = self.process.wait()
        self.log_queue.put(f"\n[UI] Process terminé (code={rc}).\n")
        self.process_exit_code = rc

    def _flush_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.insert("end", line)
                self.log_text.see("end")
        except queue.Empty:
            pass

        if self.process_exit_code is not None:
            self.status_var.set(f"Terminé (code={self.process_exit_code})")
            self.process_exit_code = None

        self.after(250, self._flush_log_queue)

    def _refresh_from_outputs(self) -> None:
        out_dir = Path(self.out_dir_var.get()).expanduser()
        csv_path = out_dir / "training_log.csv"
        log_path = out_dir / "run.log"

        if csv_path.exists():
            rows = self._read_last_rows(csv_path, limit=40)
            self._refresh_tree(rows)
            if rows:
                best = max(rows, key=lambda r: r.score)
                self.best_var.set(f"Best score: {best.score:.5f} ({best.run_id})")

        if log_path.exists():
            self.status_var.set(f"Monitoring: {log_path}")

        self._refresh_floorplan_preview(out_dir)

        self.last_refresh_var.set(f"Dernière lecture: {time.strftime('%H:%M:%S')}")
        self.after(1000, self._refresh_from_outputs)

    def _read_last_rows(self, csv_path: Path, limit: int) -> list[IterRow]:
        rows: list[IterRow] = []
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        rows.append(
                            IterRow(
                                it=r.get("iter", ""),
                                run_id=r.get("run_id", ""),
                                valid_eda=r.get("valid_eda", ""),
                                score=float(r.get("score", "0") or 0),
                                perf_mips=float(r.get("perf_ips", "0") or 0) / 1e6,
                                avg_temp=float(r.get("avg_temp_c", "0") or 0),
                                avg_power=float(r.get("avg_power_w", "0") or 0),
                            )
                        )
                    except ValueError:
                        continue
        except OSError:
            return []
        return rows[-limit:]

    def _refresh_floorplan_preview(self, out_dir: Path) -> None:
        png_path = out_dir / "floorplan_best.png"
        if not png_path.exists():
            self.floorplan_var.set("Floorplan: aucun PNG détecté (matplotlib optionnel)")
            return

        try:
            mtime = png_path.stat().st_mtime
            if self.floorplan_mtime == mtime:
                return
            self.floorplan_mtime = mtime
            self.floorplan_image = tk.PhotoImage(file=str(png_path))
            self.floorplan_label.configure(image=self.floorplan_image)
            self.floorplan_var.set(f"Floorplan: {png_path.name} @ {time.strftime('%H:%M:%S')}")
        except tk.TclError as exc:
            self.floorplan_var.set(f"Floorplan: erreur de lecture PNG ({exc})")

    def _refresh_tree(self, rows: list[IterRow]) -> None:
        self.tree.delete(*self.tree.get_children())
        for r in rows:
            eda = "PASS" if r.valid_eda == "1" else "FAIL"
            self.tree.insert(
                "",
                "end",
                values=(
                    r.it,
                    r.run_id,
                    eda,
                    f"{r.score:.5f}",
                    f"{r.perf_mips:.3f}",
                    f"{r.avg_temp:.2f}",
                    f"{r.avg_power:.6f}",
                ),
            )


if __name__ == "__main__":
    app = AIDashboard()
    app.mainloop()
