"""Small Windows-friendly GUI launcher for Codeforces statement fetching.

This is a local helper tool, not part of the research pipeline. It launches the
existing ``cf_diff.statement_features`` module in a background subprocess and
monitors progress from cached HTML files, the summary JSON, and the log file.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = Path("data/processed/features/model_table.parquet")
CACHE_DIR = Path("data/raw/codeforces/problem_pages")
OUTPUT_DIR = Path("data/processed/statement_features")
LOG_PATH = Path("outputs/logs/statement_features.log")
SUMMARY_PATH = OUTPUT_DIR / "statement_feature_summary.json"
SUBPROCESS_LOG_PATH = Path("outputs/logs/statement_features_gui_subprocess.log")
REFRESH_MS = 3000


class StatementFetchGui:
    """Tkinter launcher and progress monitor for statement page fetching."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the GUI."""
        self.root = root
        self.root.title("Codeforces Statement Fetcher")
        self.root.geometry("920x680")
        self.root.minsize(820, 580)

        self.process: subprocess.Popen[object] | None = None
        self.process_log_handle = None
        self.user_stopped = True
        self.restart_after_timestamp: float | None = None
        self.restart_count = 0
        self.last_exit_code: int | None = None
        self.last_seen_process: subprocess.Popen[object] | None = None
        self.last_total_problem_count = 0
        self.last_cached_page_count = 0
        self.last_summary: dict[str, object] = {}

        self.status_var = tk.StringVar(value="Stopped")
        self.restart_count_var = tk.StringVar(value="Auto restarts: 0")
        self.total_var = tk.StringVar(value="Total problems: unknown")
        self.cached_var = tk.StringVar(value="Cached pages: 0")
        self.percent_var = tk.StringVar(value="Progress: 0.00%")
        self.summary_var = tk.StringVar(value="No summary yet.")
        self.message_var = tk.StringVar(value="")
        self.auto_restart_var = tk.BooleanVar(value=True)

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh()

    def _build_widgets(self) -> None:
        """Create all visible widgets."""
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            outer,
            text="Codeforces Statement Text-Light Feature Fetcher",
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(anchor=tk.W)

        command_text = (
            "Runs: python -m cf_diff.statement_features "
            "--feature-path data/processed/features/model_table.parquet "
            "--cache-dir data/raw/codeforces/problem_pages "
            "--output-dir data/processed/statement_features "
            "--sleep-seconds 2.5 --timeout 30 "
            "--log-path outputs/logs/statement_features.log"
        )
        ttk.Label(
            outer,
            text=command_text,
            wraplength=880,
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(4, 10))

        button_frame = ttk.Frame(outer)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_button = ttk.Button(
            button_frame,
            text="Start / Continue",
            command=self.start_fetch,
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self.stop_fetch,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Checkbutton(
            button_frame,
            text="Auto-restart on crash",
            variable=self.auto_restart_var,
        ).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Button(
            button_frame,
            text="Open output folder",
            command=lambda: self.open_path(OUTPUT_DIR, is_folder=True),
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            button_frame,
            text="Open cache folder",
            command=lambda: self.open_path(CACHE_DIR, is_folder=True),
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            button_frame,
            text="Open log file",
            command=lambda: self.open_path(LOG_PATH, is_folder=False),
        ).pack(side=tk.LEFT)

        progress_frame = ttk.LabelFrame(outer, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X)

        ttk.Label(progress_frame, textvariable=self.status_var).grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, 16),
            pady=2,
        )
        ttk.Label(progress_frame, textvariable=self.total_var).grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=(0, 16),
            pady=2,
        )
        ttk.Label(progress_frame, textvariable=self.cached_var).grid(
            row=2,
            column=0,
            sticky=tk.W,
            padx=(0, 16),
            pady=2,
        )
        ttk.Label(progress_frame, textvariable=self.percent_var).grid(
            row=3,
            column=0,
            sticky=tk.W,
            padx=(0, 16),
            pady=2,
        )
        ttk.Label(progress_frame, textvariable=self.restart_count_var).grid(
            row=4,
            column=0,
            sticky=tk.W,
            padx=(0, 16),
            pady=2,
        )
        ttk.Label(progress_frame, textvariable=self.summary_var, wraplength=650).grid(
            row=0,
            column=1,
            rowspan=5,
            sticky=tk.W,
            pady=2,
        )
        progress_frame.columnconfigure(1, weight=1)

        self.progress_bar = ttk.Progressbar(
            outer,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100.0,
        )
        self.progress_bar.pack(fill=tk.X, pady=(10, 10))

        message = ttk.Label(
            outer,
            textvariable=self.message_var,
            foreground="#9A3412",
            wraplength=880,
        )
        message.pack(anchor=tk.W, pady=(0, 8))

        log_frame = ttk.LabelFrame(outer, text="Last 20 log lines", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=18,
            font=("Consolas", 9),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def start_fetch(self) -> None:
        """Start or resume the background statement feature extraction process."""
        self.user_stopped = False
        self.restart_after_timestamp = None
        self._start_fetch_impl(manual=True)

    def _start_fetch_impl(self, *, manual: bool) -> None:
        """Start the fetch subprocess, optionally as an automatic restart."""
        if self.is_running():
            if manual:
                messagebox.showinfo(
                    "Already running",
                    "A statement fetch subprocess is already running.",
                )
            return

        feature_abs = PROJECT_ROOT / FEATURE_PATH
        if not feature_abs.exists():
            self.message_var.set(f"Model table does not exist: {feature_abs}")
            if manual:
                messagebox.showerror(
                    "Missing model table",
                    f"Cannot start because this file is missing:\n{feature_abs}",
                )
            return

        (PROJECT_ROOT / CACHE_DIR).mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / SUBPROCESS_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "cf_diff.statement_features",
            "--feature-path",
            FEATURE_PATH.as_posix(),
            "--cache-dir",
            CACHE_DIR.as_posix(),
            "--output-dir",
            OUTPUT_DIR.as_posix(),
            "--sleep-seconds",
            "2.5",
            "--timeout",
            "30",
            "--log-path",
            LOG_PATH.as_posix(),
        ]

        env = os.environ.copy()
        src_path = str(PROJECT_ROOT / "src")
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            src_path
            if not existing_pythonpath
            else src_path + os.pathsep + existing_pythonpath
        )

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        self.process_log_handle = (PROJECT_ROOT / SUBPROCESS_LOG_PATH).open(
            "a",
            encoding="utf-8",
        )
        self.process_log_handle.write(
            "\n=== GUI launch at "
            + datetime.now().isoformat(timespec="seconds")
            + " ===\n"
        )
        self.process_log_handle.flush()

        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=self.process_log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except OSError as error:
            self._close_process_log_handle()
            self.process = None
            self.message_var.set(f"Failed to start subprocess: {error}")
            if manual:
                messagebox.showerror("Failed to start", str(error))
            return

        self.last_seen_process = self.process
        self.last_exit_code = None
        if manual:
            self.message_var.set(
                "Started statement fetching. Cached pages will be skipped."
            )
        else:
            self.restart_count += 1
            self.restart_count_var.set(f"Auto restarts: {self.restart_count}")
            self.message_var.set(
                "Auto-restarted statement fetching. Cached pages will be skipped."
            )
        self.refresh_once()

    def stop_fetch(self) -> None:
        """Terminate the running fetch subprocess, if any."""
        self.user_stopped = True
        self.restart_after_timestamp = None
        if not self.is_running():
            self.status_var.set("Status: stopped by user")
            self.message_var.set("No statement fetch subprocess is running.")
            self._update_button_state()
            return
        assert self.process is not None
        self.message_var.set("Stopping statement fetch subprocess...")
        self.process.terminate()
        self.root.after(5000, self._kill_if_still_running)
        self.refresh_once()

    def _kill_if_still_running(self) -> None:
        """Kill the subprocess if terminate did not finish it."""
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.message_var.set("Subprocess did not stop in time and was killed.")
        self._close_process_log_handle()
        self.refresh_once()

    def is_running(self) -> bool:
        """Return whether the managed subprocess is still active."""
        return self.process is not None and self.process.poll() is None

    def refresh(self) -> None:
        """Refresh process status, progress numbers, and log display."""
        self.refresh_once()
        self.root.after(REFRESH_MS, self.refresh)

    def refresh_once(self) -> None:
        """Refresh all GUI fields once without scheduling another timer."""
        self._refresh_progress()
        self._refresh_process_state()
        self._handle_auto_restart()
        self._refresh_log()
        self._update_button_state()

    def _refresh_process_state(self) -> None:
        """Update status text from the subprocess state."""
        if self.extraction_is_complete():
            self.restart_after_timestamp = None
            self.status_var.set("Status: complete")
            return
        if self.process is None:
            if self.user_stopped:
                self.status_var.set("Status: stopped by user")
            else:
                self.status_var.set("Status: stopped")
            return
        return_code = self.process.poll()
        if return_code is None:
            self.status_var.set("Status: running")
            return
        self.last_exit_code = return_code
        if self.user_stopped:
            self.status_var.set(f"Status: stopped by user (exit code: {return_code})")
        else:
            self.status_var.set(
                f"Status: crashed / exited unexpectedly (exit code: {return_code})"
            )
        self._close_process_log_handle()

    def _refresh_progress(self) -> None:
        """Read progress from model table, cache directory, and summary JSON."""
        total, total_message = self._read_total_problem_count()
        cached = self._read_cached_page_count()
        self.last_total_problem_count = total
        self.last_cached_page_count = cached
        self.last_summary = self._read_summary()
        percentage = (cached / total * 100.0) if total else 0.0
        percentage = min(max(percentage, 0.0), 100.0)

        self.total_var.set(total_message)
        self.cached_var.set(f"Cached HTML pages: {cached:,}")
        self.percent_var.set(f"Progress: {cached:,} / {total:,} ({percentage:.2f}%)")
        self.progress_bar["value"] = percentage
        self.summary_var.set(self._summary_text_from_payload(self.last_summary))

    def _read_total_problem_count(self) -> tuple[int, str]:
        """Read total unique problems from the feature table."""
        feature_abs = PROJECT_ROOT / FEATURE_PATH
        if not feature_abs.exists():
            return 0, f"Total problems: unavailable; missing {FEATURE_PATH}"
        try:
            frame = pd.read_parquet(feature_abs, engine="pyarrow")
            contest_column = next(
                (
                    column
                    for column in ("contest_id", "contestId", "contestid")
                    if column in frame.columns
                ),
                None,
            )
            if contest_column is None or "index" not in frame.columns:
                return 0, "Total problems: unavailable; missing contest/index columns"
            total = int(frame[[contest_column, "index"]].drop_duplicates().shape[0])
            return total, f"Total unique problems: {total:,}"
        except Exception as error:
            return 0, f"Total problems: failed to read model table ({error})"

    def _read_cached_page_count(self) -> int:
        """Count cached HTML pages."""
        cache_abs = PROJECT_ROOT / CACHE_DIR
        if not cache_abs.exists():
            return 0
        return sum(1 for path in cache_abs.glob("*.html") if path.is_file())

    def _read_summary_text(self) -> str:
        """Read selected fields from statement_feature_summary.json."""
        return self._summary_text_from_payload(self._read_summary())

    def _read_summary(self) -> dict[str, object]:
        """Read statement_feature_summary.json as a dictionary."""
        summary_abs = PROJECT_ROOT / SUMMARY_PATH
        if not summary_abs.exists():
            return {}
        try:
            payload = json.loads(summary_abs.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError) as error:
            return {"_error": f"Could not read summary JSON: {error}"}

    def _summary_text_from_payload(self, summary: dict[str, object]) -> str:
        """Render selected summary fields for the progress panel."""
        if not summary:
            return "No statement feature summary yet."
        if "_error" in summary:
            return str(summary["_error"])
        fields = [
            ("attempted_page_count", "attempted"),
            ("failed_fetch_count", "failed"),
            ("parsed_success_count", "parsed"),
            ("statement_available_rate", "available_rate"),
            ("fetched_page_count", "fetched"),
            ("cached_page_count", "summary_cached"),
        ]
        parts = []
        for key, label in fields:
            value = summary.get(key, "n/a")
            if isinstance(value, float):
                value = f"{value:.4f}"
            parts.append(f"{label}: {value}")
        return "Summary: " + " | ".join(parts)

    def extraction_is_complete(self) -> bool:
        """Return whether cached pages or summary indicate completion."""
        total = self.last_total_problem_count
        cached = self.last_cached_page_count
        if total > 0 and cached >= total:
            return True

        summary = self.last_summary
        try:
            attempted = int(summary.get("attempted_page_count", -1))
            input_rows = int(summary.get("input_row_count", -1))
            failed = int(summary.get("failed_fetch_count", -1))
        except (TypeError, ValueError):
            return False
        return input_rows > 0 and attempted >= input_rows and failed == 0

    def _handle_auto_restart(self) -> None:
        """Schedule or run automatic restarts without blocking the GUI."""
        if self.is_running():
            return
        if self.extraction_is_complete():
            self.restart_after_timestamp = None
            return
        if self.user_stopped:
            self.restart_after_timestamp = None
            return
        if not self.auto_restart_var.get():
            self.restart_after_timestamp = None
            return
        if self.process is None:
            return
        if self.process.poll() is None:
            return

        now = time.monotonic()
        if self.restart_after_timestamp is None:
            self.restart_after_timestamp = now + 15.0
        remaining = int(math.ceil(max(self.restart_after_timestamp - now, 0.0)))
        if remaining > 0:
            self.status_var.set(f"Status: restarting in {remaining}s")
            self.message_var.set(
                f"Process stopped unexpectedly. Restarting in {remaining}s..."
            )
            return

        self.restart_after_timestamp = None
        self.process = None
        self._start_fetch_impl(manual=False)

    def _refresh_log(self) -> None:
        """Show the last 20 lines of the extractor log."""
        log_abs = PROJECT_ROOT / LOG_PATH
        if not log_abs.exists():
            text = "No log yet."
        else:
            try:
                lines = log_abs.read_text(encoding="utf-8", errors="replace").splitlines()
                text = "\n".join(lines[-20:]) if lines else "No log yet."
            except OSError as error:
                text = f"Could not read log file: {error}"

        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, text)
        self.log_text.configure(state=tk.DISABLED)

    def _update_button_state(self) -> None:
        """Enable or disable buttons according to process state."""
        if self.is_running():
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
        else:
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)

    def _close_process_log_handle(self) -> None:
        """Close the subprocess log handle if it is open."""
        if self.process_log_handle is not None:
            self.process_log_handle.flush()
            self.process_log_handle.close()
            self.process_log_handle = None

    def open_path(self, path: Path, *, is_folder: bool) -> None:
        """Open a folder or file using the operating system shell."""
        absolute = PROJECT_ROOT / path
        if is_folder:
            absolute.mkdir(parents=True, exist_ok=True)
        elif not absolute.exists():
            messagebox.showinfo("Missing file", f"This file does not exist yet:\n{absolute}")
            return

        try:
            if hasattr(os, "startfile"):
                os.startfile(str(absolute))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(absolute)])
            else:
                subprocess.Popen(["xdg-open", str(absolute)])
        except OSError as error:
            messagebox.showerror("Open failed", str(error))

    def on_close(self) -> None:
        """Ask before closing while the fetch subprocess is running."""
        self.user_stopped = True
        self.restart_after_timestamp = None
        if self.is_running():
            should_stop = messagebox.askyesno(
                "Fetcher is running",
                "The statement fetch subprocess is still running.\n\n"
                "Stop it and close the GUI?",
            )
            if not should_stop:
                return
            self.stop_fetch()
            self.root.after(800, self.root.destroy)
            return
        self._close_process_log_handle()
        self.root.destroy()


def main() -> int:
    """Launch the Tkinter GUI."""
    root = tk.Tk()
    StatementFetchGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
