"""
Aegis-Quant-Lab — GUI Interface
=================================
Lightweight customtkinter GUI with:
  • Asset selection panel (checkboxes for 55+ coins)
  • Date range selector
  • Timeframe selector (15m, 1h, 3h)
  • "Download/Update Data" button
  • "Run Quantum Backtest" button
  • Real-time scrolling log window
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Callable

import config

# ──────────────────────────────────────────────
# Attempt customtkinter, fall back to tkinter
# ──────────────────────────────────────────────
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    _USE_CTK = True
except ImportError:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    _USE_CTK = False

logger = logging.getLogger("aegis.gui")


# ──────────────────────────────────────────────
# Log redirect handler → queue → GUI text widget
# ──────────────────────────────────────────────

class QueueHandler(logging.Handler):
    """Push log records into a queue for the GUI to consume."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)


# ──────────────────────────────────────────────
# Main App class
# ──────────────────────────────────────────────

class AegisQuantLabGUI:
    """
    Main application window.
    All heavy work runs in background threads to keep GUI responsive.
    """

    def __init__(self):
        self.log_queue: queue.Queue = queue.Queue()
        self._running = False

        # ── Window setup ──
        if _USE_CTK:
            self.root = ctk.CTk()
            self.root.title("Aegis-Quant-Lab v1.0")
            self.root.geometry("1280x820")
        else:
            self.root = tk.Tk()
            self.root.title("Aegis-Quant-Lab v1.0")
            self.root.geometry("1280x820")
            self.root.configure(bg="#1a1a2e")

        self._build_layout()
        self._setup_logging()
        self._poll_log_queue()

    # ──────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────

    def _build_layout(self):
        # ── Left panel: asset checkboxes ──
        if _USE_CTK:
            self._build_ctk_layout()
        else:
            self._build_tk_layout()

    def _build_ctk_layout(self):
        # Main grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_rowconfigure(0, weight=1)

        # ── Left frame: controls ──
        left = ctk.CTkFrame(self.root)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(left, text="ASSETS", font=("Helvetica", 14, "bold")).pack(pady=(10, 5))

        # Scrollable asset frame
        asset_scroll = ctk.CTkScrollableFrame(left, width=260, height=350)
        asset_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.asset_vars: dict[str, ctk.BooleanVar] = {}  # type: ignore
        for sym in config.DEFAULT_SYMBOLS:
            var = ctk.BooleanVar(value=True)  # type: ignore
            self.asset_vars[sym] = var
            ctk.CTkCheckBox(asset_scroll, text=sym.replace("/USDT", ""), variable=var, width=100).pack(anchor="w", padx=2, pady=1)

        # Select / Deselect all
        btn_frame = ctk.CTkFrame(left)
        btn_frame.pack(fill="x", padx=5, pady=3)
        ctk.CTkButton(btn_frame, text="Select All", width=100, command=self._select_all).pack(side="left", padx=3)
        ctk.CTkButton(btn_frame, text="Deselect All", width=100, command=self._deselect_all).pack(side="left", padx=3)

        # Date range
        ctk.CTkLabel(left, text="Date Range", font=("Helvetica", 12, "bold")).pack(pady=(10, 3))
        date_frame = ctk.CTkFrame(left)
        date_frame.pack(fill="x", padx=5)
        ctk.CTkLabel(date_frame, text="Start:").pack(side="left", padx=3)
        self.start_entry = ctk.CTkEntry(date_frame, width=100, placeholder_text="2024-01-01")
        self.start_entry.pack(side="left", padx=3)
        self.start_entry.insert(0, "2024-01-01")
        ctk.CTkLabel(date_frame, text="End:").pack(side="left", padx=3)
        self.end_entry = ctk.CTkEntry(date_frame, width=100, placeholder_text="2025-01-01")
        self.end_entry.pack(side="left", padx=3)
        self.end_entry.insert(0, "2025-01-01")

        # Timeframe
        ctk.CTkLabel(left, text="Timeframe", font=("Helvetica", 12, "bold")).pack(pady=(10, 3))
        self.tf_var = ctk.StringVar(value="1h")
        tf_frame = ctk.CTkFrame(left)
        tf_frame.pack(fill="x", padx=5)
        for tf in config.SUPPORTED_TIMEFRAMES:
            ctk.CTkRadioButton(tf_frame, text=tf, variable=self.tf_var, value=tf).pack(side="left", padx=8)

        # Action buttons
        ctk.CTkLabel(left, text="Actions", font=("Helvetica", 12, "bold")).pack(pady=(15, 3))
        ctk.CTkButton(left, text="Download / Update Data from Bybit",
                       command=self._on_download, fg_color="#e67e22", hover_color="#d35400").pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(left, text="Run Global Quantum Backtest",
                       command=self._on_run_backtest, fg_color="#27ae60", hover_color="#1e8449").pack(fill="x", padx=10, pady=5)

        # Status
        self.status_label = ctk.CTkLabel(left, text="Ready", font=("Helvetica", 11))
        self.status_label.pack(pady=10)

        # ── Right frame: log output ──
        right = ctk.CTkFrame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(right, text="REAL-TIME LOG", font=("Helvetica", 14, "bold")).pack(pady=(10, 5))
        self.log_text = ctk.CTkTextbox(right, font=("Consolas", 11), wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._enable_text_copy(self.log_text)

    def _build_tk_layout(self):
        """Fallback pure-tkinter layout."""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_rowconfigure(0, weight=1)

        # Left
        left = tk.Frame(self.root, bg="#16213e")
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(left, text="ASSETS", bg="#16213e", fg="white", font=("Helvetica", 14, "bold")).pack(pady=(10, 5))

        canvas = tk.Canvas(left, bg="#16213e", highlightthickness=0, width=260)
        scrollbar = tk.Scrollbar(left, orient="vertical", command=canvas.yview)
        asset_inner = tk.Frame(canvas, bg="#16213e")
        asset_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=asset_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=5)
        scrollbar.pack(side="right", fill="y")

        self.asset_vars: dict[str, tk.BooleanVar] = {}
        for sym in config.DEFAULT_SYMBOLS:
            var = tk.BooleanVar(value=True)
            self.asset_vars[sym] = var
            tk.Checkbutton(asset_inner, text=sym.replace("/USDT", ""), variable=var,
                           bg="#16213e", fg="white", selectcolor="#0f3460",
                           activebackground="#16213e", activeforeground="white").pack(anchor="w")

        # Buttons
        btn_f = tk.Frame(left, bg="#16213e")
        btn_f.pack(fill="x", padx=5, pady=3)
        tk.Button(btn_f, text="Select All", command=self._select_all, bg="#0f3460", fg="white").pack(side="left", padx=3)
        tk.Button(btn_f, text="Deselect All", command=self._deselect_all, bg="#0f3460", fg="white").pack(side="left", padx=3)

        # Date
        tk.Label(left, text="Start Date (YYYY-MM-DD):", bg="#16213e", fg="white").pack(pady=(10, 2))
        self.start_entry = tk.Entry(left, width=15)
        self.start_entry.pack()
        self.start_entry.insert(0, "2024-01-01")
        tk.Label(left, text="End Date:", bg="#16213e", fg="white").pack(pady=(5, 2))
        self.end_entry = tk.Entry(left, width=15)
        self.end_entry.pack()
        self.end_entry.insert(0, "2025-01-01")

        # Timeframe
        tk.Label(left, text="Timeframe:", bg="#16213e", fg="white").pack(pady=(10, 2))
        self.tf_var = tk.StringVar(value="1h")
        for tf in config.SUPPORTED_TIMEFRAMES:
            tk.Radiobutton(left, text=tf, variable=self.tf_var, value=tf,
                           bg="#16213e", fg="white", selectcolor="#0f3460").pack(anchor="w", padx=20)

        tk.Button(left, text="Download / Update Data", command=self._on_download,
                  bg="#e67e22", fg="white", font=("Helvetica", 11, "bold")).pack(fill="x", padx=10, pady=8)
        tk.Button(left, text="Run Global Quantum Backtest", command=self._on_run_backtest,
                  bg="#27ae60", fg="white", font=("Helvetica", 11, "bold")).pack(fill="x", padx=10, pady=8)

        self.status_label = tk.Label(left, text="Ready", bg="#16213e", fg="#2ecc71", font=("Helvetica", 11))
        self.status_label.pack(pady=10)

        # Right — log
        right = tk.Frame(self.root, bg="#0a0a23")
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        tk.Label(right, text="REAL-TIME LOG", bg="#0a0a23", fg="white", font=("Helvetica", 14, "bold")).pack(pady=(10, 5))

        import tkinter.scrolledtext as st
        self.log_text = st.ScrolledText(right, bg="#0a0a23", fg="#00ff41",
                                        font=("Consolas", 10), insertbackground="white")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._enable_text_copy(self.log_text)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _enable_text_copy(widget):
        """Enable Ctrl+C / Ctrl+A copy and right-click context menu on a text widget."""
        import tkinter as tk

        # ── Keyboard shortcuts ──
        def copy_selection(event=None):
            try:
                text = widget.selection_get()
                widget.clipboard_clear()
                widget.clipboard_append(text)
            except tk.TclError:
                pass  # no selection
            return "break"

        def select_all(event=None):
            widget.tag_add("sel", "1.0", "end")
            return "break"

        widget.bind("<Control-c>", copy_selection)
        widget.bind("<Control-C>", copy_selection)
        widget.bind("<Control-a>", select_all)
        widget.bind("<Control-A>", select_all)

        # ── Right-click context menu ──
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy", command=copy_selection, accelerator="Ctrl+C")

        def show_context_menu(event):
            menu.tk_popup(event.x_root, event.y_root)

        widget.bind("<Button-3>", show_context_menu)  # right-click

    def _select_all(self):
        for var in self.asset_vars.values():
            var.set(True)

    def _deselect_all(self):
        for var in self.asset_vars.values():
            var.set(False)

    def _get_selected_symbols(self) -> list[str]:
        return [sym for sym, var in self.asset_vars.items() if var.get()]

    def _get_start_date(self) -> datetime:
        text = self.start_entry.get().strip()
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def _get_end_date(self) -> datetime:
        text = self.end_entry.get().strip()
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def _log(self, msg: str):
        """Thread-safe log append."""
        self.log_queue.put(msg)

    def _set_status(self, text: str):
        if _USE_CTK:
            self.status_label.configure(text=text)
        else:
            self.status_label.config(text=text)

    # ──────────────────────────────────────────
    # Logging infrastructure
    # ──────────────────────────────────────────

    def _setup_logging(self):
        root_logger = logging.getLogger("aegis")
        root_logger.setLevel(logging.INFO)

        qh = QueueHandler(self.log_queue)
        qh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
        root_logger.addHandler(qh)

        # Also log to file
        fh = logging.FileHandler(os.path.join(config.LOG_DIR, "aegis_quant_lab.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s — %(message)s"))
        root_logger.addHandler(fh)

    def _poll_log_queue(self):
        """Poll the log queue and append messages to the text widget."""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                if _USE_CTK:
                    self.log_text.insert("end", msg + "\n")
                    self.log_text.see("end")
                else:
                    self.log_text.insert("end", msg + "\n")
                    self.log_text.see("end")
            except queue.Empty:
                break
        self.root.after(100, self._poll_log_queue)

    # ──────────────────────────────────────────
    # Download action
    # ──────────────────────────────────────────

    def _on_download(self):
        if self._running:
            self._log("Another task is already running. Please wait.")
            return
        self._running = True
        self._set_status("Downloading …")
        threading.Thread(target=self._download_worker, daemon=True).start()

    def _download_worker(self):
        try:
            import data_loader

            symbols = self._get_selected_symbols()
            tf = self.tf_var.get()
            start = self._get_start_date()
            end = self._get_end_date()

            if not symbols:
                self._log("ERROR: No symbols selected.")
                return

            self._log(f"Starting download: {len(symbols)} symbols, {tf}, {start.date()} to {end.date()}")

            data_loader.download_batch(symbols, tf, start, end, progress_cb=self._log)

            self._log("Download complete.")
            self._set_status("Download complete")
        except Exception as e:
            self._log(f"DOWNLOAD ERROR: {e}")
            logger.exception("Download failed")
            self._set_status("Download failed")
        finally:
            self._running = False

    # ──────────────────────────────────────────
    # Backtest action
    # ──────────────────────────────────────────

    def _on_run_backtest(self):
        if self._running:
            self._log("Another task is already running. Please wait.")
            return
        self._running = True
        self._set_status("Running backtest …")
        threading.Thread(target=self._backtest_worker, daemon=True).start()

    def _backtest_worker(self):
        try:
            import data_loader
            import feature_factory
            import cross_asset_analyst
            import backtest_simulator
            import risk_generator
            import visualizer

            symbols = self._get_selected_symbols()
            tf = self.tf_var.get()
            start = self._get_start_date()
            end = self._get_end_date()

            if not symbols:
                self._log("ERROR: No symbols selected.")
                return

            # 1. Load data
            self._log(f"\n{'=' * 60}")
            self._log("PHASE 1: Loading historical data …")
            self._log(f"{'=' * 60}")
            raw_data = data_loader.load_all_local(symbols, tf, start, end)
            if not raw_data:
                self._log("No cached data found. Please download first.")
                self._set_status("No data")
                return
            self._log(f"Loaded {len(raw_data)} symbols from cache")

            # 2. Feature generation
            self._log(f"\n{'=' * 60}")
            self._log("PHASE 2: Feature Factory — computing 200+ indicators …")
            self._log(f"{'=' * 60}")
            enriched = feature_factory.compute_features_batch(raw_data, progress_cb=self._log)
            self._log(f"Feature computation complete for {len(enriched)} symbols")

            # 3. Cross-asset analysis
            self._log(f"\n{'=' * 60}")
            self._log("PHASE 3: Cross-Asset Dependency Analysis …")
            self._log(f"{'=' * 60}")
            cross_results = cross_asset_analyst.run_full_analysis(raw_data, progress_cb=self._log)

            # 4. Backtest simulation
            self._log(f"\n{'=' * 60}")
            self._log("PHASE 4: Walk-Forward Backtest Simulation …")
            self._log(f"{'=' * 60}")
            bt_results = backtest_simulator.run_full_backtest(enriched, tf, progress_cb=self._log)

            # 5. Risk block generation
            self._log(f"\n{'=' * 60}")
            self._log("PHASE 5: Anti-Strategy Risk Block Generation …")
            self._log(f"{'=' * 60}")
            blocks = risk_generator.generate_risk_blocks(bt_results, enriched, cross_results, progress_cb=self._log)
            risk_path = risk_generator.save_risk_blocks(blocks)
            self._log(f"Risk blocks saved to: {risk_path}")

            # 6. Summary report
            report = risk_generator.generate_summary_report(bt_results, cross_results)
            self._log(report)

            report_path = os.path.join(config.OUTPUT_DIR, "summary_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            # 7. Visualisation
            self._log(f"\n{'=' * 60}")
            self._log("PHASE 6: Generating charts …")
            self._log(f"{'=' * 60}")
            chart_paths = visualizer.generate_all_charts(bt_results, cross_results)
            for p in chart_paths:
                self._log(f"  Chart saved: {p}")

            self._log(f"\n{'=' * 60}")
            self._log("ALL PHASES COMPLETE")
            self._log(f"Output directory: {config.OUTPUT_DIR}")
            self._log(f"{'=' * 60}")
            self._set_status("Backtest complete")

        except Exception as e:
            self._log(f"\nCRITICAL ERROR: {e}")
            logger.exception("Backtest pipeline failed")
            self._set_status("Backtest failed")
        finally:
            self._running = False

    # ──────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────

    def run(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()
