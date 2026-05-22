import tkinter as tk
from tkinter import ttk
import math
import random
import time
import threading

# ─── Color Palette ────────────────────────────────────────────────────────────
BG          = "#0f1117"
PANEL       = "#1a1d27"
ACCENT1     = "#4f8ef7"   # HMS  – blue
ACCENT2     = "#f7764f"   # HMS-OS – orange
TEXT        = "#e8eaf0"
SUBTEXT     = "#7a7f9a"
BORDER      = "#2a2d3e"
SUCCESS     = "#4fbd7a"
ENTRY_BG    = "#232637"
PROGRESS_BG = "#12151f"


def fmt(v):
    """Shared formatter used for BOTH cards and chart labels."""
    if abs(v) >= 1e6:  return f"{v:.2e}"
    if abs(v) >= 1000: return f"{v:,.0f}"
    return f"{v:.4f}"          # 4 dp keeps small memory values distinct


# ─── Simulated Algorithm Runners (with callback) ──────────────────────────────

def run_hms(N, D, T, K, progress_cb=None):
    t0 = time.time()
    ops = 0
    best_val = float("inf")
    solutions = [[random.uniform(-5, 5) for _ in range(D)] for _ in range(N)]

    for iteration in range(T):
        fitness = [sum(x**2 for x in sol) for sol in solutions]
        best_idx = fitness.index(min(fitness))
        best_val = fitness[best_idx]

        for i in range(N):
            neighbor = [solutions[i][d] + random.gauss(0, 1) for d in range(D)]
            if sum(x**2 for x in neighbor) < fitness[i]:
                solutions[i] = neighbor
            ops += D

        centroids = random.sample(solutions, min(K, N))
        for _ in range(3):
            clusters = [[] for _ in range(K)]
            for sol in solutions:
                dists = [sum((sol[d]-c[d])**2 for d in range(D)) for c in centroids]
                clusters[dists.index(min(dists))].append(sol)
                ops += D * K
            centroids = [
                [sum(s[d] for s in cl) / len(cl) if cl else centroids[k][d]
                 for d in range(D)]
                for k, cl in enumerate(clusters)
            ]

        W = solutions[best_idx]
        for i in range(N):
            solutions[i] = [solutions[i][d] + random.random() * (W[d] - solutions[i][d])
                            for d in range(D)]
            ops += D

        elapsed = (time.time() - t0) * 1000
        if progress_cb:
            progress_cb(iteration + 1, T, best_val, elapsed, ops)

    elapsed = (time.time() - t0) * 1000
    mem = N * D * 8 / (1024**2)
    return elapsed, mem, best_val, ops


def run_hms_os(N, D, T, K1, K2, progress_cb=None):
    t0 = time.time()
    ops = 0
    best_val = float("inf")
    solutions = [[random.uniform(-5, 5) for _ in range(D)] for _ in range(N)]

    for iteration in range(T):
        fitness = [sum(x**2 for x in sol) for sol in solutions]
        best_idx = fitness.index(min(fitness))
        best_val = fitness[best_idx]

        ranked = sorted(range(N), key=lambda i: fitness[i])
        for rank, i in enumerate(ranked):
            scale = 0.5 if rank < N // 2 else 1.5
            neighbor = [solutions[i][d] + random.gauss(0, scale) for d in range(D)]
            if sum(x**2 for x in neighbor) < fitness[i]:
                solutions[i] = neighbor
            ops += D

        c1 = random.sample(solutions, min(K1, N))
        for _ in range(3):
            cl1 = [[] for _ in range(K1)]
            for sol in solutions:
                dists = [sum((sol[d]-c[d])**2 for d in range(D)) for c in c1]
                cl1[dists.index(min(dists))].append(sol)
                ops += D * K1
            c1 = [[sum(s[d] for s in cl) / len(cl) if cl else c1[k][d]
                   for d in range(D)] for k, cl in enumerate(cl1)]

        fvals = [[fitness[i]] for i in range(N)]
        c2 = [[random.choice(fvals)[0]] for _ in range(K2)]
        for _ in range(3):
            cl2 = [[] for _ in range(K2)]
            for j, fv in enumerate(fvals):
                dists = [abs(fv[0] - c[0]) for c in c2]
                cl2[dists.index(min(dists))].append(j)
                ops += K2
            c2 = [[sum(fvals[j][0] for j in cl) / len(cl) if cl else c2[k][0]]
                  for k, cl in enumerate(cl2)]

        W = solutions[best_idx]
        best_cluster = min(range(K1), key=lambda k: len(cl1[k]) == 0 or
                           sum(sum(s[d]**2 for d in range(D)) for s in cl1[k]) / max(len(cl1[k]), 1))
        C = c1[best_cluster]
        for i in range(N):
            r1, r2 = random.random(), random.random()
            solutions[i] = [
                solutions[i][d] + r1*(W[d]-solutions[i][d]) + r2*(C[d]-solutions[i][d])
                for d in range(D)]
            ops += D

        elapsed = (time.time() - t0) * 1000
        if progress_cb:
            progress_cb(iteration + 1, T, best_val, elapsed, ops)

    elapsed = (time.time() - t0) * 1000
    mem = N * D * 8 / (1024**2) * 1.12
    return elapsed, mem, best_val, ops


# ─── Main Application ─────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HMS vs HMS-OS")
        self.configure(bg=BG)
        try:
            self.state("zoomed")
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                self.attributes("-fullscreen", True)
        self.resizable(True, True)

        self._lock = threading.Lock()
        self._hms_live   = {"time": 0, "mem": 0, "best": float("inf"), "ops": 0, "iter": 0}
        self._hmsos_live = {"time": 0, "mem": 0, "best": float("inf"), "ops": 0, "iter": 0}
        self._hms_result   = None   # filled when HMS thread finishes
        self._hmsos_result = None   # filled when HMS-OS thread finishes
        self._T = 100

        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        title_fr = tk.Frame(self, bg=BG)
        title_fr.pack(fill="x", padx=40, pady=(28, 6))
        tk.Label(title_fr, text="HMS vs HMS-OS", font=("Courier New", 28, "bold"),
                 fg=TEXT, bg=BG).pack(side="left")

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x", padx=40, pady=(14, 0))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=40, pady=18)

        # ── Left panel: parameters ─────────────────────────────────────────────
        left = tk.Frame(body, bg=PANEL, bd=0, relief="flat",
                        highlightthickness=1, highlightbackground=BORDER)
        left.pack(side="left", fill="y", ipadx=20, ipady=20, padx=(0, 16))

        tk.Label(left, text="PARAMETERS", font=("Courier New", 12, "bold"),
                 fg=ACCENT1, bg=PANEL).grid(row=0, column=0, columnspan=3,
                                            sticky="w", padx=20, pady=(16, 12))
        params = [
            ("Solutions  (N)", "50",  "10–200"),
            ("Dimensions (D)", "50",  "10–200"),
            ("Iterations  (T)", "100", "10–500"),
            ("Clusters K / K₁", "3",  "2–10"),
            ("Clusters    K₂",  "3",  "2–10  (HMS-OS only)"),
        ]
        self.entries = []
        for r, (lbl, default, hint) in enumerate(params, start=1):
            tk.Label(left, text=lbl, font=("Courier New", 12), fg=TEXT, bg=PANEL,
                     anchor="w").grid(row=r, column=0, sticky="w", padx=20, pady=8)
            e = tk.Entry(left, width=7, font=("Courier New", 13), bg=ENTRY_BG,
                         fg=TEXT, insertbackground=TEXT, relief="flat",
                         highlightthickness=1, highlightcolor=ACCENT1,
                         highlightbackground=BORDER)
            e.insert(0, default)
            e.grid(row=r, column=1, padx=(6, 16), pady=8, sticky="w")
            tk.Label(left, text=hint, font=("Courier New", 9), fg=SUBTEXT,
                     bg=PANEL).grid(row=r, column=2, padx=(0, 16), sticky="w")
            self.entries.append(e)

        self.run_btn = tk.Button(
            left, text="▶  RUN", font=("Courier New", 13, "bold"),
            bg=ACCENT1, fg="#fff", activebackground="#3a6fd4",
            relief="flat", cursor="hand2", padx=24, pady=10,
            command=self._start_run)
        self.run_btn.grid(row=len(params)+1, column=0, columnspan=3,
                          padx=20, pady=(22, 8), sticky="ew")

        self.status_lbl = tk.Label(left, text="", font=("Courier New", 10),
                                   fg=SUBTEXT, bg=PANEL)
        self.status_lbl.grid(row=len(params)+2, column=0, columnspan=3, padx=20)

        # ── Progress bars ──────────────────────────────────────────────────────
        prog_frame = tk.Frame(left, bg=PANEL)
        prog_frame.grid(row=len(params)+3, column=0, columnspan=3,
                        padx=20, pady=(14, 4), sticky="ew")

        tk.Label(prog_frame, text="HMS", font=("Courier New", 9, "bold"),
                 fg=ACCENT1, bg=PANEL, width=7, anchor="w").grid(row=0, column=0, sticky="w")
        self.hms_prog_bar = tk.Canvas(prog_frame, height=10, bg=PROGRESS_BG,
                                      highlightthickness=0, relief="flat")
        self.hms_prog_bar.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.hms_prog_pct = tk.Label(prog_frame, text="0%", font=("Courier New", 9),
                                     fg=ACCENT1, bg=PANEL, width=5)
        self.hms_prog_pct.grid(row=0, column=2, padx=(4, 0))

        tk.Label(prog_frame, text="HMS-OS", font=("Courier New", 9, "bold"),
                 fg=ACCENT2, bg=PANEL, width=7, anchor="w").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.hmsos_prog_bar = tk.Canvas(prog_frame, height=10, bg=PROGRESS_BG,
                                        highlightthickness=0, relief="flat")
        self.hmsos_prog_bar.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(6, 0))
        self.hmsos_prog_pct = tk.Label(prog_frame, text="0%", font=("Courier New", 9),
                                       fg=ACCENT2, bg=PANEL, width=5)
        self.hmsos_prog_pct.grid(row=1, column=2, padx=(4, 0), pady=(6, 0))
        prog_frame.columnconfigure(1, weight=1)

        # Live status labels (one per algo, stacked)
        self.hms_iter_lbl = tk.Label(left, text="", font=("Courier New", 9),
                                     fg=ACCENT1, bg=PANEL)
        self.hms_iter_lbl.grid(row=len(params)+4, column=0, columnspan=3, padx=20, pady=(4, 0))
        self.hmsos_iter_lbl = tk.Label(left, text="", font=("Courier New", 9),
                                       fg=ACCENT2, bg=PANEL)
        self.hmsos_iter_lbl.grid(row=len(params)+5, column=0, columnspan=3, padx=20, pady=(2, 0))

        # ── Right panel: results ───────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        cards_fr = tk.Frame(right, bg=BG)
        cards_fr.pack(fill="x")

        metrics = ["Execution Time (ms)", "Memory Usage (MB)",
                   "Best Solution Value", "Total Operations"]
        self.hms_vals   = []
        self.hmsos_vals = []

        for col, metric in enumerate(metrics):
            card = tk.Frame(cards_fr, bg=PANEL, highlightthickness=1,
                            highlightbackground=BORDER)
            card.grid(row=0, column=col, padx=6, pady=6, sticky="nsew", ipadx=14, ipady=14)
            cards_fr.columnconfigure(col, weight=1)

            tk.Label(card, text=metric, font=("Courier New", 10), fg=SUBTEXT,
                     bg=PANEL, wraplength=160, justify="center").pack(pady=(12, 6))

            h_lbl = tk.Label(card, text="—", font=("Courier New", 18, "bold"),
                             fg=ACCENT1, bg=PANEL)
            h_lbl.pack()
            tk.Label(card, text="HMS", font=("Courier New", 10), fg=ACCENT1,
                     bg=PANEL).pack()

            o_lbl = tk.Label(card, text="—", font=("Courier New", 18, "bold"),
                             fg=ACCENT2, bg=PANEL)
            o_lbl.pack(pady=(10, 0))
            tk.Label(card, text="HMS-OS", font=("Courier New", 10), fg=ACCENT2,
                     bg=PANEL).pack(pady=(0, 12))

            self.hms_vals.append(h_lbl)
            self.hmsos_vals.append(o_lbl)

        # Bar chart
        chart_wrap = tk.Frame(right, bg=PANEL, highlightthickness=1,
                              highlightbackground=BORDER)
        chart_wrap.pack(fill="both", expand=True, pady=(12, 0))
        tk.Label(chart_wrap, text="Runtime & Memory Comparison",
                 font=("Courier New", 11, "bold"), fg=SUBTEXT, bg=PANEL).pack(pady=(12, 0))
        self.canvas = tk.Canvas(chart_wrap, bg=PANEL, bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=24, pady=(6, 16))
        self._chart_items = None   # created lazily on first draw
        self._chart_size  = (0, 0) # last known (W, H) — rebuild if resized
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Winner banner
        self.winner_fr = tk.Frame(right, bg=BG)
        self.winner_fr.pack(fill="x", pady=(12, 0))
        self.winner_lbl = tk.Label(self.winner_fr, text="",
                                   font=("Courier New", 13, "bold"),
                                   fg=SUCCESS, bg=BG)
        self.winner_lbl.pack()

    # ── Progress bar drawing (flicker-free) ───────────────────────────────────

    def _draw_progress(self, canvas, pct_label, pct, color):
        canvas.update_idletasks()
        w = canvas.winfo_width() or 160
        h = canvas.winfo_height() or 10
        fill_w = max(0, int(w * pct))
        shine_h = max(1, h // 3)
        light = self._lighten(color)

        if not getattr(canvas, "_pb_items", None):
            bg_id    = canvas.create_rectangle(0, 0, w, h, fill=PROGRESS_BG, outline="")
            fill_id  = canvas.create_rectangle(0, 0, 0, h, fill=color, outline="")
            shine_id = canvas.create_rectangle(0, 0, 0, shine_h, fill=light, outline="")
            canvas._pb_items = (bg_id, fill_id, shine_id)

        bg_id, fill_id, shine_id = canvas._pb_items
        canvas.coords(bg_id,    0, 0, w, h)
        canvas.coords(fill_id,  0, 0, fill_w, h)
        canvas.coords(shine_id, 0, 0, fill_w, shine_h)
        canvas.itemconfig(fill_id,  fill=color)
        canvas.itemconfig(shine_id, fill=light)
        pct_label.config(text=f"{int(pct*100)}%")

    def _lighten(self, hex_color):
        try:
            r = min(255, int(hex_color[1:3], 16) + 40)
            g = min(255, int(hex_color[3:5], 16) + 40)
            b = min(255, int(hex_color[5:7], 16) + 40)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color

    # ── Run logic (parallel threads) ───────────────────────────────────────────

    def _start_run(self):
        self.run_btn.config(state="disabled", text="running…")
        self.status_lbl.config(text="")
        self.winner_lbl.config(text="")
        self.canvas.delete("all")
        self._chart_items = None   # force recreate on next draw
        self.hms_iter_lbl.config(text="")
        self.hmsos_iter_lbl.config(text="")

        for lbl in self.hms_vals + self.hmsos_vals:
            lbl.config(text="…", fg=SUBTEXT)

        self._draw_progress(self.hms_prog_bar,   self.hms_prog_pct,   0, ACCENT1)
        self._draw_progress(self.hmsos_prog_bar, self.hmsos_prog_pct, 0, ACCENT2)
        # Reset pb item caches so they're recreated cleanly
        self.hms_prog_bar._pb_items   = None
        self.hmsos_prog_bar._pb_items = None

        with self._lock:
            self._hms_live   = {"time": 0, "mem": 0, "best": float("inf"), "ops": 0, "iter": 0}
            self._hmsos_live = {"time": 0, "mem": 0, "best": float("inf"), "ops": 0, "iter": 0}
            self._hms_result   = None
            self._hmsos_result = None

        try:
            N  = int(self.entries[0].get())
            D  = int(self.entries[1].get())
            T  = int(self.entries[2].get())
            K  = int(self.entries[3].get())
            K2 = int(self.entries[4].get())
        except ValueError:
            self._show_error("Invalid input — use integers only.")
            return

        self._T = T
        self._pending = 2   # countdown: both threads must finish

        threading.Thread(target=self._thread_hms,   args=(N, D, T, K),     daemon=True).start()
        threading.Thread(target=self._thread_hmsos, args=(N, D, T, K, K2), daemon=True).start()

    # ── Worker threads ─────────────────────────────────────────────────────────

    def _thread_hms(self, N, D, T, K):
        def cb(it, total, best, elapsed, ops):
            snap = {"time": elapsed, "mem": N*D*8/(1024**2),
                    "best": best, "ops": ops, "iter": it}
            with self._lock:
                self._hms_live = snap
            pct = it / total
            self.after(0, lambda p=pct, s=snap, i=it, tot=total:
                       self._live_update_hms(p, s, i, tot))

        try:
            result = run_hms(N, D, T, K, progress_cb=cb)
        except Exception as ex:
            self.after(0, lambda: self._show_error(str(ex)))
            return

        with self._lock:
            self._hms_result = result
        self.after(0, self._check_both_done)

    def _thread_hmsos(self, N, D, T, K1, K2):
        def cb(it, total, best, elapsed, ops):
            snap = {"time": elapsed, "mem": N*D*8/(1024**2)*1.12,
                    "best": best, "ops": ops, "iter": it}
            with self._lock:
                self._hmsos_live = snap
            pct = it / total
            self.after(0, lambda p=pct, s=snap, i=it, tot=total:
                       self._live_update_hmsos(p, s, i, tot))

        try:
            result = run_hms_os(N, D, T, K1, K2, progress_cb=cb)
        except Exception as ex:
            self.after(0, lambda: self._show_error(str(ex)))
            return

        with self._lock:
            self._hmsos_result = result
        self.after(0, self._check_both_done)

    def _check_both_done(self):
        with self._lock:
            h = self._hms_result
            o = self._hmsos_result
        if h is not None and o is not None:
            self._draw_progress(self.hms_prog_bar,   self.hms_prog_pct,   1.0, ACCENT1)
            self._draw_progress(self.hmsos_prog_bar, self.hmsos_prog_pct, 1.0, ACCENT2)
            self._update_ui(*h, *o)

    # ── Live update helpers ────────────────────────────────────────────────────

    def _live_update_hms(self, pct, snap, it, total):
        self._draw_progress(self.hms_prog_bar, self.hms_prog_pct, pct, ACCENT1)
        self.hms_iter_lbl.config(
            text=f"HMS  iter {it}/{total}  best={snap['best']:.4f}")
        self._refresh_card_live(snap["time"], snap["mem"], snap["best"], snap["ops"],
                                self.hms_vals, ACCENT1)
        with self._lock:
            o = dict(self._hmsos_live)
        self._draw_bars(snap["time"], o["time"], snap["mem"], o["mem"])

    def _live_update_hmsos(self, pct, snap, it, total):
        self._draw_progress(self.hmsos_prog_bar, self.hmsos_prog_pct, pct, ACCENT2)
        self.hmsos_iter_lbl.config(
            text=f"HMS-OS  iter {it}/{total}  best={snap['best']:.4f}")
        self._refresh_card_live(snap["time"], snap["mem"], snap["best"], snap["ops"],
                                self.hmsos_vals, ACCENT2)
        with self._lock:
            h = dict(self._hms_live)
        self._draw_bars(h["time"], snap["time"], h["mem"], snap["mem"])

    def _refresh_card_live(self, t, mem, best, ops, labels, color):
        vals = [t, mem, best, ops]
        for i, lbl in enumerate(labels):
            lbl.config(text=fmt(vals[i]), fg=color)

    # ── Final update ───────────────────────────────────────────────────────────

    def _show_error(self, msg):
        self.status_lbl.config(text=msg, fg="#e05a5a")
        self.run_btn.config(state="normal", text="▶  RUN")

    def _update_ui(self, h_time, h_mem, h_best, h_ops,
                         o_time, o_mem, o_best, o_ops):
        vals_h = [h_time, h_mem, h_best, h_ops]
        vals_o = [o_time, o_mem, o_best, o_ops]

        for i in range(4):
            self.hms_vals[i].config(text=fmt(vals_h[i]))
            self.hmsos_vals[i].config(text=fmt(vals_o[i]))

        for i in range(4):
            if vals_h[i] < vals_o[i]:
                self.hms_vals[i].config(fg=SUCCESS)
                self.hmsos_vals[i].config(fg=ACCENT2)
            elif vals_o[i] < vals_h[i]:
                self.hmsos_vals[i].config(fg=SUCCESS)
                self.hms_vals[i].config(fg=ACCENT1)
            else:
                self.hms_vals[i].config(fg=ACCENT1)
                self.hmsos_vals[i].config(fg=ACCENT2)

        self._draw_bars(h_time, o_time, h_mem, o_mem)

        hms_wins = sum(1 for h, o in zip(vals_h, vals_o) if h < o)
        os_wins  = sum(1 for h, o in zip(vals_h, vals_o) if o < h)
        if os_wins > hms_wins:
            msg, clr = "✦  HMS-OS outperforms HMS overall", ACCENT2
        elif hms_wins > os_wins:
            msg, clr = "✦  HMS outperforms HMS-OS overall", ACCENT1
        else:
            msg, clr = "✦  Both algorithms tied", SUCCESS

        self.winner_lbl.config(text=msg, fg=clr)
        self.hms_iter_lbl.config(text="✔  HMS complete",   fg=SUCCESS)
        self.hmsos_iter_lbl.config(text="✔  HMS-OS complete", fg=SUCCESS)
        self.run_btn.config(state="normal", text="▶  RUN")
        self.status_lbl.config(text="")

    def _on_canvas_resize(self, event):
        """Rebuild static items when the canvas is resized."""
        new_size = (event.width, event.height)
        if new_size != self._chart_size:
            self._chart_items = None   # force recreate
            self._chart_size  = new_size
            # Redraw with last known values if available
            if self._hms_live["time"] or self._hmsos_live["time"]:
                h = self._hms_live
                o = self._hmsos_live
                self._draw_bars(h["time"], o["time"], h["mem"], o["mem"])

    def _init_chart_items(self, W, H, group_w, bar_w, pad_l, pad_r, pad_t, pad_b):
        """Create all canvas items once. Returns dict of item IDs."""
        c = self.canvas
        ax_y_bot = H - pad_b
        ax_y_top = pad_t

        items = {}
        # Axes
        items["ax_v"] = c.create_line(pad_l, ax_y_top, pad_l, ax_y_bot,
                                       fill=BORDER, width=1)
        items["ax_h"] = c.create_line(pad_l, ax_y_bot, W - pad_r, ax_y_bot,
                                       fill=BORDER, width=1)

        group_labels = ["Time (ms)", "Memory (MB)"]
        for g in range(2):
            cx = pad_l + g * group_w + group_w / 2
            pfx = f"g{g}"

            # HMS bar + label
            items[f"{pfx}_hbar"] = c.create_rectangle(0, 0, 0, 0, fill=ACCENT1, outline="")
            items[f"{pfx}_htxt"] = c.create_text(0, 0, text="", font=("Courier New", 10),
                                                  fill=ACCENT1, anchor="s")
            # HMS-OS bar + label
            items[f"{pfx}_obar"] = c.create_rectangle(0, 0, 0, 0, fill=ACCENT2, outline="")
            items[f"{pfx}_otxt"] = c.create_text(0, 0, text="", font=("Courier New", 10),
                                                  fill=ACCENT2, anchor="s")
            # Group label
            items[f"{pfx}_lbl"] = c.create_text(cx, ax_y_bot + 18, text=group_labels[g],
                                                  font=("Courier New", 11), fill=SUBTEXT)

        # Legend
        legend_y = H - 12
        center_x = W / 2
        items["leg_h_rect"] = c.create_rectangle(center_x-90, legend_y-6,
                                                   center_x-78, legend_y+6,
                                                   fill=ACCENT1, outline="")
        items["leg_h_txt"]  = c.create_text(center_x-72, legend_y, text="HMS",
                                             anchor="w", font=("Courier New", 10), fill=ACCENT1)
        items["leg_o_rect"] = c.create_rectangle(center_x+10, legend_y-6,
                                                   center_x+22, legend_y+6,
                                                   fill=ACCENT2, outline="")
        items["leg_o_txt"]  = c.create_text(center_x+28, legend_y, text="HMS-OS",
                                             anchor="w", font=("Courier New", 10), fill=ACCENT2)
        return items

    def _draw_bars(self, h_time, o_time, h_mem, o_mem):
        self.canvas.update_idletasks()
        W = self.canvas.winfo_width()  or 560
        H = self.canvas.winfo_height() or 200

        pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 60
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b
        group_w = plot_w / 2
        bar_w   = group_w * 0.25
        ax_y_bot = H - pad_b

        # Create items on first call or after resize
        if self._chart_items is None:
            self._chart_items = self._init_chart_items(W, H, group_w, bar_w,
                                                        pad_l, pad_r, pad_t, pad_b)
            self._chart_size  = (W, H)

        it = self._chart_items
        c  = self.canvas

        data = [
            (h_time, o_time),
            (h_mem,  o_mem),
        ]

        for g, (hv, ov) in enumerate(data):
            pfx   = f"g{g}"
            max_v = max(hv, ov, 1e-9)
            cx    = pad_l + g * group_w + group_w / 2

            # HMS bar
            bh = (hv / max_v) * plot_h
            x0, x1 = cx - bar_w - 4, cx - 4
            y0 = ax_y_bot - bh
            c.coords(it[f"{pfx}_hbar"], x0, y0, x1, ax_y_bot)
            c.coords(it[f"{pfx}_htxt"], (x0+x1)/2, y0 - 4)
            c.itemconfig(it[f"{pfx}_htxt"], text=fmt(hv))

            # HMS-OS bar
            bh2 = (ov / max_v) * plot_h
            x2, x3 = cx + 4, cx + bar_w + 4
            y2 = ax_y_bot - bh2
            c.coords(it[f"{pfx}_obar"], x2, y2, x3, ax_y_bot)
            c.coords(it[f"{pfx}_otxt"], (x2+x3)/2, y2 - 4)
            c.itemconfig(it[f"{pfx}_otxt"], text=fmt(ov))


if __name__ == "__main__":
    app = App()
    app.mainloop()
