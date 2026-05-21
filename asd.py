import tkinter as tk
from tkinter import ttk
import math
import random
import time
import threading

# ─── Color Palette ────────────────────────────────────────────────────────────
BG        = "#0f1117"
PANEL     = "#1a1d27"
ACCENT1   = "#4f8ef7"   # HMS  – blue
ACCENT2   = "#f7764f"   # HMS-OS – orange
TEXT      = "#e8eaf0"
SUBTEXT   = "#7a7f9a"
BORDER    = "#2a2d3e"
SUCCESS   = "#4fbd7a"
ENTRY_BG  = "#232637"

# ─── Simulated Algorithm Runners ─────────────────────────────────────────────

def run_hms(N, D, T, K):
    """Simulate HMS runtime & memory based on complexity O(T×N×K)."""
    t0 = time.time()
    ops = 0
    best_val = float("inf")
    solutions = [[random.uniform(-5, 5) for _ in range(D)] for _ in range(N)]

    for iteration in range(T):
        # Fitness evaluation
        fitness = [sum(x**2 for x in sol) for sol in solutions]
        best_idx = fitness.index(min(fitness))
        best_val = fitness[best_idx]

        # Mental search (random neighbor)
        for i in range(N):
            neighbor = [solutions[i][d] + random.gauss(0, 1) for d in range(D)]
            if sum(x**2 for x in neighbor) < fitness[i]:
                solutions[i] = neighbor
            ops += D

        # K-means clustering (simplified)
        centroids = random.sample(solutions, min(K, N))
        for _ in range(3):  # 3 clustering steps
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

        # Move toward best
        W = solutions[best_idx]
        for i in range(N):
            solutions[i] = [solutions[i][d] + random.random() * (W[d] - solutions[i][d])
                            for d in range(D)]
            ops += D

    elapsed = (time.time() - t0) * 1000  # ms
    mem = N * D * 8 / (1024**2)          # MB (float64)
    return elapsed, mem, best_val, ops


def run_hms_os(N, D, T, K1, K2):
    """Simulate HMS-OS runtime & memory based on O(T×N×(K1+K2))."""
    t0 = time.time()
    ops = 0
    best_val = float("inf")
    solutions = [[random.uniform(-5, 5) for _ in range(D)] for _ in range(N)]

    for iteration in range(T):
        fitness = [sum(x**2 for x in sol) for sol in solutions]
        best_idx = fitness.index(min(fitness))
        best_val = fitness[best_idx]

        # Adaptive mental search (bias toward better half)
        ranked = sorted(range(N), key=lambda i: fitness[i])
        for rank, i in enumerate(ranked):
            scale = 0.5 if rank < N // 2 else 1.5
            neighbor = [solutions[i][d] + random.gauss(0, scale) for d in range(D)]
            if sum(x**2 for x in neighbor) < fitness[i]:
                solutions[i] = neighbor
            ops += D

        # Dual clustering: search space
        c1 = random.sample(solutions, min(K1, N))
        for _ in range(3):
            cl1 = [[] for _ in range(K1)]
            for sol in solutions:
                dists = [sum((sol[d]-c[d])**2 for d in range(D)) for c in c1]
                cl1[dists.index(min(dists))].append(sol)
                ops += D * K1
            c1 = [[sum(s[d] for s in cl) / len(cl) if cl else c1[k][d]
                   for d in range(D)] for k, cl in enumerate(cl1)]

        # Dual clustering: objective space
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

        # Move toward best solution W AND centroid C
        W = solutions[best_idx]
        best_cluster = min(range(K1), key=lambda k: len(cl1[k]) == 0 or
                           sum(sum(s[d]**2 for d in range(D)) for s in cl1[k]) / max(len(cl1[k]),1))
        C = c1[best_cluster]
        for i in range(N):
            r1, r2 = random.random(), random.random()
            solutions[i] = [
                solutions[i][d] + r1*(W[d]-solutions[i][d]) + r2*(C[d]-solutions[i][d])
                for d in range(D)]
            ops += D

    elapsed = (time.time() - t0) * 1000
    mem = N * D * 8 / (1024**2) * 1.12   # ~12% more for dual structures
    return elapsed, mem, best_val, ops


# ─── Main Application ─────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HMS vs HMS-OS")
        self.geometry("820x620")
        self.resizable(False, False)
        self.configure(bg=BG)
        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        title_fr = tk.Frame(self, bg=BG)
        title_fr.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(title_fr, text="HMS vs HMS-OS", font=("Courier New", 20, "bold"),
                 fg=TEXT, bg=BG).pack(side="left")

        tk.Label(self, text="Human Mental Search  ·  Dual Clustering Variant",
                 font=("Courier New", 9), fg=SUBTEXT, bg=BG).pack(anchor="w", padx=26)

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x", padx=24, pady=(10, 0))

        # Body frame
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=12)

        # Left: parameters
        left = tk.Frame(body, bg=PANEL, bd=0, relief="flat",
                        highlightthickness=1, highlightbackground=BORDER)
        left.pack(side="left", fill="y", ipadx=14, ipady=14, padx=(0, 10))

        tk.Label(left, text="PARAMETERS", font=("Courier New", 9, "bold"),
                 fg=ACCENT1, bg=PANEL).grid(row=0, column=0, columnspan=2,
                                            sticky="w", padx=14, pady=(10, 8))

        params = [
            ("Solutions  (N)", "50", "10–200"),
            ("Dimensions (D)", "50", "10–200"),
            ("Iterations  (T)", "100", "10–500"),
            ("Clusters K / K₁", "3",  "2–10"),
            ("Clusters    K₂", "3",  "2–10  (HMS-OS only)"),
        ]
        self.entries = []
        for r, (lbl, default, hint) in enumerate(params, start=1):
            tk.Label(left, text=lbl, font=("Courier New", 9), fg=TEXT, bg=PANEL,
                     anchor="w").grid(row=r, column=0, sticky="w", padx=14, pady=5)
            e = tk.Entry(left, width=6, font=("Courier New", 10), bg=ENTRY_BG,
                         fg=TEXT, insertbackground=TEXT, relief="flat",
                         highlightthickness=1, highlightcolor=ACCENT1,
                         highlightbackground=BORDER)
            e.insert(0, default)
            e.grid(row=r, column=1, padx=(4, 14), pady=5, sticky="w")
            tk.Label(left, text=hint, font=("Courier New", 7), fg=SUBTEXT,
                     bg=PANEL).grid(row=r, column=2, padx=(0, 10), sticky="w")
            self.entries.append(e)

        # Run button
        self.run_btn = tk.Button(
            left, text="▶  RUN", font=("Courier New", 10, "bold"),
            bg=ACCENT1, fg="#fff", activebackground="#3a6fd4",
            relief="flat", cursor="hand2", padx=20, pady=6,
            command=self._start_run)
        self.run_btn.grid(row=len(params)+1, column=0, columnspan=3,
                          padx=14, pady=(16, 6), sticky="ew")

        self.status_lbl = tk.Label(left, text="", font=("Courier New", 8),
                                   fg=SUBTEXT, bg=PANEL)
        self.status_lbl.grid(row=len(params)+2, column=0, columnspan=3, padx=14)

        # Right: results
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Metric cards
        cards_fr = tk.Frame(right, bg=BG)
        cards_fr.pack(fill="x")

        metrics = ["Execution Time (ms)", "Memory Usage (MB)",
                   "Best Solution Value", "Total Operations"]
        self.hms_vals   = []
        self.hmsos_vals = []

        for col, metric in enumerate(metrics):
            card = tk.Frame(cards_fr, bg=PANEL, highlightthickness=1,
                            highlightbackground=BORDER)
            card.grid(row=0, column=col, padx=4, pady=4, sticky="nsew", ipadx=8, ipady=8)
            cards_fr.columnconfigure(col, weight=1)

            tk.Label(card, text=metric, font=("Courier New", 7), fg=SUBTEXT,
                     bg=PANEL, wraplength=120, justify="center").pack(pady=(8, 4))

            h_lbl = tk.Label(card, text="—", font=("Courier New", 13, "bold"),
                             fg=ACCENT1, bg=PANEL)
            h_lbl.pack()
            tk.Label(card, text="HMS", font=("Courier New", 7), fg=ACCENT1,
                     bg=PANEL).pack()

            o_lbl = tk.Label(card, text="—", font=("Courier New", 13, "bold"),
                             fg=ACCENT2, bg=PANEL)
            o_lbl.pack(pady=(6, 0))
            tk.Label(card, text="HMS-OS", font=("Courier New", 7), fg=ACCENT2,
                     bg=PANEL).pack(pady=(0, 8))

            self.hms_vals.append(h_lbl)
            self.hmsos_vals.append(o_lbl)

        # Bar chart canvas
        chart_wrap = tk.Frame(right, bg=PANEL, highlightthickness=1,
                              highlightbackground=BORDER)
        chart_wrap.pack(fill="both", expand=True, pady=(8, 0))
        tk.Label(chart_wrap, text="Runtime & Memory Comparison",
                 font=("Courier New", 8, "bold"), fg=SUBTEXT, bg=PANEL).pack(pady=(8, 0))

        self.canvas = tk.Canvas(chart_wrap, bg=PANEL, bd=0,
                                highlightthickness=0, height=220)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=(4, 12))

        # Winner banner
        self.winner_fr = tk.Frame(right, bg=BG)
        self.winner_fr.pack(fill="x", pady=(8, 0))
        self.winner_lbl = tk.Label(self.winner_fr, text="",
                                   font=("Courier New", 9, "bold"),
                                   fg=SUCCESS, bg=BG)
        self.winner_lbl.pack()

    # ── Run Logic ──────────────────────────────────────────────────────────────

    def _start_run(self):
        self.run_btn.config(state="disabled", text="running…")
        self.status_lbl.config(text="")
        self.winner_lbl.config(text="")
        self.canvas.delete("all")
        for lbl in self.hms_vals + self.hmsos_vals:
            lbl.config(text="…")
        threading.Thread(target=self._run_algorithms, daemon=True).start()

    def _run_algorithms(self):
        try:
            N  = int(self.entries[0].get())
            D  = int(self.entries[1].get())
            T  = int(self.entries[2].get())
            K  = int(self.entries[3].get())
            K2 = int(self.entries[4].get())
        except ValueError:
            self.after(0, lambda: self._show_error("Invalid input — use integers only."))
            return

        try:
            h_time, h_mem, h_best, h_ops   = run_hms(N, D, T, K)
            o_time, o_mem, o_best, o_ops   = run_hms_os(N, D, T, K, K2)
        except Exception as ex:
            self.after(0, lambda: self._show_error(str(ex)))
            return

        self.after(0, lambda: self._update_ui(
            h_time, h_mem, h_best, h_ops,
            o_time, o_mem, o_best, o_ops))

    def _show_error(self, msg):
        self.status_lbl.config(text=msg, fg="#e05a5a")
        self.run_btn.config(state="normal", text="▶  RUN")

    def _update_ui(self, h_time, h_mem, h_best, h_ops,
                         o_time, o_mem, o_best, o_ops):
        # Format helpers
        def fmt(v):
            if abs(v) >= 1e6:  return f"{v:.2e}"
            if abs(v) >= 1000: return f"{v:,.0f}"
            return f"{v:.2f}"

        vals_h = [h_time, h_mem, h_best, h_ops]
        vals_o = [o_time, o_mem, o_best, o_ops]

        for i in range(4):
            self.hms_vals[i].config(text=fmt(vals_h[i]))
            self.hmsos_vals[i].config(text=fmt(vals_o[i]))

        # Highlight winner per metric (lower = better)
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

        # Draw bar chart (Time + Memory side by side)
        self._draw_bars(h_time, o_time, h_mem, o_mem)

        # Winner summary
        hms_wins = sum(1 for h, o in zip(vals_h, vals_o) if h < o)
        os_wins  = sum(1 for h, o in zip(vals_h, vals_o) if o < h)
        if os_wins > hms_wins:
            msg = "✦  HMS-OS outperforms HMS overall"
            clr = ACCENT2
        elif hms_wins > os_wins:
            msg = "✦  HMS outperforms HMS-OS overall"
            clr = ACCENT1
        else:
            msg = "✦  Both algorithms tied"
            clr = SUCCESS
        self.winner_lbl.config(text=msg, fg=clr)

        self.run_btn.config(state="normal", text="▶  RUN")
        self.status_lbl.config(text="")

    def _draw_bars(self, h_time, o_time, h_mem, o_mem):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        W = self.canvas.winfo_width()  or 560
        H = self.canvas.winfo_height() or 200

        pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 40
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b

        groups = [
            ("Time (ms)", h_time, o_time),
            ("Memory (MB)", h_mem, o_mem),
        ]
        n_groups = len(groups)
        group_w  = plot_w / n_groups
        bar_w    = group_w * 0.25

        # Axis
        ax_x = pad_l;  ax_y_top = pad_t;  ax_y_bot = H - pad_b
        self.canvas.create_line(ax_x, ax_y_top, ax_x, ax_y_bot,
                                fill=BORDER, width=1)
        self.canvas.create_line(ax_x, ax_y_bot, W - pad_r, ax_y_bot,
                                fill=BORDER, width=1)

        for g, (label, hv, ov) in enumerate(groups):
            max_v = max(hv, ov, 1e-9)
            cx = pad_l + g * group_w + group_w / 2

            # HMS bar
            bh = (hv / max_v) * plot_h
            x0 = cx - bar_w - 4
            x1 = cx - 4
            y0 = ax_y_bot - bh
            self.canvas.create_rectangle(x0, y0, x1, ax_y_bot,
                                         fill=ACCENT1, outline="")
            self.canvas.create_text((x0+x1)//2, y0 - 6,
                                    text=f"{hv:.1f}", font=("Courier New", 7),
                                    fill=ACCENT1)

            # HMS-OS bar
            bh2 = (ov / max_v) * plot_h
            x2 = cx + 4
            x3 = cx + bar_w + 4
            y2 = ax_y_bot - bh2
            self.canvas.create_rectangle(x2, y2, x3, ax_y_bot,
                                         fill=ACCENT2, outline="")
            self.canvas.create_text((x2+x3)//2, y2 - 6,
                                    text=f"{ov:.1f}", font=("Courier New", 7),
                                    fill=ACCENT2)

            # Group label
            self.canvas.create_text(cx, ax_y_bot + 14, text=label,
                                    font=("Courier New", 8), fill=SUBTEXT)

        # Legend
        lx = W - pad_r - 120
        self.canvas.create_rectangle(lx, pad_t, lx+10, pad_t+10,
                                     fill=ACCENT1, outline="")
        self.canvas.create_text(lx+14, pad_t+5, text="HMS",
                                anchor="w", font=("Courier New", 8), fill=ACCENT1)
        self.canvas.create_rectangle(lx, pad_t+16, lx+10, pad_t+26,
                                     fill=ACCENT2, outline="")
        self.canvas.create_text(lx+14, pad_t+21, text="HMS-OS",
                                anchor="w", font=("Courier New", 8), fill=ACCENT2)


if __name__ == "__main__":
    app = App()
    app.mainloop()
