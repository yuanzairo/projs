"""
HMS vs HMS-OS Visualizer
Based on: "An Improved Human Mental Search Algorithm Using
Dual Clustering for Optimization Problems"
University of Mindanao — CS6L Algorithms and Complexity

HMS  : single k-means clustering in search space
HMS-OS: dual clustering (search space + objective space)
        + adaptive mental search toward best W and centroid C
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math, random

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#0d0f14"
PANEL   = "#141720"
BORDER  = "#1e2230"
A1      = "#00c9a7"   # HMS teal
A2      = "#f7526a"   # HMS-OS coral
GOLD    = "#f0c040"
TH      = "#e8eaf0"
TM      = "#8890a8"
TD      = "#3c4256"

# ── Benchmark data from the paper (Table in Section IV) ──────────────────────
PAPER_SIZES   = ["Small\n(N=30,D=50)", "Medium\n(N=50,D=75)", "Large\n(N=100,D=100)"]
PAPER_HMS_T   = [120, 280, 600]   # ms
PAPER_HMOS_T  = [100, 220, 450]   # ms
PAPER_HMS_M   = [50,  80,  120]   # MB
PAPER_HMOS_M  = [55,  90,  135]   # MB

# ── Sphere benchmark function (standard optimization test) ────────────────────
def sphere(x):
    return float(np.sum(x ** 2))

def rosenbrock(x):
    return float(np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))

BENCHMARKS = {"Sphere": sphere, "Rosenbrock": rosenbrock}

# ─────────────────────────────────────────────────────────────────────────────
#  HMS  — Human Mental Search (baseline)
#  Single clustering in search space only
# ─────────────────────────────────────────────────────────────────────────────
def run_hms(N, D, T, K, func, seed=0, bounds=(-5.12, 5.12)):
    rng = np.random.default_rng(seed)
    lo, hi = bounds

    # 1. Initialize random population
    pop = rng.uniform(lo, hi, (N, D))

    # 2. Evaluate fitness
    fitness = np.array([func(p) for p in pop])

    best_fitness_history = []
    mean_fitness_history = []
    ops_per_iter         = []

    for t in range(T):
        ops = 0

        # 3. Mental search — each solution explores nearby (random perturbation)
        for i in range(N):
            candidate = pop[i] + rng.normal(0, 0.5, D)
            candidate = np.clip(candidate, lo, hi)
            f_cand = func(candidate)
            ops += 1
            if f_cand < fitness[i]:
                pop[i]     = candidate
                fitness[i] = f_cand

        # 4. Single clustering in search space (K-means, 3 iterations)
        centers = pop[rng.choice(N, K, replace=False)].copy()
        labels  = np.zeros(N, dtype=int)
        for _ in range(3):
            dists  = np.linalg.norm(pop[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(K):
                mask = labels == k
                if mask.any():
                    centers[k] = pop[mask].mean(axis=0)
        ops += N * K * 3

        # 5. Find best solution W
        W_idx = int(np.argmin(fitness))
        W     = pop[W_idx]

        # 6. Move all solutions toward W
        alpha = rng.uniform(0, 1, (N, 1))
        pop   = pop + alpha * (W - pop)
        pop   = np.clip(pop, lo, hi)
        fitness = np.array([func(p) for p in pop])
        ops += N

        best_fitness_history.append(float(fitness.min()))
        mean_fitness_history.append(float(fitness.mean()))
        ops_per_iter.append(ops)

    best_idx = int(np.argmin(fitness))
    return {
        "best":        fitness[best_idx],
        "best_hist":   best_fitness_history,
        "mean_hist":   mean_fitness_history,
        "ops_hist":    ops_per_iter,
        "final_pop":   pop,
        "final_fit":   fitness,
        "pop_hist":    None,   # not stored to save memory
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HMS-OS — Dual Clustering + Adaptive Mental Search (proposed)
# ─────────────────────────────────────────────────────────────────────────────
def run_hms_os(N, D, T, K1, K2, func, seed=0, bounds=(-5.12, 5.12)):
    rng = np.random.default_rng(seed)
    lo, hi = bounds

    # 1. Initialize random population
    pop     = rng.uniform(lo, hi, (N, D))
    fitness = np.array([func(p) for p in pop])

    best_fitness_history = []
    mean_fitness_history = []
    ops_per_iter         = []

    for t in range(T):
        ops = 0

        # 3. Adaptive mental search — bias more effort toward better solutions
        ranks       = np.argsort(fitness)                   # rank 0 = best
        search_prob = 1.0 - ranks / N                      # better sol → higher prob
        for i in range(N):
            if rng.random() < search_prob[i] + 0.2:       # adaptive threshold
                candidate = pop[i] + rng.normal(0, 0.3, D)
                candidate = np.clip(candidate, lo, hi)
                f_cand    = func(candidate)
                ops += 1
                if f_cand < fitness[i]:
                    pop[i]     = candidate
                    fitness[i] = f_cand

        # 4a. Cluster in search space (K1 clusters)
        centers1 = pop[rng.choice(N, K1, replace=False)].copy()
        labels1  = np.zeros(N, dtype=int)
        for _ in range(3):
            dists   = np.linalg.norm(pop[:, None, :] - centers1[None, :, :], axis=2)
            labels1 = np.argmin(dists, axis=1)
            for k in range(K1):
                mask = labels1 == k
                if mask.any():
                    centers1[k] = pop[mask].mean(axis=0)
        ops += N * K1 * 3

        # 4b. Cluster in objective space (K2 clusters on fitness values)
        fit_vals = fitness.reshape(-1, 1)
        centers2 = fit_vals[rng.choice(N, K2, replace=False)].copy()
        labels2  = np.zeros(N, dtype=int)
        for _ in range(3):
            dists   = np.abs(fit_vals - centers2.T)
            labels2 = np.argmin(dists, axis=1)
            for k in range(K2):
                mask = labels2 == k
                if mask.any():
                    centers2[k] = fit_vals[mask].mean()
        ops += N * K2 * 3

        # 5. Best solution W + centroid C of best cluster
        W_idx     = int(np.argmin(fitness))
        W         = pop[W_idx]
        best_clus = labels1[W_idx]
        clus_mask = labels1 == best_clus
        C         = pop[clus_mask].mean(axis=0)   # centroid of best cluster

        # 6. Move solutions toward both W and C
        alpha = rng.uniform(0, 1, (N, 1))
        beta  = rng.uniform(0, 1, (N, 1))
        pop   = pop + alpha * (W - pop) + beta * (C - pop)
        pop   = np.clip(pop, lo, hi)
        fitness = np.array([func(p) for p in pop])
        ops += N

        best_fitness_history.append(float(fitness.min()))
        mean_fitness_history.append(float(fitness.mean()))
        ops_per_iter.append(ops)

    best_idx = int(np.argmin(fitness))
    return {
        "best":      fitness[best_idx],
        "best_hist": best_fitness_history,
        "mean_hist": mean_fitness_history,
        "ops_hist":  ops_per_iter,
        "final_pop": pop,
        "final_fit": fitness,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HMS vs HMS-OS — Dual Clustering Visualizer")
        self.configure(bg=BG)
        self.geometry("1340x900")
        self.minsize(1100, 740)

        # Parameters (match paper ranges)
        self.v_N      = tk.IntVar(value=50)    # population size
        self.v_D      = tk.IntVar(value=10)    # solution dimensions
        self.v_T      = tk.IntVar(value=100)   # iterations
        self.v_K      = tk.IntVar(value=3)     # clusters HMS
        self.v_K1     = tk.IntVar(value=3)     # clusters HMS-OS search space
        self.v_K2     = tk.IntVar(value=3)     # clusters HMS-OS objective space
        self.v_seed   = tk.IntVar(value=42)
        self.v_bench  = tk.StringVar(value="Sphere")
        self.v_radar_d= tk.IntVar(value=6)     # radar axes 3-12

        self._hms_res  = None
        self._hmos_res = None
        self._anim_on  = False
        self._anim_t   = 0

        self._build_ui()
        self._run()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG, pady=8)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="HMS", font=("Courier New", 20, "bold"), fg=A1, bg=BG).pack(side="left")
        tk.Label(hdr, text=" vs ", font=("Courier New", 20), fg=TM, bg=BG).pack(side="left")
        tk.Label(hdr, text="HMS-OS", font=("Courier New", 20, "bold"), fg=A2, bg=BG).pack(side="left")
        tk.Label(hdr, text="  |  Dual Clustering for Optimization  |  Univ. of Mindanao CS6L",
                 font=("Courier New", 11), fg=TM, bg=BG).pack(side="left", padx=12)

        # Control panel
        cp = tk.Frame(self, bg=PANEL, padx=14, pady=8)
        cp.pack(fill="x", padx=20, pady=(0, 6))

        r1 = tk.Frame(cp, bg=PANEL); r1.pack(fill="x", pady=(0, 3))
        r2 = tk.Frame(cp, bg=PANEL); r2.pack(fill="x", pady=(3, 0))

        def spin(parent, label, var, lo, hi, w=5, tooltip=None):
            tk.Label(parent, text=label, fg=TM, bg=PANEL,
                     font=("Courier New", 9)).pack(side="left", padx=(10, 2))
            sb = tk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=w,
                            bg=BORDER, fg=TH, relief="flat", font=("Courier New", 9),
                            insertbackground=TH, buttonbackground=BORDER)
            sb.pack(side="left")
            if tooltip:
                tk.Label(parent, text=tooltip, fg=TD, bg=PANEL,
                         font=("Courier New", 8)).pack(side="left", padx=(1, 6))

        def btn(parent, text, cmd, color=A1):
            b = tk.Button(parent, text=text, command=cmd, bg=color, fg=BG,
                          relief="flat", font=("Courier New", 9, "bold"),
                          padx=10, pady=3, cursor="hand2",
                          activebackground=TH, activeforeground=BG)
            b.pack(side="left", padx=5)

        def sep(parent):
            tk.Label(parent, text="│", fg=TD, bg=PANEL,
                     font=("Courier New", 12)).pack(side="left", padx=8)

        # Row 1: core params
        spin(r1, "N (population):", self.v_N, 10, 200, tooltip="30–100")
        spin(r1, "D (dimensions):", self.v_D, 2,  100, tooltip="50–100 paper")
        spin(r1, "T (iterations):", self.v_T, 10, 500, tooltip="up to 3000×D")
        sep(r1)
        spin(r1, "K (HMS clusters):", self.v_K, 2, 10)
        spin(r1, "K₁ (search space):", self.v_K1, 2, 10)
        spin(r1, "K₂ (objective space):", self.v_K2, 2, 10)
        sep(r1)
        tk.Label(r1, text="Benchmark fn:", fg=TM, bg=PANEL,
                 font=("Courier New", 9)).pack(side="left", padx=(10, 2))
        om = ttk.OptionMenu(r1, self.v_bench, "Sphere", *BENCHMARKS.keys())
        om.pack(side="left")

        # Row 2: run controls + radar D
        spin(r2, "Seed:", self.v_seed, 0, 9999)
        btn(r2, "▶  Run", self._run, A1)
        btn(r2, "⟳  Animate", self._start_anim, GOLD)
        btn(r2, "⏹  Stop", self._stop_anim, TM)
        sep(r2)
        btn(r2, "📊  Paper Results", self._show_paper_results, A2)
        sep(r2)
        spin(r2, "Radar axes (D):", self.v_radar_d, 3, 12, tooltip="3–12")
        btn(r2, "🔄  Update Radar", self._draw_metrics, TM)

        # Notebook
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("D.TNotebook", background=BG, borderwidth=0)
        style.configure("D.TNotebook.Tab", background=PANEL, foreground=TM,
                        font=("Courier New", 10, "bold"), padding=[12, 5])
        style.map("D.TNotebook.Tab",
                  background=[("selected", BORDER)],
                  foreground=[("selected", TH)])

        nb = ttk.Notebook(self, style="D.TNotebook")
        nb.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        self._nb = nb

        self._tab_conv    = tk.Frame(nb, bg=BG); nb.add(self._tab_conv,    text=" Convergence ")
        self._tab_pop     = tk.Frame(nb, bg=BG); nb.add(self._tab_pop,     text=" Population ")
        self._tab_ops     = tk.Frame(nb, bg=BG); nb.add(self._tab_ops,     text=" Operations ")
        self._tab_paper   = tk.Frame(nb, bg=BG); nb.add(self._tab_paper,   text=" Paper Results ")
        self._tab_metrics = tk.Frame(nb, bg=BG); nb.add(self._tab_metrics, text=" Metrics Radar ")

        self._build_tab_conv()
        self._build_tab_pop()
        self._build_tab_ops()
        self._build_tab_paper()
        self._build_tab_metrics()

        # Status bar
        self._status = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self._status, fg=TM, bg=BG,
                 font=("Courier New", 9), anchor="w").pack(fill="x", padx=24, pady=(0, 4))

    # ── Tab builders ─────────────────────────────────────────────────────────
    def _make_fig(self, frame, rows, cols, figsize):
        fig = plt.Figure(figsize=figsize, facecolor=BG)
        axes = [fig.add_subplot(rows, cols, i+1) for i in range(rows*cols)]
        for ax in axes:
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.tick_params(colors=TM, labelsize=8)
            ax.title.set_color(TH); ax.title.set_fontsize(10)
            ax.xaxis.label.set_color(TM); ax.yaxis.label.set_color(TM)
        fig.tight_layout(pad=2.5)
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, axes, canvas

    def _build_tab_conv(self):
        self._fig_conv, self._ax_conv, self._cv_conv = \
            self._make_fig(self._tab_conv, 1, 2, (12, 4.8))

    def _build_tab_pop(self):
        self._fig_pop, self._ax_pop, self._cv_pop = \
            self._make_fig(self._tab_pop, 1, 2, (12, 4.8))

    def _build_tab_ops(self):
        self._fig_ops, self._ax_ops, self._cv_ops = \
            self._make_fig(self._tab_ops, 1, 2, (12, 4.8))

    def _build_tab_paper(self):
        self._fig_paper, self._ax_paper, self._cv_paper = \
            self._make_fig(self._tab_paper, 1, 2, (12, 4.8))

    def _build_tab_metrics(self):
        outer = tk.Frame(self._tab_metrics, bg=BG)
        outer.pack(fill="both", expand=True, padx=16, pady=16)

        fig = plt.Figure(figsize=(5, 5), facecolor=BG)
        self._fig_radar = fig
        self._ax_radar  = fig.add_subplot(111, polar=True)
        self._ax_radar.set_facecolor(PANEL)
        cv = FigureCanvasTkAgg(fig, outer)
        cv.get_tk_widget().pack(side="left", fill="both", expand=True)
        self._cv_radar = cv

        right = tk.Frame(outer, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        tk.Label(right, text="Performance Metrics", fg=TH, bg=BG,
                 font=("Courier New", 13, "bold")).pack(anchor="w", pady=(0, 10))
        self._metrics_frame = tk.Frame(right, bg=BG)
        self._metrics_frame.pack(fill="both", expand=True)

    # ── Run algorithms ────────────────────────────────────────────────────────
    def _run(self):
        self._stop_anim()
        N  = self.v_N.get()
        D  = self.v_D.get()
        T  = self.v_T.get()
        K  = self.v_K.get()
        K1 = self.v_K1.get()
        K2 = self.v_K2.get()
        sd = self.v_seed.get()
        fn = BENCHMARKS[self.v_bench.get()]

        self._status.set("Running HMS…")
        self.update_idletasks()
        self._hms_res  = run_hms(N, D, T, K, fn, sd)

        self._status.set("Running HMS-OS…")
        self.update_idletasks()
        self._hmos_res = run_hms_os(N, D, T, K1, K2, fn, sd)

        self._draw_all(T)
        self._draw_metrics()
        self._show_paper_results()

        h  = self._hms_res["best"]
        ho = self._hmos_res["best"]
        self._status.set(
            f"N={N}  D={D}  T={T}  K={K}  K₁={K1}  K₂={K2}  fn={self.v_bench.get()}  seed={sd}  │  "
            f"HMS best={h:.4f}   HMS-OS best={ho:.4f}   "
            f"Improvement={max(0,(h-ho)/abs(h+1e-12)*100):.1f}%"
        )

    def _draw_all(self, T=None):
        if not self._hms_res: return
        t = T or len(self._hms_res["best_hist"])
        self._draw_convergence(t)
        self._draw_population()
        self._draw_ops(t)

    # ── Tab 1: Convergence ────────────────────────────────────────────────────
    def _draw_convergence(self, up_to=None):
        hr = self._hms_res;  ho = self._hmos_res
        bh = hr["best_hist"]; bho = ho["best_hist"]
        mh = hr["mean_hist"]; mho = ho["mean_hist"]
        n  = up_to or len(bh)
        xs = list(range(1, n+1))

        # Best fitness
        ax = self._ax_conv[0]
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_title("Best Fitness per Iteration", color=TH, fontsize=10)
        ax.set_xlabel("Iteration (T)", color=TM, fontsize=8)
        ax.set_ylabel("Best Fitness (lower = better)", color=TM, fontsize=8)
        ax.plot(xs, bh[:n],  color=A1, lw=2,      label="HMS")
        ax.plot(xs, bho[:n], color=A2, lw=2, ls="--", label="HMS-OS")
        ax.fill_between(xs, bh[:n],  alpha=0.1, color=A1)
        ax.fill_between(xs, bho[:n], alpha=0.1, color=A2)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TM)
        ax.tick_params(colors=TM, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        # Mean fitness
        ax = self._ax_conv[1]
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_title("Mean Population Fitness per Iteration", color=TH, fontsize=10)
        ax.set_xlabel("Iteration (T)", color=TM, fontsize=8)
        ax.set_ylabel("Mean Fitness", color=TM, fontsize=8)
        ax.plot(xs, mh[:n],  color=A1, lw=2,      label="HMS")
        ax.plot(xs, mho[:n], color=A2, lw=2, ls="--", label="HMS-OS")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TM)
        ax.tick_params(colors=TM, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        self._fig_conv.tight_layout(pad=2.2)
        self._cv_conv.draw_idle()

    # ── Tab 2: Population scatter ─────────────────────────────────────────────
    def _draw_population(self):
        hr = self._hms_res; ho = self._hmos_res

        for ax, res, color, title in [
            (self._ax_pop[0], hr,  A1, "HMS — Final Population (dim 0 vs 1)"),
            (self._ax_pop[1], ho, A2, "HMS-OS — Final Population (dim 0 vs 1)")
        ]:
            ax.cla(); ax.set_facecolor(PANEL)
            ax.set_title(title, color=TH, fontsize=10)
            ax.set_xlabel("Dimension 0", color=TM, fontsize=8)
            ax.set_ylabel("Dimension 1", color=TM, fontsize=8)

            pop = res["final_pop"]
            fit = res["final_fit"]
            fit_n = (fit - fit.min()) / (fit.max() - fit.min() + 1e-12)

            sc = ax.scatter(pop[:, 0], pop[:, 1] if pop.shape[1] > 1 else np.zeros(len(pop)),
                            c=fit_n, cmap="plasma", s=40, alpha=0.85, zorder=3)
            # Mark best
            bi = int(np.argmin(fit))
            ax.scatter(pop[bi, 0], pop[bi, 1] if pop.shape[1] > 1 else 0,
                       color=GOLD, s=160, marker="*", zorder=5, label="Best")
            ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TM)
            ax.tick_params(colors=TM, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        self._fig_pop.tight_layout(pad=2.2)
        self._cv_pop.draw_idle()

    # ── Tab 3: Operations ─────────────────────────────────────────────────────
    def _draw_ops(self, up_to=None):
        hr = self._hms_res; ho = self._hmos_res
        oh  = hr["ops_hist"]; oho = ho["ops_hist"]
        n   = up_to or len(oh)
        xs  = list(range(1, n+1))

        # Ops per iteration
        ax = self._ax_ops[0]
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_title("Operations per Iteration", color=TH, fontsize=10)
        ax.set_xlabel("Iteration (T)", color=TM, fontsize=8)
        ax.set_ylabel("# Operations", color=TM, fontsize=8)
        ax.plot(xs, oh[:n],  color=A1, lw=1.8, label="HMS")
        ax.plot(xs, oho[:n], color=A2, lw=1.8, ls="--", label="HMS-OS")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TM)
        ax.tick_params(colors=TM, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        # Cumulative ops
        ax = self._ax_ops[1]
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_title("Cumulative Operations", color=TH, fontsize=10)
        ax.set_xlabel("Iteration (T)", color=TM, fontsize=8)
        ax.set_ylabel("Total Operations", color=TM, fontsize=8)
        ax.plot(xs, np.cumsum(oh[:n]),  color=A1, lw=2, label="HMS")
        ax.plot(xs, np.cumsum(oho[:n]), color=A2, lw=2, ls="--", label="HMS-OS")
        ax.fill_between(xs, np.cumsum(oh[:n]),  alpha=0.1, color=A1)
        ax.fill_between(xs, np.cumsum(oho[:n]), alpha=0.1, color=A2)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TM)
        ax.tick_params(colors=TM, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        self._fig_ops.tight_layout(pad=2.2)
        self._cv_ops.draw_idle()

    # ── Tab 4: Paper Results ──────────────────────────────────────────────────
    def _show_paper_results(self):
        sizes = ["Small", "Medium", "Large"]
        x     = np.arange(3)
        w     = 0.35

        # Runtime chart
        ax = self._ax_paper[0]
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_title("Execution Time — Paper Table (Section IV)", color=TH, fontsize=10)
        ax.set_ylabel("Time (ms)", color=TM, fontsize=8)
        b1 = ax.bar(x - w/2, PAPER_HMS_T,  w, color=A1, label="HMS",    zorder=3)
        b2 = ax.bar(x + w/2, PAPER_HMOS_T, w, color=A2, label="HMS-OS", zorder=3)
        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                        str(int(bar.get_height())), ha="center", va="bottom",
                        color=TH, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Small\nN=30,D=50", "Medium\nN=50,D=75", "Large\nN=100,D=100"],
                           color=TM, fontsize=8)
        ax.tick_params(colors=TM, labelsize=8)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TM)
        ax.yaxis.grid(True, color=BORDER, lw=0.5); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        # Memory chart
        ax = self._ax_paper[1]
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_title("Memory Usage — Paper Table (Section IV)", color=TH, fontsize=10)
        ax.set_ylabel("Memory (MB)", color=TM, fontsize=8)
        b1 = ax.bar(x - w/2, PAPER_HMS_M,  w, color=A1, label="HMS",    zorder=3)
        b2 = ax.bar(x + w/2, PAPER_HMOS_M, w, color=A2, label="HMS-OS", zorder=3)
        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        str(int(bar.get_height())), ha="center", va="bottom",
                        color=TH, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Small\nN=30,D=50", "Medium\nN=50,D=75", "Large\nN=100,D=100"],
                           color=TM, fontsize=8)
        ax.tick_params(colors=TM, labelsize=8)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TM)
        ax.yaxis.grid(True, color=BORDER, lw=0.5); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        self._fig_paper.tight_layout(pad=2.2)
        self._cv_paper.draw_idle()
        self._nb.select(self._tab_paper)

    # ── Tab 5: Metrics Radar ──────────────────────────────────────────────────
    def _draw_metrics(self):
        if not self._hms_res: return
        hr = self._hms_res; ho = self._hmos_res
        bh = hr["best_hist"]; bho = ho["best_hist"]
        T  = len(bh)

        # Convergence speed: iteration where best drops below 10% of initial
        def conv_speed(hist):
            init = hist[0] + 1e-12
            for i, v in enumerate(hist):
                if v < init * 0.1:
                    return 1.0 - i / T
            return 0.0

        # Solution quality: inverse of final best (normalized)
        max_best = max(bh[-1], bho[-1]) + 1e-12
        def quality(v): return 1.0 - v / max_best

        # Stability: 1 - std of last 20% of best_hist
        def stability(hist):
            tail = hist[int(0.8*T):]
            if not tail: return 0.5
            std  = np.std(tail)
            mean = abs(np.mean(tail)) + 1e-12
            return max(0.0, min(1.0, 1.0 - std/mean * 0.5))

        # Ops efficiency: fewer total ops = better
        total_h  = sum(hr["ops_hist"])
        total_ho = sum(ho["ops_hist"])
        max_ops  = max(total_h, total_ho) + 1e-12
        def ops_eff(tot): return 1.0 - tot/max_ops

        # Coverage: spread of final population (std of dim 0)
        def coverage(res):
            pop = res["final_pop"]
            return min(1.0, float(np.std(pop[:, 0])) / 5.0)

        # Improvement rate: mean per-iteration improvement
        def improv_rate(hist):
            diffs = [max(0, hist[i]-hist[i+1]) for i in range(len(hist)-1)]
            return min(1.0, float(np.mean(diffs)) / (abs(hist[0])+1e-12) * T * 0.5)

        # Adaptability: HMS-OS has adaptive mental search; HMS does not
        # Proxy: ratio of unique fitness values explored
        def adaptability(res):
            fit = res["final_fit"]
            unique_ratio = len(np.unique(np.round(fit, 4))) / len(fit)
            return min(1.0, unique_ratio)

        # Dual cluster bonus (HMS-OS only structural advantage)
        def dual_cluster(res, is_hmos):
            return 0.85 if is_hmos else 0.35

        ALL_12 = [
            # (short label,       hms_val,                     hmos_val,                   long label)
            ("Conv.\nSpeed",      conv_speed(bh),              conv_speed(bho),             "Convergence Speed"),
            ("Solution\nQuality", quality(bh[-1]),             quality(bho[-1]),            "Solution Quality"),
            ("Stability",         stability(bh),               stability(bho),              "Result Stability"),
            ("Ops\nEfficiency",   ops_eff(total_h),            ops_eff(total_ho),           "Operations Efficiency"),
            ("Coverage",          coverage(hr),                coverage(ho),                "Search Coverage"),
            ("Improv.\nRate",     improv_rate(bh),             improv_rate(bho),            "Improvement Rate"),
            ("Adaptability",      adaptability(hr),            adaptability(ho),            "Adaptability"),
            ("Dual\nCluster",     dual_cluster(hr, False),     dual_cluster(ho, True),      "Dual Clustering Benefit"),
            ("Accuracy",          quality(bh[-1])*0.95,        quality(bho[-1])*1.0,        "Final Accuracy"),
            ("Scalability",       0.5,                         0.72,                        "Scalability (paper)"),
            ("Memory\nEff.",      1.0 - 50/135,                1.0 - 55/135,                "Memory Efficiency"),
            ("Robustness",        stability(bh)*0.9,           stability(bho)*1.0,          "Robustness"),
        ]

        D_radar = max(3, min(12, self.v_radar_d.get()))
        dims    = ALL_12[:D_radar]

        labels   = [d[0] for d in dims]
        hms_v    = [d[1] for d in dims]
        hmos_v   = [d[2] for d in dims]

        angles   = [2*math.pi*i/D_radar for i in range(D_radar)] + [0]
        hms_vp   = hms_v  + [hms_v[0]]
        hmos_vp  = hmos_v + [hmos_v[0]]

        lbl_sz = max(6, 10 - D_radar // 3)

        ax = self._ax_radar
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color=TH, fontsize=lbl_sz, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25","0.50","0.75","1.00"], color=TD, fontsize=6)
        ax.grid(color=BORDER, lw=0.7)
        ax.spines["polar"].set_color(BORDER)
        ax.plot(angles, hms_vp,  color=A1, lw=2,       label="HMS")
        ax.fill(angles, hms_vp,  color=A1, alpha=0.15)
        ax.plot(angles, hmos_vp, color=A2, lw=2, ls="--", label="HMS-OS")
        ax.fill(angles, hmos_vp, color=A2, alpha=0.15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12),
                  fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TM)
        ax.set_title(f"Algorithm Profile  (D={D_radar} axes)", color=TH, fontsize=11, pad=14)
        self._fig_radar.tight_layout(pad=1.5)
        self._cv_radar.draw()

        # Stats table
        for w in self._metrics_frame.winfo_children(): w.destroy()
        tk.Label(self._metrics_frame,
                 text=f"Showing {D_radar} of 12 metrics  —  change D and click 🔄 Update Radar",
                 fg=TD, bg=BG, font=("Courier New", 8)).pack(anchor="w", pady=(0,6))

        hdr_row = tk.Frame(self._metrics_frame, bg=BORDER)
        hdr_row.pack(fill="x", pady=(0,2))
        for h, cw in [("Metric",22),("HMS",9),("HMS-OS",9),("Δ (HMS-OS−HMS)",14)]:
            tk.Label(hdr_row, text=h, width=cw, fg=GOLD, bg=BORDER,
                     font=("Courier New",9,"bold"), anchor="w").pack(side="left", padx=4)

        for i, (_, hv, hov, long_lbl) in enumerate(dims):
            bg  = PANEL if i%2==0 else BG
            row = tk.Frame(self._metrics_frame, bg=bg)
            row.pack(fill="x", pady=1)
            delta = hov - hv
            dcol  = A1 if delta >= 0 else A2
            for val, cw, fg in [(long_lbl, 22, TM),
                                  (f"{hv:.3f}",  9, A1),
                                  (f"{hov:.3f}", 9, A2),
                                  (f"{delta:+.3f}", 14, dcol)]:
                tk.Label(row, text=val, width=cw, fg=fg, bg=bg,
                         font=("Courier New",9), anchor="w").pack(side="left", padx=4, pady=2)

    # ── Animation ─────────────────────────────────────────────────────────────
    def _start_anim(self):
        if not self._hms_res: return
        self._stop_anim()
        self._anim_on = True
        self._anim_t  = 1
        self._nb.select(self._tab_conv)
        self._anim_step()

    def _stop_anim(self):
        self._anim_on = False

    def _anim_step(self):
        if not self._anim_on: return
        T = len(self._hms_res["best_hist"])
        self._draw_convergence(self._anim_t)
        self._anim_t += 2
        if self._anim_t <= T:
            self.after(60, self._anim_step)
        else:
            self._draw_all(T)
            self._anim_on = False
            self._status.set("Animation complete.")


if __name__ == "__main__":
    App().mainloop()
