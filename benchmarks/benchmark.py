"""
Empirical Complexity Benchmark
==============================
Times each algorithm on increasing sample sizes to compare
theoretical Big-O against real hardware performance.

Usage:
    python benchmarks/benchmark.py
"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# make sure we can import mymllib from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mymllib as ml


def make_data(n, d=20, task='classify'):
    """Quick synthetic data generator for timing."""
    np.random.seed(42)
    X = np.random.randn(n, d)
    if task == 'classify':
        y = np.random.randint(0, 2, size=n)
    elif task == 'multiclass':
        y = np.random.randint(0, 3, size=n)
    else:
        y = X @ np.random.randn(d) + np.random.randn(n) * 0.1
    return X, y


def time_it(name, n):
    """Train one algorithm on n samples, return wall-clock seconds."""
    X_cls, y_cls = make_data(n, task='classify')
    X_reg, y_reg = make_data(n, task='regression')

    t0 = time.time()
    try:
        if name == 'Linear Regression':
            m = ml.Linear(method='normal')
            m.fit(X_reg, y_reg)
        elif name == 'Logistic Regression':
            m = ml.Logistic(max_iter=50)
            m.fit(X_cls, y_cls)
        elif name == 'SVM':
            m = ml.SVM(kernel='linear', max_iter=20)
            m.fit(X_cls, y_cls)
        elif name == 'Decision Tree':
            m = ml.Tree(max_depth=5)
            m.fit(X_cls, y_cls)
        elif name == 'KMeans':
            m = ml.KMeans(k=3, max_iter=20)
            m.fit(X_reg)
        elif name == 'PCA':
            m = ml.PCA(n_components=5)
            m.fit(X_reg)
        elif name == 'Random Forest':
            m = ml.Forest(n_trees=10, max_depth=5, task='classify')
            m.fit(X_cls, y_cls)
    except Exception as e:
        print(f"  Warning: {name} failed at n={n}: {e}")
        return 0.0

    return time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Empirical Complexity Benchmarks for mymllib")
    parser.add_argument(
        '--sizes',
        nargs='+',
        type=int,
        default=[100, 200, 300, 400, 500],
        help="Sample sizes N to test (e.g. --sizes 100 300 500 1000)"
    )
    args = parser.parse_args()
    sizes = sorted(args.sizes)

    print(f"Running complexity benchmarks on sizes: {sizes}\n")

    algos = [
        'Linear Regression', 'Logistic Regression', 'SVM',
        'Decision Tree', 'KMeans', 'PCA', 'Random Forest',
    ]

    results = {a: [] for a in algos}

    for n in sizes:
        print(f"N = {n}")
        for a in algos:
            dt = time_it(a, n)
            print(f"  {a}: {dt:.4f}s")
            results[a].append(dt)
        print()

    # --- plot ---
    plt.figure(figsize=(10, 6))
    for a, times in results.items():
        plt.plot(sizes, times, marker='o', label=a)

    plt.title("Empirical Complexity: Wall-Clock Time vs Sample Size")
    plt.xlabel("N (samples)")
    plt.ylabel("Time (seconds)")
    plt.legend(fontsize=8)
    plt.grid(True, ls='--', alpha=.5)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), 'complexity_plot.png')
    plt.savefig(out, dpi=150)
    print(f"Plot saved -> {out}")

    # growth summary
    print("\nGrowth factors (N={} -> N={}):".format(sizes[0], sizes[-1]))
    for a in algos:
        if results[a][0] > 0:
            g = results[a][-1] / (results[a][0] + 1e-9)
            print(f"  {a}: {g:.1f}x")


if __name__ == '__main__':
    main()
