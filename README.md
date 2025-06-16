# neon_ops: High-Performance SIMD Array Operations for Apple Silicon

NeonOps is a high-performance, header-only Python extension written in C++23 that accelerates common array operations—addition, multiplication, and scaling—using NEON SIMD instructions on ARM-based Apple M-series processors (M1, M2, M3, M4).


## 📦 Installation

```bash
pip install -r requirements.txt
```

## Run Benchmark

```bash
python benchmark.py 
```

## Sample results
```
                                 Performance Summary                                 
╭───────────┬───────────┬────────────────┬───────────┬───────────────────┬──────────╮
│      Size │ Operation │ Implementation │ Time (µs) │ Throughput (GB/s) │ MOps/sec │
├───────────┼───────────┼────────────────┼───────────┼───────────────────┼──────────┤
│       100 │ add       │ python         │      3.52 │              0.11 │     0.28 │
│       100 │ add       │ numpy          │      0.29 │              1.38 │     3.44 │
│       100 │ add       │ neon           │      0.90 │              0.44 │     1.11 │
│       100 │ multiply  │ python         │      3.09 │              0.13 │     0.32 │
│       100 │ multiply  │ numpy          │      0.28 │              1.42 │     3.56 │
│       100 │ multiply  │ neon           │      0.90 │              0.45 │     1.12 │
│       100 │ scale     │ python         │      2.14 │              0.19 │     0.47 │
│       100 │ scale     │ numpy          │      0.52 │              0.77 │     1.93 │
│       100 │ scale     │ neon           │      0.66 │              0.61 │     1.52 │
│     1,000 │ add       │ python         │     29.02 │              0.14 │     0.03 │
│     1,000 │ add       │ numpy          │      0.39 │             10.38 │     2.59 │
│     1,000 │ add       │ neon           │      1.05 │              3.79 │     0.95 │
│     1,000 │ multiply  │ python         │     27.76 │              0.14 │     0.04 │
│     1,000 │ multiply  │ numpy          │      0.38 │             10.39 │     2.60 │
│     1,000 │ multiply  │ neon           │      0.98 │              4.10 │     1.02 │
│     1,000 │ scale     │ python         │     18.79 │              0.21 │     0.05 │
│     1,000 │ scale     │ numpy          │      0.63 │              6.35 │     1.59 │
│     1,000 │ scale     │ neon           │      0.72 │              5.58 │     1.40 │
│    10,000 │ add       │ python         │    284.14 │              0.14 │     0.00 │
│    10,000 │ add       │ numpy          │      1.03 │             38.76 │     0.97 │
│    10,000 │ add       │ neon           │     47.01 │              0.85 │     0.02 │
│    10,000 │ multiply  │ python         │    284.35 │              0.14 │     0.00 │
│    10,000 │ multiply  │ numpy          │      1.02 │             39.23 │     0.98 │
│    10,000 │ multiply  │ neon           │     43.12 │              0.93 │     0.02 │
│    10,000 │ scale     │ python         │    188.83 │              0.21 │     0.01 │
│    10,000 │ scale     │ numpy          │      1.11 │             35.97 │     0.90 │
│    10,000 │ scale     │ neon           │     42.57 │              0.94 │     0.02 │
│   100,000 │ add       │ python         │   2855.87 │              0.14 │     0.00 │
│   100,000 │ add       │ numpy          │     12.74 │             31.40 │     0.08 │
│   100,000 │ add       │ neon           │    117.66 │              3.40 │     0.01 │
│   100,000 │ multiply  │ python         │   2795.39 │              0.14 │     0.00 │
│   100,000 │ multiply  │ numpy          │     11.90 │             33.62 │     0.08 │
│   100,000 │ multiply  │ neon           │    119.04 │              3.36 │     0.01 │
│   100,000 │ scale     │ python         │   1817.02 │              0.22 │     0.00 │
│   100,000 │ scale     │ numpy          │      9.41 │             42.52 │     0.11 │
│   100,000 │ scale     │ neon           │    122.96 │              3.25 │     0.01 │
│ 1,000,000 │ add       │ python         │  30940.94 │              0.13 │     0.00 │
│ 1,000,000 │ add       │ numpy          │    332.15 │             12.04 │     0.00 │
│ 1,000,000 │ add       │ neon           │    220.94 │             18.10 │     0.00 │
│ 1,000,000 │ multiply  │ python         │  30024.23 │              0.13 │     0.00 │
│ 1,000,000 │ multiply  │ numpy          │    317.78 │             12.59 │     0.00 │
│ 1,000,000 │ multiply  │ neon           │    220.37 │             18.15 │     0.00 │
│ 1,000,000 │ scale     │ python         │  19923.41 │              0.20 │     0.00 │
│ 1,000,000 │ scale     │ numpy          │    297.80 │             13.43 │     0.00 │
│ 1,000,000 │ scale     │ neon           │    210.48 │             19.00 │     0.00 │
╰───────────┴───────────┴────────────────┴───────────┴───────────────────┴──────────╯

Speedup Analysis (relative to Python)
                    Speedup Factors                     
╭───────────┬───────────┬───────────────┬──────────────╮
│      Size │ Operation │ NumPy Speedup │ NEON Speedup │
├───────────┼───────────┼───────────────┼──────────────┤
│       100 │ add       │        12.11x │        3.91x │
│       100 │ multiply  │        10.99x │        3.45x │
│       100 │ scale     │         4.13x │        3.25x │
│     1,000 │ add       │        75.31x │       27.53x │
│     1,000 │ multiply  │        72.13x │       28.44x │
│     1,000 │ scale     │        29.82x │       26.21x │
│    10,000 │ add       │       275.30x │        6.04x │
│    10,000 │ multiply  │       278.87x │        6.59x │
│    10,000 │ scale     │       169.78x │        4.44x │
│   100,000 │ add       │       224.17x │       24.27x │
│   100,000 │ multiply  │       234.98x │       23.48x │
│   100,000 │ scale     │       193.14x │       14.78x │
│ 1,000,000 │ add       │        93.15x │      140.04x │
│ 1,000,000 │ multiply  │        94.48x │      136.24x │
│ 1,000,000 │ scale     │        66.90x │       94.66x │
╰───────────┴───────────┴───────────────┴──────────────╯
```
