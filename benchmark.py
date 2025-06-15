import numpy as np
import neon_ops
import time
from typing import Callable, Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich import box
import json
from pathlib import Path
from datetime import datetime
import platform
import psutil


@dataclass
class BenchmarkResult:
    size: int
    operation: str
    implementation: str
    mean_time_us: float
    std_time_us: float
    throughput_gbs: float  # Gigabytes per second
    operations_per_sec: float


class BenchmarkSuite:
    def __init__(self):
        self.console = Console()
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """Gather system information"""
        return {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
            "memory_available_gb": psutil.virtual_memory().available / (1024 ** 3)
        }

    def plain_python_add(self, a: list, b: list) -> list:
        return [x + y for x, y in zip(a, b)]

    def plain_python_multiply(self, a: list, b: list) -> list:
        return [x * y for x, y in zip(a, b)]

    def plain_python_scale(self, a: list, scalar: float) -> list:
        return [x * scalar for x in a]

    def benchmark_operation(self, operation: Callable, *args, num_runs: int = 100) -> tuple[float, float]:
        """Benchmark a single operation with multiple runs."""
        times = []
        # Warm-up run
        _ = operation(*args)

        for _ in range(num_runs):
            start = time.perf_counter_ns()
            _ = operation(*args)
            end = time.perf_counter_ns()
            times.append((end - start) / 1000)  # Convert to microseconds

        return np.mean(times), np.std(times)

    def calculate_throughput(self, size: int, time_us: float) -> float:
        """Calculate throughput in GB/s"""
        bytes_processed = size * 4  # 4 bytes per float32
        return (bytes_processed / 1e9) / (time_us / 1e6)

    def run_benchmarks(self, sizes: List[int]):
        """Run all benchmarks"""
        for size in track(sizes, description="Running benchmarks"):
            self.console.print(f"[blue]Benchmarking size: {size:,}")

            # Create arrays
            a_np = np.array(np.random.rand(size), dtype=np.float32)
            b_np = np.array(np.random.rand(size), dtype=np.float32)
            a_list = a_np.tolist()
            b_list = b_np.tolist()
            scalar = 2.0

            operations = {
                'add_python': (self.plain_python_add, (a_list, b_list)),
                'add_numpy': (np.add, (a_np, b_np)),
                'add_neon': (neon_ops.add_arrays, (a_np, b_np)),
                'multiply_python': (self.plain_python_multiply, (a_list, b_list)),
                'multiply_numpy': (np.multiply, (a_np, b_np)),
                'multiply_neon': (neon_ops.multiply_arrays, (a_np, b_np)),
                'scale_python': (self.plain_python_scale, (a_list, scalar)),
                'scale_numpy': (np.multiply, (a_np, scalar)),
                'scale_neon': (neon_ops.scale_array, (a_np, scalar))
            }

            for op_name, (op, args) in operations.items():
                try:
                    mean_time, std_time = self.benchmark_operation(op, *args)
                    throughput = self.calculate_throughput(size, mean_time)
                    ops_per_sec = 1e6 / mean_time

                    self.results.append(BenchmarkResult(
                        size=size,
                        operation=op_name.split('_')[0],
                        implementation=op_name.split('_')[1],
                        mean_time_us=mean_time,
                        std_time_us=std_time,
                        throughput_gbs=throughput,
                        operations_per_sec=ops_per_sec
                    ))
                except Exception as e:
                    self.console.print(f"[red]Error in {op_name} with size {size}: {e}")

    def create_report(self):
        """Generate comprehensive benchmark report"""
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame([vars(r) for r in self.results])

        # System Information
        self.console.print(Panel.fit(
            "\n".join([f"{k}: {v}" for k, v in self.system_info.items()]),
            title="[bold blue]System Information",
            border_style="blue"
        ))

        # Performance Summary Table
        table = Table(title="Performance Summary", box=box.ROUNDED)
        table.add_column("Size", justify="right")
        table.add_column("Operation")
        table.add_column("Implementation")
        table.add_column("Time (µs)", justify="right")
        table.add_column("Throughput (GB/s)", justify="right")
        table.add_column("MOps/sec", justify="right")

        for result in self.results:
            table.add_row(
                f"{result.size:,}",
                result.operation,
                result.implementation,
                f"{result.mean_time_us:.2f}",
                f"{result.throughput_gbs:.2f}",
                f"{result.operations_per_sec / 1e6:.2f}"
            )

        self.console.print(table)

        # Speedup Analysis
        self.console.print("\n[bold green]Speedup Analysis (relative to Python)")

        # Calculate speedups
        speedup_table = Table(title="Speedup Factors", box=box.ROUNDED)
        speedup_table.add_column("Size", justify="right")
        speedup_table.add_column("Operation")
        speedup_table.add_column("NumPy Speedup", justify="right")
        speedup_table.add_column("NEON Speedup", justify="right")

        # Group by size and operation
        for size in sorted(df['size'].unique()):
            for op in sorted(df['operation'].unique()):
                data = df[(df['size'] == size) & (df['operation'] == op)]
                if len(data) >= 3:  # Ensure we have all three implementations
                    python_time = data[data['implementation'] == 'python']['mean_time_us'].iloc[0]
                    numpy_time = data[data['implementation'] == 'numpy']['mean_time_us'].iloc[0]
                    neon_time = data[data['implementation'] == 'neon']['mean_time_us'].iloc[0]

                    numpy_speedup = python_time / numpy_time
                    neon_speedup = python_time / neon_time

                    speedup_table.add_row(
                        f"{size:,}",
                        op,
                        f"{numpy_speedup:.2f}x",
                        f"{neon_speedup:.2f}x"
                    )

        self.console.print(speedup_table)

    def create_plots(self, output_dir: str = "benchmark_results"):
        """Generate interactive plots"""
        Path(output_dir).mkdir(exist_ok=True)
        df = pd.DataFrame([vars(r) for r in self.results])

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Execution Time (lower is better)',
                'Throughput (higher is better)',
                'Speedup vs Python (higher is better)',
                'Operations per Second (higher is better)'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        colors = {'python': '#1f77b4', 'numpy': '#ff7f0e', 'neon': '#2ca02c'}

        # Execution Time plot
        for impl in ['python', 'numpy', 'neon']:
            for op in df['operation'].unique():
                data = df[(df['implementation'] == impl) & (df['operation'] == op)]
                fig.add_trace(
                    go.Scatter(
                        x=data['size'],
                        y=data['mean_time_us'],
                        name=f'{impl} {op}',
                        line=dict(color=colors[impl]),
                        showlegend=True
                    ),
                    row=1, col=1
                )

        # Throughput plot
        for impl in ['python', 'numpy', 'neon']:
            for op in df['operation'].unique():
                data = df[(df['implementation'] == impl) & (df['operation'] == op)]
                fig.add_trace(
                    go.Scatter(
                        x=data['size'],
                        y=data['throughput_gbs'],
                        name=f'{impl} {op}',
                        line=dict(color=colors[impl]),
                        showlegend=False
                    ),
                    row=1, col=2
                )

        # Speedup plot
        for op in df['operation'].unique():
            for impl in ['numpy', 'neon']:
                speedups = []
                sizes = []
                for size in sorted(df['size'].unique()):
                    data = df[(df['size'] == size) & (df['operation'] == op)]
                    if len(data) >= 3:
                        python_time = data[data['implementation'] == 'python']['mean_time_us'].iloc[0]
                        impl_time = data[data['implementation'] == impl]['mean_time_us'].iloc[0]
                        speedups.append(python_time / impl_time)
                        sizes.append(size)

                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=speedups,
                        name=f'{impl} {op} speedup',
                        line=dict(color=colors[impl]),
                        showlegend=True
                    ),
                    row=2, col=1
                )

        # Operations per second plot
        for impl in ['python', 'numpy', 'neon']:
            for op in df['operation'].unique():
                data = df[(df['implementation'] == impl) & (df['operation'] == op)]
                fig.add_trace(
                    go.Scatter(
                        x=data['size'],
                        y=data['operations_per_sec'],
                        name=f'{impl} {op}',
                        line=dict(color=colors[impl]),
                        showlegend=False
                    ),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            height=1000,
            width=1400,
            title_text=f"Benchmark Results - {platform.processor()}",
            showlegend=True
        )

        # Update axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Array Size", type="log", row=i, col=j)

        fig.update_yaxes(title_text="Time (µs)", type="log", row=1, col=1)
        fig.update_yaxes(title_text="GB/s", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Speedup Factor", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Operations/sec", type="log", row=2, col=2)

        # Save plots
        fig.write_html(f"{output_dir}/benchmark_results.html")
        fig.write_image(f"{output_dir}/benchmark_results.png")

    def save_results(self, output_dir: str = "benchmark_results"):
        """Save benchmark results and system info to JSON"""
        Path(output_dir).mkdir(exist_ok=True)

        results_dict = {
            "system_info": self.system_info,
            "results": [vars(r) for r in self.results]
        }

        with open(f"{output_dir}/benchmark_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)


def main():
    # Initialize benchmark suite
    suite = BenchmarkSuite()

    # Run benchmarks
    sizes = [100, 1000, 10000, 100000, 1000000]
    suite.run_benchmarks(sizes)

    # Generate reports and visualizations
    suite.create_report()
    suite.create_plots()
    suite.save_results()


if __name__ == "__main__":
    main()