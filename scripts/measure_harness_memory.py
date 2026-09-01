"""把「跑完一次全量推理的峰值内存」拆成 harness 的和模型的两块。

## 为什么需要拆

`scripts/verify_delivery_runtime.py` 2026-08-23 首次量到峰值 RSS **11.47 GB**，而官方评测
环境是 **12 GB**（`docs/competition_description.md:158-159`），cgroup 实测 990 次顶到上限。
但这个数是「进程」的，不是「我们的模型」的 —— `timeseries_api/runner.py` 是主办方原文，
它自己就要 `pd.read_parquet` 一个 1.68 GB 的 test 分区，并把 214,538 个逐 time_id 的小
DataFrame 累积在 `rows` 列表里最后 concat。

这两块的性质完全不同：

- **harness 那块**：主办方在他们自己的机器上跑他们自己的 harness，我们改不了，也不该为它买单。
- **模型那块**：这是我们能控制的部分，也是唯一能为 12 GB 买回余量的地方。

不拆开就无法回答「要不要为内存改模型」。

## 怎么拆

`--arm harness`：用一个 predict 恒返回 0 的桩模型走**完全相同**的 `run_loaded_model` 路径。
`--arm production`：加载真实生产模型（与 `verify_delivery_runtime.py` 同一条路）。
两臂的峰值 RSS 之差 = 模型的净footprint。

⚠️ 两臂必须在**同样的 cgroup 约束下**跑，否则内核回收行为不同、数字不可比。
⚠️ 不写任何 CSV（CLAUDE.md §1.4）。

用法：
    OMP_NUM_THREADS=4 .venv/bin/python scripts/measure_harness_memory.py --arm harness
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_UPSTREAM_HINT = (
    "缺少主办方原文目录 `timeseries_api/`。它是主办方公开发布包的一部分，"
    "版权归主办方，不随本仓库分发 —— 见仓库根目录 UPSTREAM.md 的获取与放置说明。")

try:
    from timeseries_api.runner import load_model, run_loaded_model  # noqa: E402
except ImportError as _exc:  # pragma: no cover - 仅在缺少主办方原文时触发
    # 用 ImportError 而不是 SystemExit：后者在 import 期会让 pytest 整个 INTERNALERROR，
    # 一个测试都收集不到；ImportError 只让依赖它的那几个模块报收集错误。
    raise ImportError(f"{_UPSTREAM_HINT} 原始错误：{_exc}") from _exc
from verify_delivery_runtime import (EVAL_MEMORY_GB, peak_rss_bytes,  # noqa: E402
                                     rss_verdict)


class ZeroModel:
    """predict 恒返回 0 的桩。存在的唯一目的是让 harness 走完整条路而不加载任何模型。

    返回长度必须等于喂进来的行数 —— runner 会校验，长度不符会记 error 并填 0，
    那样测出来的就不是同一条路径了。
    """

    backend = "stub"

    def predict(self, test, **_: Any) -> np.ndarray:
        return np.zeros(len(test), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["harness", "production"])
    p.add_argument("--strategy-dir", default=str(_REPO_ROOT / "strategies" / "v3_hybrid"))
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--split", default="test")
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--rss-limit-gb", type=float, default=EVAL_MEMORY_GB)
    p.add_argument("--rss-headroom", type=float, default=0.20)
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default=None)
    p.add_argument("--trace-interval", type=float, default=0.0,
                   help=">0 时开一个采样线程，按该间隔（秒）记录 (elapsed, VmHWM, VmRSS)。"
                        "用来定位峰值发生在哪个阶段 —— 分区加载 vs 最后的 concat，"
                        "两者对私榜期长度的敏感性完全不同")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


class RssSampler:
    """后台采样 VmHWM/VmRSS。只读 /proc，开销可忽略（不影响被测量的峰值）。"""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _read(self) -> tuple[int, int]:
        hwm = rss = 0
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) * 1024
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
        except OSError:
            pass
        return hwm, rss

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            hwm, rss = self._read()
            self.samples.append({"elapsed": time.perf_counter() - self._t0,
                                 "hwm_gb": hwm / (1 << 30), "rss_gb": rss / (1 << 30)})

    @property
    def t0(self) -> float:
        return self._t0

    def start(self) -> "RssSampler":
        self._t0 = time.perf_counter()
        if self.interval > 0:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.samples


def main() -> None:
    args = parse_args()
    label = args.label or f"harness_memory_{args.arm}_{args.threads}t"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{label}.json", out_dir / f"{label}.md"
    if not args.force and json_path.exists():
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    if os.environ.get("OMP_NUM_THREADS") != str(args.threads):
        raise SystemExit(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r} "
                         f"与 --threads {args.threads} 不符")

    strategy_dir = Path(args.strategy_dir)
    sampler = RssSampler(args.trace_interval).start()
    rss_before_model = peak_rss_bytes()
    started = time.perf_counter()
    if args.arm == "harness":
        model: Any = ZeroModel()
        init_seconds = 0.0
    else:
        t0 = time.perf_counter()
        model = load_model(strategy_dir)
        init_seconds = time.perf_counter() - t0
    rss_after_model = peak_rss_bytes()

    submission, messages, timing = run_loaded_model(
        model=model, data_root=args.data_root, strategy_dir=strategy_dir,
        split=args.split, per_step_timeout_seconds=None,
        total_timeout_seconds=None, timeout_policy="zero_step")

    run_end = time.perf_counter()
    resources = rss_verdict(peak_rss_bytes(), args.rss_limit_gb, args.rss_headroom)
    resources["rss_before_model_load_bytes"] = int(rss_before_model)
    resources["rss_after_model_load_bytes"] = int(rss_after_model)
    resources["model_load_delta_bytes"] = int(rss_after_model - rss_before_model)

    timing_d = timing.as_dict()
    timing_d["model_init_seconds"] = float(init_seconds)
    timing_d["wall_clock_seconds"] = float(time.perf_counter() - started)
    timing_d["predict_total_minutes"] = timing_d["predict_total_seconds"] / 60.0

    trace = sampler.stop()
    if trace:
        # 峰值第一次达到最终高水位的时刻，占整段运行的比例 —— 这是判断
        # 「峰值属于分区加载还是最后的 concat」的直接读数。
        final_hwm = max(t["hwm_gb"] for t in trace)
        first_at = next(t["elapsed"] for t in trace if t["hwm_gb"] >= final_hwm - 1e-9)
        # ⚠️ run_end 是 perf_counter 的**绝对**读数，trace 里的 elapsed 相对 sampler.t0
        # —— 直接相除会得出「0% 处」这种无意义读数（2026-08-23 实测踩到）。折同原点。
        run_seconds = run_end - sampler.t0
        resources["trace"] = {
            "interval_seconds": args.trace_interval,
            "samples": len(trace),
            "final_hwm_gb": final_hwm,
            "first_reached_at_seconds": first_at,
            "run_seconds": run_seconds,
            "reached_at_fraction_of_run": (first_at / run_seconds) if run_seconds > 0 else None,
            "series": trace,
        }
        print(f"峰值 {final_hwm:.2f} GB 首次达到于 {first_at:.1f}s / "
              f"{run_seconds:.1f}s（{first_at/run_seconds:.0%} 处）", flush=True)

    payload = {
        "experiment": "harness_vs_model_memory_attribution",
        "purpose": ("把全量推理的峰值 RSS 拆成主办方 harness 的和我们模型的两块 —— "
                    "12 GB 上限下，只有后者是我们能买回余量的部分"),
        "arm": args.arm,
        "environment": {
            "threads_declared": args.threads, "os_cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "resources": resources,
        "timing": timing_d,
        "rows": int(len(submission)),
        "runner_messages": [m.as_dict() for m in messages],
        "note": "不写任何 CSV；桩模型只用于隔离 harness 自身开销",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    lines = [f"# 内存归属 —— `{args.arm}` 臂 @ {args.threads} 线程", "",
             f"> {payload['purpose']}", "",
             "| 项 | 值 |", "|---|---:|",
             f"| **峰值 RSS** | **{resources['peak_rss_gb']:.2f} GB** |",
             f"| 上限 | {resources['limit_gb']:.1f} GB |",
             f"| 占用率 | {resources['utilization']:.1%} |",
             f"| 模型加载净增 | {resources['model_load_delta_bytes']/(1<<30):.3f} GB |",
             f"| 行数 | {payload['rows']:,} |",
             f"| predict_total | {timing_d['predict_total_minutes']:.2f} 分钟 |",
             "", f"> {payload['note']}", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{args.arm}] 峰值 RSS = {resources['peak_rss_gb']:.2f} GB "
          f"（模型加载净增 {resources['model_load_delta_bytes']/(1<<30):.3f} GB）\n"
          f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
