"""线程数默认值探针 —— 回答「`num_threads` 不设（-1）在评测机上会不会劣化」。

## 为什么要有这个脚本

2026-08-29 主办方在 `docs/competition_description.md` 新增「重要 Note」，要求策略代码
**内部**手动设置最大线程数，并警告默认值 `-1` 会让推理性能「极大劣化且稳定性下降」。

而 `strategies/v3_hybrid/main.py` 的 `booster.predict(...)` **没有传 `num_threads`**。
我们历次交付验证的「4 线程」全部来自**外部**环境变量 `OMP_NUM_THREADS=4` ——
评测机不会替我们设它。⟹ 这是一个 CLAUDE.md §8.11 形状的归属缺口：
**测量的口径来自被测物之外的一个开关，而那个开关不随提交包走。**

## 为什么本地量不出来，必须上云

libgomp 尊重 `sched_getaffinity`，所以 **cpuset 型**限核（`taskset`、docker `--cpuset-cpus`）
下 `-1` 会自己收敛到实配核数，测不出劣化 —— 本机 32 核实测：

    可见32/实配32： -1 = 0.416 ms   num_threads=4 = 0.463 ms
    可见32/实配 4： -1 = 0.475 ms   num_threads=4 = 0.492 ms

危险的是**另一种**：cgroup `cpu.max` **配额型**限制**不改变** affinity，
`-1` 会在全部可见核上起线程去挤一个小得多的配额。云端 jhub 实测 `os_cpu_count = 128`
（`outputs/cloud/delivery_cloud_py311_4t.json`），但那两次云端跑**都设了**
`OMP_NUM_THREADS=4` ⟹ **我们从未观测过评测机上的默认行为**。

## 怎么跑（在云端 ~/Quant_trade 下）

    # 关键：不要设 OMP_NUM_THREADS，那正是被测对象
    env -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS python experiments/thread_default_probe.py

判读：`-1` 相对 `num_threads=4` 的比值。
  ≲ 1.2×  ⟹ 该机器上默认行为无害，交付件可按原样交。
  ≫ 1.2×  ⟹ 必须给 main.py 补 num_threads 并重打包。

只做 predict 微基准（几十秒），不跑全量、不写 CSV、不碰 model/。
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STRATEGY = _REPO_ROOT / "strategies" / "v3_hybrid"
CANDIDATES = (None, 1, 2, 4, 8)          # None = 不传 num_threads，即官方警告的默认 -1
REPEATS = 2000

# 云端真机全量实测的锚点（outputs/cloud/delivery_cloud_{py311,numpy}_4t.json，均钉 4 线程）。
# 微基准只能给**比值**，绝对值要靠这两个锚点折算 —— 合成输入比真实分区便宜。
CALLS = 214_538
BUDGET_S = 0.05 * CALLS                  # 官方公式取 a=b=0 的下限
CLOUD_LGBM_MEAN_MS = 3.393               # lightgbm 主路径，total 821.0 s = 0.228 h
CLOUD_LGBM_TOTAL_S = 821.0
CLOUD_NUMPY_MEAN_MS = 9.448              # numpy 兜底，total 2158.6 s = 0.600 h


def bench(boosters, design, num_iteration: int, num_threads: int | None) -> float:
    """一次 `predict` 调用 = 三片森林各跑一遍，与 `main.py:_forest_mean` 同形状。"""
    kwargs = {} if num_threads is None else {"num_threads": num_threads}
    for booster in boosters:                                  # 预热，别把首调用计进去
        booster.predict(design, num_iteration=num_iteration, **kwargs)
    start = time.perf_counter()
    for _ in range(REPEATS):
        for booster in boosters:
            booster.predict(design, num_iteration=num_iteration, **kwargs)
    return (time.perf_counter() - start) / REPEATS * 1000.0


def bench_end_to_end() -> dict:
    """整条 `Model.predict` 的每次调用耗时 —— 树推理只是其中一段。

    为什么要这一节：上面那个微基准只量 `booster.predict`，而**兜底路径根本不走它**。
    纯 numpy 树遍历不用 BLAS，但岭回归那步是 `@`（`main.py:399`），在多核容器上
    OpenBLAS 同样可能按可见核数展开 —— 这条**从来没在不钉线程的情况下量过**。
    三个臂：出厂设置 / 抹掉 num_threads（改动前的行为）/ numpy 兜底。
    """
    import pandas as pd

    sys.path.insert(0, str(_STRATEGY))
    from main import Model                                   # noqa: PLC0415

    def frame_for(model, time_id: int) -> "pd.DataFrame":
        rng = np.random.default_rng(time_id)
        rows = model.forest.n_assets
        data = {name: rng.normal(0.0, 1.0, rows).astype(np.float32)
                for name in model.feature_columns}
        data["time_id"] = np.full(rows, time_id, dtype=np.int64)
        data["asset_id"] = np.arange(rows, dtype=np.int64)
        data["row_id"] = np.arange(rows, dtype=np.int64) + time_id * rows
        return pd.DataFrame(data)

    def run(label: str, backend: str | None, strip_threads: bool, calls: int = 300) -> float:
        model = Model(str(_STRATEGY / "model"), backend=backend)
        if strip_threads:
            model.predict_kwargs = {k: v for k, v in model.predict_kwargs.items()
                                    if k != "num_threads"}
        note = f"predict_kwargs={model.predict_kwargs}" if backend != "numpy" else "纯 numpy"
        model.predict(frame_for(model, 1))                   # 预热 + 建好历史状态
        start = time.perf_counter()
        for step in range(2, calls + 2):
            model.predict(frame_for(model, step))
        ms = (time.perf_counter() - start) / calls * 1000.0
        print(f"  {label:34s} {ms:9.3f} ms / predict 调用   [{note}]")
        return ms

    print("\n端到端 `Model.predict`（含取列 / 岭回归 / 历史状态，不只是树）：")
    out = {}
    try:
        out["lightgbm_num_threads_4"] = run("出厂设置（num_threads=4）", None, False)
        out["lightgbm_default_threads"] = run("抹掉 num_threads（改动前）", None, True)
    except Exception as error:                               # noqa: BLE001
        print(f"  lightgbm 臂跳过：{error!r}")
    out["numpy_fallback"] = run("numpy 兜底（单线程树遍历）", "numpy", False)

    # ⚠️ 只报**同一次运行内的比值**，不报折算小时数：合成输入比真实分区便宜
    # （本机端到端 2.09 ms vs 云端全量实测 3.39 ms），把微基准的绝对值折算成全量
    # 会系统性低估 —— CLAUDE.md §5.7「代理量不可跨结构搬用」。比值是尺度无关的，
    # 把它乘到已知的全量读数上才是对的用法。
    # ⚠️ 外推必须用**加性**而不是乘性，且基准要选对那一条。
    # 线程颠簸是给树推理**加上**一段固定开销，不是把整条 predict 按比例放大；
    # 而 `Model.predict` 里还有取列、岭回归、历史状态这些不受线程数影响的部分。
    # 乘性外推会把这些也一起放大 ⟹ 高估。
    # 2026-08-29 初版这里犯了两个错，一起记下来：
    #   ① 基准写成 0.60 h —— 那是**numpy 兜底**的 total，lightgbm 主路径是 0.228 h；
    #   ② 用了乘性。两个错叠加打出 44.70 h，而加性 + 正确基准是 **20.98 h**。
    a, b = out.get("lightgbm_default_threads"), out.get("lightgbm_num_threads_4")
    n = out.get("numpy_fallback")
    if a and b:
        extra = a - b                                        # 每次调用多出来的秒数（ms）
        mean = CLOUD_LGBM_MEAN_MS + extra
        total_h = mean * CALLS / 3.6e6
        abort = BUDGET_S / (mean / 1000.0)
        print(f"\n端到端 默认/显式4 = **{a / b:.2f}×**，加性开销 **{extra:.1f} ms/次**")
        print(f"  ⟹ 折到云端全量实测（lightgbm mean {CLOUD_LGBM_MEAN_MS:.2f} ms/次，"
              f"total {CLOUD_LGBM_TOTAL_S/3600:.2f} h）：{mean:.1f} ms/次 = "
              f"**{total_h:.2f} h**，官方总预算下限 {BUDGET_S/3600:.2f} h "
              f"⟹ **{total_h*3600/BUDGET_S:.0%}**")
        if abort < CALLS:
            print(f"  ⟹ 约第 {abort:,.0f} 次调用（{abort/CALLS:.1%} 处）撞总闸，"
                  f"其后 {1 - abort/CALLS:.1%} 的 time_id 填 0")
        else:
            print("  ⟹ 不撞总闸（本机测不出降级；危险形状是 affinity 远大于 CPU 配额）")
    if n and b:
        print(f"numpy 兜底 / lightgbm(显式4) = **{n / b:.2f}×**"
              f"（钉 4 线程时这个比值的历史读数是 {CLOUD_NUMPY_MEAN_MS/CLOUD_LGBM_MEAN_MS:.2f}× ⟹ "
              f"明显更大才说明兜底也受线程影响）")
    return out


def main() -> None:
    import lightgbm as lgb

    model_dir = _STRATEGY / "model"
    meta = json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))
    num_iteration = int(meta["num_iteration"])
    boosters = [lgb.Booster(model_file=str(model_dir / name))
                for name in meta["lgbm_model_files"]]

    # 官方 runner 每次 predict 恰好喂一个 time_id = 15 个资产（main.py 模块头）。
    # 合成输入即可：这里量的是线程调度开销，不是预测值。
    rows = int(meta.get("n_assets", 15) or 15)
    n_features = boosters[0].num_feature()
    rng = np.random.default_rng(0)
    design = rng.normal(0.0, 1.0, size=(rows, n_features)).astype(np.float32)

    env = {k: os.environ.get(k) for k in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    print(f"lightgbm {lgb.__version__} / numpy {np.__version__} / "
          f"Python {platform.python_version()}")
    print(f"os.cpu_count() = {os.cpu_count()}；"
          f"affinity = {len(os.sched_getaffinity(0))}；线程环境变量 = {env}")
    print(f"设计矩阵 {design.shape}，{len(boosters)} 片森林 × {num_iteration} 轮，"
          f"每格重复 {REPEATS} 次\n")

    results: dict[str, float] = {}
    for num_threads in CANDIDATES:
        label = "默认（不传，= -1）" if num_threads is None else f"num_threads={num_threads}"
        ms = bench(boosters, design, num_iteration, num_threads)
        results["default" if num_threads is None else str(num_threads)] = ms
        print(f"  {label:24s} {ms:8.3f} ms / predict 调用")

    end_to_end = bench_end_to_end()

    baseline = results.get("4")
    if baseline:
        ratio = results["default"] / baseline
        print(f"\n默认 / num_threads=4 = **{ratio:.2f}×**")
        print("判读：≲1.2× ⟹ 该机器上默认无害；≫1.2× ⟹ 必须补 num_threads 并重打包。")

    out = _REPO_ROOT / "outputs" / "experiments" / "thread_default_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "thread_default_probe",
        "purpose": "量 num_threads 默认值(-1) 相对显式 4 的劣化倍数，决定是否重打包",
        "environment": {
            "os_cpu_count": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)),
            "thread_env": env,
            "lightgbm": lgb.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
        },
        "design_shape": list(design.shape), "n_boosters": len(boosters),
        "num_iteration": num_iteration, "repeats": REPEATS,
        "ms_per_predict_call": results,
        "default_over_4x": (results["default"] / baseline) if baseline else None,
        "end_to_end_ms_per_call": end_to_end,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
