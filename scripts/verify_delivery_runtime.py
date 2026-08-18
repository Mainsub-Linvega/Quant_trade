"""P0 交付验证：在**钉死的线程数**下走官方 runner 全量推理，落盘一份可审计的 JSON。

## 为什么需要这个脚本

ROADMAP §2「性能风险」记着当前生产模型 `predict_total = 5.15 分钟`、NumPy 兜底 10.44 分钟。
但这两个数**只写在 ROADMAP 和 ledger 里，没有落盘的 runner JSON，而且线程数没有记录**。
P0 动作 3 要的恰恰是「在接近私榜环境的 4 核设置下验证 LightGBM 主路径；单独记录 NumPy
兜底风险」—— 开发机是 32 核，32 线程下的耗时不能替 4 核背书。

本脚本把这件事变成可复现的产物：线程数、模型文件 hash、后端选择、逐项计时和预测健康度
全部写进 JSON。

## 两条路径都测

- `--backend lightgbm`：官方 runner 的默认路径（`Model()` 自动选后端），断言实际选中 lightgbm。
- `--backend numpy-fallback`：**装 import shim 让 `import lightgbm` 抛错**，这才是评测机上
  lightgbm 不可用时的真实路径。不是传 `--backend numpy` 假装 —— 那个参数根本传不进官方
  runner（`load_model` 调的是无参 `Model()`）。

## 不产生任何提交文件

调的是 `run_loaded_model` 而不是 `run_strategy`，后者会 `to_csv`。本脚本只在内存里对返回的
submission 做统计，**不写 CSV、不打包 zip**（CLAUDE.md §1.4：那两件事只能由用户执行）。

用法：
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
        scripts/verify_delivery_runtime.py --backend lightgbm --threads 4
输出：outputs/experiments/<label>.{json,md}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from timeseries_api.runner import load_model, run_loaded_model  # noqa: E402

EXPECTED_ROWS = 3_217_458
EXPECTED_CALLS = 214_538
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy-dir", default=str(_REPO_ROOT / "strategies" / "v3_hybrid"))
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--split", default="test")
    p.add_argument("--backend", required=True, choices=["lightgbm", "numpy-fallback"])
    p.add_argument("--threads", type=int, required=True,
                   help="预期线程数；与 OMP_NUM_THREADS 不符直接退出（避免记错口径）")
    p.add_argument("--manifest", default=str(_REPO_ROOT / "outputs" / "promotions" /
                                             "v3_hybrid_slowfast" / "promotion_manifest.json"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default=None)
    p.add_argument("--per-step-timeout", type=float, default=None)
    p.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_lightgbm_shim(stack: list) -> Path:
    """让 `import lightgbm` 抛 ImportError —— 复刻评测机没装 lightgbm 的情形。"""
    tmp = tempfile.mkdtemp(prefix="lgb_shim_")
    (Path(tmp) / "lightgbm.py").write_text(
        'raise ImportError("lightgbm intentionally unavailable '
        '(verify_delivery_runtime shim)")\n', encoding="utf-8")
    for name in [m for m in sys.modules if m == "lightgbm" or m.startswith("lightgbm.")]:
        del sys.modules[name]
    sys.path.insert(0, tmp)
    stack.append(tmp)
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return Path(tmp)
    raise SystemExit("shim 未生效：lightgbm 仍可 import ⟹ 这一跑不是真兜底路径")


def model_identity(model_dir: Path, manifest_path: Path) -> dict[str, Any]:
    files = {p.name: sha256_file(p) for p in sorted(model_dir.iterdir()) if p.is_file()}
    meta = json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))
    identity = {k: meta.get(k) for k in (
        "blend_weight", "num_iteration", "prediction_scale", "prediction_clip",
        "market_lambda", "history_window", "sample_modulo", "sampling",
        "cross_section_weighted", "slow_fast_window", "slow_fast_slow_relative",
        "slow_fast_fast_relative")}
    identity["n_lgbm_models"] = len(meta.get("lgbm_model_files", []))
    identity["n_market_models"] = len(meta.get("market_model_files", []))
    identity["n_features"] = len(meta.get("lgbm_features", []))
    identity["n_history_positions"] = len(meta.get("history_positions", []))

    out: dict[str, Any] = {"model_dir": str(model_dir), "file_sha256": files,
                           "meta_identity": identity}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        staged = manifest.get("staged_files") or manifest.get("source_files") or {}
        mismatch = {n: {"model_dir": h, "manifest": staged.get(n)}
                    for n, h in files.items() if staged.get(n) != h}
        out["manifest"] = {
            "path": str(manifest_path),
            "files_compared": len(files),
            "files_in_manifest": len(staged),
            "mismatches": mismatch,
            "identical": bool(files and not mismatch and len(staged) == len(files)),
        }
    else:
        out["manifest"] = {"path": str(manifest_path), "identical": None,
                           "note": "manifest 不存在，跳过比对"}
    return out


def main() -> None:
    args = parse_args()
    label = args.label or f"delivery_runtime_{args.backend.replace('-', '_')}_{args.threads}t"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{label}.json", out_dir / f"{label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    env = {k: os.environ.get(k) for k in THREAD_ENV}
    if env["OMP_NUM_THREADS"] != str(args.threads):
        raise SystemExit(
            f"OMP_NUM_THREADS={env['OMP_NUM_THREADS']!r} 与 --threads {args.threads} 不符。"
            f"请这样跑：OMP_NUM_THREADS={args.threads} OPENBLAS_NUM_THREADS={args.threads} ...")

    shim_dirs: list = []
    lgb_version: Any = None
    if args.backend == "numpy-fallback":
        install_lightgbm_shim(shim_dirs)
        lgb_version = "shimmed (ImportError)"
    else:
        import lightgbm as lgb
        lgb_version = lgb.__version__

    strategy_dir = Path(args.strategy_dir)
    identity = model_identity(strategy_dir / "model", Path(args.manifest))
    print(f"模型身份：{identity['manifest'].get('identical')} "
          f"（{len(identity['file_sha256'])} 个文件）", flush=True)

    started = time.perf_counter()
    init_start = time.perf_counter()
    model = load_model(strategy_dir)
    model_init_seconds = time.perf_counter() - init_start
    selected = getattr(model, "backend", None)
    expected_backend = "numpy" if args.backend == "numpy-fallback" else "lightgbm"
    if selected != expected_backend:
        raise SystemExit(f"后端选择不符：期望 {expected_backend}，实际 {selected!r}")
    print(f"后端 = {selected}；model_init = {model_init_seconds:.2f}s；开始全量推理…",
          flush=True)

    submission, messages, timing = run_loaded_model(
        model=model, data_root=args.data_root, strategy_dir=strategy_dir,
        split=args.split, per_step_timeout_seconds=args.per_step_timeout,
        total_timeout_seconds=None, timeout_policy="zero_step")

    pred = submission["target"].to_numpy(dtype=np.float64)
    clip = float(identity["meta_identity"]["prediction_clip"] or 0.0)
    health = {
        "rows": int(len(pred)),
        "rows_expected": int(args.expected_rows),
        "rows_match": bool(len(pred) == args.expected_rows),
        "non_finite": int((~np.isfinite(pred)).sum()),
        "max_abs_prediction": float(np.max(np.abs(pred))) if len(pred) else 0.0,
        "clip_hit_rows": int((np.abs(pred) >= clip - 1e-12).sum()) if clip else 0,
        "all_zero": bool(len(pred) and not np.any(pred)),
    }
    timing_d = timing.as_dict()
    timing_d["model_init_seconds"] = float(model_init_seconds)
    timing_d["wall_clock_seconds"] = float(time.perf_counter() - started)
    timing_d["predict_total_minutes"] = timing_d["predict_total_seconds"] / 60.0
    timing_d["wall_clock_minutes"] = timing_d["wall_clock_seconds"] / 60.0

    checks = {
        "model_matches_promotion_manifest": identity["manifest"].get("identical") is True,
        "backend_as_requested": selected == expected_backend,
        "row_count_correct": health["rows_match"],
        "zero_non_finite": health["non_finite"] == 0,
        "zero_clip_rows": health["clip_hit_rows"] == 0,
        "zero_timeouts": timing_d["predict_timeout_count"] == 0,
        "not_aborted": not timing_d["aborted_after_timeout"],
        "predict_calls_expected": timing_d["predict_calls"] == EXPECTED_CALLS,
        "no_error_messages": not any(m.level == "error" for m in messages),
    }
    payload = {
        "experiment": "delivery_runtime_verification",
        "purpose": ("P0 交付闭环：在钉死线程数下走官方 runner 全量推理，"
                    "把此前只写在 ROADMAP 的耗时数字变成可审计产物"),
        "backend_mode": args.backend,
        "environment": {
            "threads_declared": args.threads, "thread_env": env,
            "os_cpu_count": os.cpu_count(), "python": platform.python_version(),
            "platform": platform.platform(), "lightgbm": lgb_version,
            "numpy": np.__version__,
        },
        "model_identity": identity,
        "timing": timing_d,
        "prediction_health": health,
        "runner_messages": [m.as_dict() for m in messages],
        "checks": checks,
        "pass": all(checks.values()),
        "note": ("调 run_loaded_model 而非 run_strategy ⟹ 全程不写任何 CSV；"
                 "提交文件只能由用户执行 scripts/make_submission.py 生成"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    lines = [f"# 交付运行时验证 —— `{args.backend}` @ {args.threads} 线程", "",
             f"> {payload['purpose']}", "",
             "## 环境", "",
             f"- 声明线程数 **{args.threads}**；`OMP_NUM_THREADS={env['OMP_NUM_THREADS']}`、"
             f"`OPENBLAS_NUM_THREADS={env['OPENBLAS_NUM_THREADS']}`；机器 {os.cpu_count()} 核",
             f"- lightgbm：`{lgb_version}`；numpy `{np.__version__}`；Python {platform.python_version()}",
             "", "## 模型身份", "",
             f"- 与 promotion manifest 逐文件 sha256 比对："
             f"**{identity['manifest'].get('identical')}**"
             f"（{identity['manifest'].get('files_compared')} 个文件）",
             f"- meta 身份：{json.dumps(identity['meta_identity'], ensure_ascii=False)}",
             "", "## 计时", "",
             "| 项 | 值 |", "|---|---:|",
             f"| model_init | {timing_d['model_init_seconds']:.2f} s |",
             f"| **predict_total** | **{timing_d['predict_total_minutes']:.2f} 分钟** |",
             f"| wall clock | {timing_d['wall_clock_minutes']:.2f} 分钟 |",
             f"| predict 调用次数 | {timing_d['predict_calls']:,} |",
             f"| 单步最大 | {timing_d['max_predict_seconds']:.3f} s |",
             f"| 单步平均 | {timing_d['mean_predict_seconds']*1000:.2f} ms |",
             f"| 超时次数 | {timing_d['predict_timeout_count']} |",
             "", "## 预测健康度", "",
             "| 项 | 值 |", "|---|---:|",
             f"| 行数 | {health['rows']:,}（期望 {health['rows_expected']:,}）|",
             f"| 非有限值 | {health['non_finite']} |",
             f"| max\\|pred\\| | {health['max_abs_prediction']:.6f}（clip {clip}）|",
             f"| 触 clip 行数 | {health['clip_hit_rows']} |",
             "", "## 门禁", ""]
    lines += [f"- {'✅' if v else '❌'} `{k}`" for k, v in checks.items()]
    lines += ["", f"## 判定：{'✅ PASS' if payload['pass'] else '❌ FAIL'}", "",
              f"> {payload['note']}", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}", flush=True)
    print(f"\npredict_total = {timing_d['predict_total_minutes']:.2f} 分钟 "
          f"（wall {timing_d['wall_clock_minutes']:.2f}）\n"
          f"判定：{'PASS' if payload['pass'] else 'FAIL'}\n"
          f"wrote {json_path}\nwrote {md_path}", flush=True)
    for d in shim_dirs:
        if d in sys.path:
            sys.path.remove(d)


if __name__ == "__main__":
    main()
