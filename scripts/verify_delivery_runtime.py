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

## 为什么还要量内存

`docs/competition_description.md:152-166` 写明评测环境是 **4 核 / 12 GB 内存 / 无 GPU / 无外网**，
且「内存超限……严重情况下提交可能被判定为无效」。而本脚本此前**一个内存字段都没有** ——
唯一那个 RSS 数字（NumPy 兜底 4.56 GB）只写在 `NOTES.md` 正文里，不是产物，还是在 30 GB
开发机上量的。私榜 8/31 截止后无法修改代码、出错按填 0 处理 ⟹ 内存是唯一一个能让整个提交
归零、而我们从未测量过的量。峰值取内核维护的高水位（`VmHWM` 与 `ru_maxrss` 取大者），
不需要采样线程。

## `--from-zip`：测真正的交付物，不是手搭目录

⚠️ 2026-08-25 补。此前本脚本永远指着 `strategies/v3_hybrid/` 跑 —— 那是**源目录**，
不是交出去的那件东西。打包会做取舍（`promotion_manifest.json` / `consistency_*.json`
被 `ignore_patterns` 排除、只收声明过的 `*.py`），而这些取舍从来没有被官方 runner
真正装载过一次。`--from-zip` 把 zip 解压到 `outputs/delivery_verify/<stem>/` 再跑，
并把 zip 的 sha256 写进落盘 JSON —— 那是**归属锚点**，证明这一次跑的就是那份 zip。
同时进程内跑一遍 `audit_submission_zip.audit(..., expect_public_baseline=True)`，
把「内容审计过了」与「官方 runner 跑通了」绑在同一件产物上。

⚠️ 解压落点是 `outputs/`（真实块设备）而**不是** `/tmp`：本机 `/tmp` 是 tmpfs，
写进去等于占内存，而这个 runner 峰值本来就 ~11.5 GB（CLAUDE.md 伤疤规则 13）。

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
import resource
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from timeseries_api.runner import load_model, run_loaded_model  # noqa: E402

# docs/competition_description.md:158-159 —— 官方评测环境的硬约束。
EVAL_CPU_CORES = 4
EVAL_MEMORY_GB = 12.0
EXPECTED_ROWS = 3_217_458
EXPECTED_CALLS = 214_538
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy-dir", default=str(_REPO_ROOT / "strategies" / "v3_hybrid"))
    p.add_argument("--from-zip", default=None,
                   help="改为解压这份提交 zip 再跑（落点 outputs/delivery_verify/<stem>/），"
                        "并把 zip 的 sha256 与内容审计结果写进 JSON。与 --strategy-dir 互斥")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--split", default="test")
    p.add_argument("--backend", required=True, choices=["lightgbm", "numpy-fallback"])
    p.add_argument("--threads", type=int, required=True,
                   help="预期线程数；与 OMP_NUM_THREADS 不符直接退出（避免记错口径）")
    # ⚠️ 2026-08-23：这里原来写死 `v3_hybrid_slowfast`，而生产早在 08-21 就换成了
    # long512 ⟹ `model_matches_promotion_manifest` 长期红着、两份报告都判 FAIL，
    # 而红的原因是**比错了对象**不是装错了模型。写死一个候选名必然随每次转正过期，
    # 所以默认改成 auto：扫描全部 staging，挑逐字节相同的那一份。
    p.add_argument("--manifest", default="auto",
                   help="promotion manifest 路径；`auto` = 扫描 outputs/promotions/* "
                        "找与生产目录逐字节相同的那一份")
    # ⚠️ 默认 0（关闭）：本地基线 5.40 / 10.90 分钟是在**没有**它的情况下量的，
    # 保持默认一致，后续跑才能与那两个数直接比。云端长跑（兜底约 25 分钟）建议
    # `--progress-every 20000` —— 否则整段是黑盒，分不清「慢」和「卡死」。
    p.add_argument("--progress-every", type=int, default=0,
                   help="每 N 次 predict 打一行进度（含已用时/速率/ETA/RSS）；0 = 关闭")
    p.add_argument("--rss-limit-gb", type=float, default=EVAL_MEMORY_GB,
                   help=f"评测环境内存上限（GB）；默认 {EVAL_MEMORY_GB} 取自官方文档")
    p.add_argument("--rss-headroom", type=float, default=0.20,
                   help="要求的余量比例；峰值须低于 limit×(1−headroom) 才算有余量")
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


_PEAK_RSS_SEEN = 0


def peak_rss_bytes() -> int:
    """本进程的峰值 RSS，单调不降。取两个来源的大者，不需要采样线程。

    - `/proc/self/status` 的 `VmHWM`，单位 kB。
    - `resource.getrusage(RUSAGE_SELF).ru_maxrss`，Linux 上也是 kB
      （macOS 是字节；本项目只在 Linux 上交付，不为此加分支）。

    ⚠️ **`VmHWM` 自己不是严格单调的**（2026-08-23 实测：221.49 → 220.94 MB）。
    内核报的是 `max(记录的 hiwater, 当前 RSS)`，而记录值更新滞后 —— 当前 RSS 一旦
    跌回记录值以下，读数就退回那个略低的数。`ru_maxrss` 是真单调的，但两者会差
    零点几 MB，取大者才不低报。再叠一层模块级高水位，让本函数无论何时调用都不低报。

    ⚠️ 只算本进程。官方 runner 全程在进程内跑（`run_loaded_model` 不 fork），
    所以这个数就是评测机上要与 12 GB 比的那个数。
    """
    global _PEAK_RSS_SEEN
    hwm = 0
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    _PEAK_RSS_SEEN = max(_PEAK_RSS_SEEN, hwm,
                         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return _PEAK_RSS_SEEN


def rss_verdict(peak_bytes: int, limit_gb: float, headroom: float) -> dict[str, Any]:
    """把峰值 RSS 折成可审计的资源块 + 两道门禁。纯函数，便于单测。

    两道分开判：`under_limit` 是「会不会被判无效」的硬线；`has_headroom` 是
    「私榜一整个月都不能改代码」所以要留的安全边际。只过前者不过后者 ⟹ 能跑，
    但没有余量吸收任何环境差异（云端实测就比本地慢 2.3×）。
    """
    gb = peak_bytes / (1 << 30)
    threshold = limit_gb * (1.0 - headroom)
    return {
        "peak_rss_bytes": int(peak_bytes),
        "peak_rss_gb": gb,
        "limit_gb": float(limit_gb),
        "headroom_fraction": float(headroom),
        "headroom_threshold_gb": threshold,
        "utilization": gb / limit_gb if limit_gb else None,
        "under_limit": gb < limit_gb,
        "has_headroom": gb < threshold,
        "eval_env_note": ("docs/competition_description.md:158-159 —— "
                          f"官方评测环境 {EVAL_CPU_CORES} 核 / {EVAL_MEMORY_GB} GB"),
    }


def extract_submission_zip(zip_path: Path, *, force: bool,
                           root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """把提交 zip 解压到 `outputs/delivery_verify/<stem>/`，并留下归属证据。

    返回 `(解压目录, 证据块)`。证据块里的 `sha256` 是这次 runner 到底跑了哪份 zip 的
    唯一凭据 —— 没有它，「测了交付物」就只是一句自述。

    ⚠️ 落点必须是真实块设备。本机 `/tmp`（含 session scratchpad）是 tmpfs，
    往那儿解压等于把交付物放进内存，而本脚本峰值 RSS 就有 ~11.5 GB
    （CLAUDE.md 伤疤规则 13：3.9 GB 的 memmap 曾先后掐死两个任务）。
    `root` 只为单测留一个不写进仓库 `outputs/` 的出口，生产路径不传它。

    ⚠️ 审计不过**不在这里抛**：证据要落盘、要能看见是哪几项红了。
    拦截由 `checks["zip_audit_passed"]` 负责，与其它门禁同一层。
    """
    if not zip_path.exists():
        raise SystemExit(f"--from-zip 指向的文件不存在：{zip_path}")

    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    try:
        from audit_submission_zip import audit as audit_zip
    finally:
        sys.path.remove(str(_REPO_ROOT / "scripts"))
    report = audit_zip(zip_path, expect_public_baseline=True)

    target = (root or _REPO_ROOT / "outputs" / "delivery_verify") / zip_path.stem
    if target.exists() and not force:
        raise SystemExit(f"解压落点已存在：{target}；用 --force 覆盖")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)

    evidence = {
        "path": str(zip_path),
        "sha256": report["sha256"],
        "bytes": report["bytes"],
        "file_count": len(report["files"]),
        "extracted_to": str(target),
        "audit_passed": report["passed"],
        "audit_failing_checks": [k for k, ok in report["checks"].items() if not ok],
        "requirements_summary": report.get("requirements_summary", {}),
    }
    return target, evidence


def resolve_manifest(model_dir: Path, spec: str) -> dict[str, Any]:
    """把 `--manifest` 解析成一个具体路径，并留下解析过程本身当证据。

    `auto` 扫描 `outputs/promotions/*/promotion_manifest.json`，挑出 staged 文件与生产目录
    **逐字节相同**的那一份。这不是循环论证 —— 它回答的是 CLAUDE.md §1.5/§6 真正在意的
    那个问题：「生产目录里的东西，是不是来自一次有记录的 staging」。手改过生产 meta、
    或装了一个从没进过 staging 的模型，都不会有任何 manifest 匹配。
    模型身份是否**等于榜上那份**由 `model_matches_public_baseline` 单独把关。
    """
    files = {q.name: sha256_file(q) for q in sorted(model_dir.iterdir()) if q.is_file()}
    if spec != "auto":
        return {"spec": spec, "resolved": spec, "scanned": [], "matched": []}

    promo_root = _REPO_ROOT / "outputs" / "promotions"
    scanned, matched = [], []
    for mp in sorted(promo_root.glob("*/promotion_manifest.json")):
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        staged = manifest.get("staged_files") or manifest.get("source_files") or {}
        same = bool(files and len(staged) == len(files)
                    and all(staged.get(n) == h for n, h in files.items()))
        scanned.append({"name": mp.parent.name, "created_at": manifest.get("created_at"),
                        "identical": same})
        if same:
            matched.append(str(mp))
    return {"spec": spec, "resolved": matched[0] if matched else None,
            "scanned": scanned, "matched": matched}


class ProgressProxy:
    """夹在 model 与官方 runner 之间，数 predict 次数并定期打点。

    存在的理由：`run_loaded_model` 是主办方原文（只读），循环里不打任何进度，
    而云端兜底要跑约 25 分钟 —— 没有进度就分不清「慢」「卡死」「被 OOM 杀」。
    2026-08-23 云端实测正是栽在这一点上。

    ⚠️ 只在显式传 `--progress-every` 时才套上。每次调用只多一个整数自增与取模
    （相对 1.5~3 ms 的 predict 可忽略），但默认关闭以保证与既有基线同口径。
    ⚠️ `__getattr__` 透传其余属性 —— runner 只调 `predict`，但 `backend` 等字段
    要能被外部读到，不然身份检查会看错对象。
    """

    __slots__ = ("_model", "_every", "_expected", "_n", "_t0")

    def __init__(self, model: Any, every: int, expected: int) -> None:
        self._model = model
        self._every = int(every)
        self._expected = int(expected)
        self._n = 0
        self._t0 = time.perf_counter()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def predict(self, test: Any) -> Any:
        out = self._model.predict(test)
        self._n += 1
        if self._every > 0 and self._n % self._every == 0:
            elapsed = time.perf_counter() - self._t0
            rate = self._n / elapsed if elapsed > 0 else 0.0
            remaining = max(self._expected - self._n, 0)
            eta = remaining / rate if rate > 0 else float("nan")
            print(f"  [进度] {self._n:,}/{self._expected:,} "
                  f"({self._n / self._expected:5.1%})  "
                  f"已用 {elapsed/60:5.1f} 分  ETA {eta/60:5.1f} 分  "
                  f"{rate:,.0f} 次/秒  RSS {peak_rss_bytes()/(1<<30):.2f} GB", flush=True)
        return out


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


def public_baseline_drift(meta: dict[str, Any]) -> list[str]:
    """meta 与「榜上那份模型」的逐键差异（空列表 = 完全一致）。

    直接复用 `scripts/audit_submission_zip.public_baseline_drift` —— 它已经维护着
    PUBLIC_BASELINE 全表的取值映射，并且在漏键时硬失败。**不要**在这里另抄一份取值表：
    两张表分头维护正是 08-18 与 08-21 两次「丢键静默降级」事故的形状。
    """
    scripts_dir = str(_REPO_ROOT / "scripts")
    inserted = scripts_dir not in sys.path
    if inserted:
        sys.path.insert(0, scripts_dir)
    try:
        from audit_submission_zip import public_baseline_drift as _drift
        return _drift(meta)
    finally:
        if inserted and scripts_dir in sys.path:
            sys.path.remove(scripts_dir)


def model_identity(model_dir: Path, resolution: dict[str, Any]) -> dict[str, Any]:
    files = {p.name: sha256_file(p) for p in sorted(model_dir.iterdir()) if p.is_file()}
    meta = json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))
    identity = {k: meta.get(k) for k in (
        "blend_weight", "num_iteration", "prediction_scale", "prediction_clip",
        "market_lambda", "history_window", "sample_modulo", "sampling",
        "cross_section_weighted", "slow_fast_window", "slow_fast_slow_relative",
        "slow_fast_fast_relative",
        # ⚠️ 2026-08-21 补：长窗块也是模型身份。这份报告号称打印「meta 身份」，
        # 漏一个身份键就等于报告在说谎 —— 与 08-18 PUBLIC_BASELINE 漏 slow/fast 同一类。
        "long_window")}
    identity["n_lgbm_models"] = len(meta.get("lgbm_model_files", []))
    identity["n_market_models"] = len(meta.get("market_model_files", []))
    identity["n_features"] = len(meta.get("lgbm_features", []))
    identity["n_history_positions"] = len(meta.get("history_positions", []))

    out: dict[str, Any] = {"model_dir": str(model_dir), "file_sha256": files,
                           "meta_identity": identity,
                           "public_baseline_drift": public_baseline_drift(meta)}
    resolved = resolution.get("resolved")
    manifest_path = Path(resolved) if resolved else None
    if manifest_path is not None and manifest_path.exists():
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
        out["manifest"] = {
            "path": resolved,
            "identical": False,
            "note": ("auto 扫描未找到与生产目录逐字节相同的 staging"
                     if resolution.get("spec") == "auto" else "manifest 不存在"),
        }
    out["manifest"]["resolution"] = resolution
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
    zip_evidence: dict[str, Any] | None = None
    if args.from_zip:
        if args.strategy_dir != str(_REPO_ROOT / "strategies" / "v3_hybrid"):
            raise SystemExit("--from-zip 与 --strategy-dir 互斥：要么测源目录，要么测交付物")
        strategy_dir, zip_evidence = extract_submission_zip(Path(args.from_zip),
                                                            force=args.force)
        print(f"交付物：{zip_evidence['path']}\n"
              f"  sha256 {zip_evidence['sha256'][:16]}…，"
              f"{zip_evidence['file_count']} 个文件，"
              f"内容审计 {'通过' if zip_evidence['audit_passed'] else '未通过 '
                          + str(zip_evidence['audit_failing_checks'])}\n"
              f"  解压到 {zip_evidence['extracted_to']}", flush=True)

    resolution = resolve_manifest(strategy_dir / "model", args.manifest)
    identity = model_identity(strategy_dir / "model", resolution)
    drift = identity["public_baseline_drift"]
    print(f"模型身份：manifest={identity['manifest'].get('identical')} "
          f"（{len(identity['file_sha256'])} 个文件，"
          f"来自 {identity['manifest'].get('path')}）\n"
          f"公榜基线偏离：{drift or '无'}", flush=True)

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

    runner_model = (ProgressProxy(model, args.progress_every, EXPECTED_CALLS)
                    if args.progress_every > 0 else model)
    submission, messages, timing = run_loaded_model(
        model=runner_model, data_root=args.data_root, strategy_dir=strategy_dir,
        split=args.split, per_step_timeout_seconds=args.per_step_timeout,
        total_timeout_seconds=None, timeout_policy="zero_step")

    resources = rss_verdict(peak_rss_bytes(), args.rss_limit_gb, args.rss_headroom)
    rss_gb = resources["peak_rss_gb"]
    rss_limit = resources["limit_gb"]
    rss_headroom_gb = resources["headroom_threshold_gb"]
    print(f"峰值 RSS = {rss_gb:.2f} GB（上限 {rss_limit} GB，"
          f"余量线 {rss_headroom_gb:.2f} GB）", flush=True)

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
        # 整条预测向量的逐位指纹。max|pred| 只反映一个极值，单个叶子翻了未必动到它；
        # 要判断两次运行（不同后端、不同机器、不同依赖版本）是否**逐位**相同，
        # 需要这个摘要。仍然不写 CSV —— 只把 float64 缓冲区喂给 sha256。
        "predictions_sha256": hashlib.sha256(
            np.ascontiguousarray(pred, dtype=np.float64).tobytes()).hexdigest(),
    }
    timing_d = timing.as_dict()
    timing_d["model_init_seconds"] = float(model_init_seconds)
    timing_d["wall_clock_seconds"] = float(time.perf_counter() - started)
    timing_d["predict_total_minutes"] = timing_d["predict_total_seconds"] / 60.0
    timing_d["wall_clock_minutes"] = timing_d["wall_clock_seconds"] / 60.0

    checks = {
        # --from-zip 时把「内容审计」与「跑通」绑在同一件产物上：不能出现
        # 「审计的是 A、跑的是 B」。没给 --from-zip 时这一项恒真（无交付物可审）。
        "zip_audit_passed": zip_evidence is None or bool(zip_evidence["audit_passed"]),
        "model_matches_promotion_manifest": identity["manifest"].get("identical") is True,
        # 非循环的那一道：PUBLIC_BASELINE 是手工维护的「榜上那份长什么样」，
        # 与「来自某次 staging」是两件独立的事，必须分开判。
        "model_matches_public_baseline": not drift,
        "peak_rss_under_limit": resources["under_limit"],
        "peak_rss_has_headroom": resources["has_headroom"],
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
            "progress_every": args.progress_every,
            "os_cpu_count": os.cpu_count(), "python": platform.python_version(),
            "platform": platform.platform(), "lightgbm": lgb_version,
            "numpy": np.__version__,
        },
        "submission_zip": zip_evidence,
        "model_identity": identity,
        "resources": resources,
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
             f"- 对照：官方评测环境 **{EVAL_CPU_CORES} 核 / {EVAL_MEMORY_GB} GB**"
             f"（`docs/competition_description.md:158-159`）",
             "", "## 资源", "",
             "| 项 | 值 |", "|---|---:|",
             f"| **峰值 RSS** | **{resources['peak_rss_gb']:.2f} GB** |",
             f"| 上限 | {rss_limit:.1f} GB |",
             f"| 余量线（{args.rss_headroom:.0%} 余量）| {rss_headroom_gb:.2f} GB |",
             f"| 占用率 | {resources['utilization']:.1%} |",
             "", "## 模型身份", "",
             f"- 与 promotion manifest 逐文件 sha256 比对："
             f"**{identity['manifest'].get('identical')}**"
             f"（{identity['manifest'].get('files_compared')} 个文件）",
             f"- manifest 来源：`{identity['manifest'].get('path')}`"
             f"（`--manifest {args.manifest}`）",
             f"- 公榜基线偏离：{('**' + '；'.join(drift) + '**') if drift else '无'}",
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
