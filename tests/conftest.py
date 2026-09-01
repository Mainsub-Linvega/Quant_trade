"""缺少主办方原文 `timeseries_api/` 时，跳过依赖它的测试模块。

背景：`docs/`、`examples/`、`timeseries_api/` 是主办方原文，版权归主办方，不随本仓库
分发（见 UPSTREAM.md）。少数交付验证脚本在**模块层**导入 `timeseries_api.runner`，
导入它们的测试模块因此会在**收集期**失败 —— 而 pytest 一旦收集出错就中断整轮，
公开仓库的克隆者会一个测试都跑不到。

这里不是把门禁关掉：本机与任何放回了主办方原文的环境上，这三个模块照常收集、照常跑。
只有在真的缺目录时才跳过，并打一行说明。
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

def _upstream_dependents() -> set[str]:
    """`scripts/` 里（直接或间接）依赖 `timeseries_api` 的模块名。

    间接也要算：`check_consistency` 自己不碰 timeseries_api，但它 import
    `verify_delivery_runtime`，后者碰。所以按传递闭包展开，而不是只扫一层 ——
    只扫一层的版本漏掉了 test_check_consistency_window.py。
    """
    sources = {path.stem: path.read_text(encoding="utf-8")
               for path in (_ROOT / "scripts").glob("*.py")}
    tainted = {"timeseries_api"}
    changed = True
    while changed:
        changed = False
        for name, source in sources.items():
            if name in tainted:
                continue
            if any(dep in source for dep in tainted):
                tainted.add(name)
                changed = True
    return tainted


collect_ignore: list[str] = []

if not (_ROOT / "timeseries_api" / "runner.py").exists():
    dependents = _upstream_dependents()
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        if any(name in source for name in dependents):
            collect_ignore.append(path.name)
    if collect_ignore:
        print(f"[conftest] 缺少主办方 timeseries_api/（见 UPSTREAM.md），"
              f"跳过 {len(collect_ignore)} 个依赖它的测试模块：{', '.join(collect_ignore)}")


# 需要 `data/` 的测试各自用 skipUnless 处理，不在这里统一拦。
