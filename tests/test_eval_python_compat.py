"""交付路径上的代码必须能被**评测机的 Python** 解析。

2026-08-29 的事故：`verify_delivery_runtime.py` 里有一处把三元表达式**跨行**写在
f-string 内 —— 那是 PEP 701（**Python 3.12+**）才允许的写法。本地是 3.13，跑得好好的；
评测机是 **3.11.15**，直接 `SyntaxError: unterminated string literal`。
后果：`--from-zip` 从 2026-08-25 加进来那天起**从没在评测机的 Python 上跑过**，
而我们一直以为它验过了 —— 此前两次云端跑用的是没有这个分支的旧版脚本。

⚠️ 为什么不能用 `ast.parse(..., feature_version=(3, 11))`：**它抓不到**。
`feature_version` 只影响少数语法门，3.12 的 f-string tokenizer 不受它约束，
本地实测那段代码在 `feature_version=(3, 11)` 下照样通过。所以只能自己扫 token。

覆盖三条 PEP 701 新增能力（在 3.11 上都是语法错）：

1. **跨行的 f-string**（非三引号）—— 这次真正踩到的那条；
2. f-string **表达式里嵌套的字符串**含反斜杠；
3. 嵌套字符串用了与外层 f-string **相同的引号**。

⚠️ 初版把第 2 条写成「`FSTRING_MIDDLE` 里有反斜杠」，那是**假阳性**：`FSTRING_MIDDLE`
是 f-string 的**字面量**部分，字面量里的转义在 3.11 完全合法，一口气误报了 15 处。
只有落在 `{}` **表达式**里的反斜杠才是 3.12+ 特性。教训与本文件要防的事同形 ——
**会乱叫的门禁和不会叫的门禁一样没用** —— 所以下面用两个正例、两个反例把它钉死。

范围：进提交包的模块（最要命 —— 语法错等于整个提交按填 0 处理）+ 交付验证脚本
（要在评测机上跑）。
"""
from __future__ import annotations

import io
import json
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 评测机实测版本，见 outputs/cloud/delivery_cloud_py311_4t.json 的 environment.python
EVAL_PYTHON = (3, 11)

SUBMISSION_MODULES = sorted((ROOT / "strategies" / "v3_hybrid").glob("*.py"))
DELIVERY_SCRIPTS = [ROOT / "scripts" / name for name in (
    "verify_delivery_runtime.py", "audit_submission_zip.py",
    "promote_v3_candidate.py", "make_submission.py", "check_consistency.py")]

TRIPLES = ('"""', "'''")


def pep701_violations(path: Path) -> list[str]:
    """返回该文件里所有「只有 3.12+ 才合法」的 f-string 用法。"""
    if not hasattr(tokenize, "FSTRING_START"):        # 本地解释器 < 3.12，扫不了
        raise unittest.SkipTest("需要 3.12+ 的 tokenizer 才能识别 f-string token")

    source = path.read_text(encoding="utf-8")
    where = path.relative_to(ROOT)
    eval_py = f"{EVAL_PYTHON[0]}.{EVAL_PYTHON[1]}"
    found: list[str] = []
    open_stack: list[tokenize.TokenInfo] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.FSTRING_START:
            open_stack.append(token)
        elif token.type == tokenize.FSTRING_END and open_stack:
            start = open_stack.pop()
            quote = start.string.lstrip("fFrRbB")
            if not quote.startswith(TRIPLES) and token.end[0] != start.start[0]:
                found.append(f"{where}:{start.start[0]}: 跨行 f-string"
                             f"（PEP 701，需 3.12+），评测机 {eval_py} 会 SyntaxError")
        elif token.type == tokenize.STRING and open_stack:
            # 落在 FSTRING_START..END 之间的 STRING = 写在 {} 表达式里的嵌套字符串。
            # f-string 的字面量部分是 FSTRING_MIDDLE，不会走到这里 —— 这正是
            # 初版误报 15 处的原因。
            outer = open_stack[-1].string.lstrip("fFrRbB")[0]
            inner = token.string.lstrip("fFrRbB")
            if "\\" in token.string:
                found.append(f"{where}:{token.start[0]}: f-string 表达式里的嵌套字符串"
                             f"含反斜杠（PEP 701，需 3.12+），评测机 {eval_py} 会 SyntaxError")
            elif inner.startswith(outer):
                found.append(f"{where}:{token.start[0]}: f-string 表达式里的嵌套字符串"
                             f"与外层同引号（PEP 701，需 3.12+），评测机 {eval_py} 会 SyntaxError")
    return found


class EvalPythonCompatTest(unittest.TestCase):
    def test_submission_modules_parse_on_eval_python(self) -> None:
        # 这些文件会被打进 zip。语法错 = 整个提交按填 0 处理，没有第二次机会。
        self.assertTrue(SUBMISSION_MODULES, "没找到提交模块，测试本身失效了")
        for path in SUBMISSION_MODULES:
            with self.subTest(path=path.name):
                self.assertEqual(pep701_violations(path), [])

    def test_delivery_scripts_parse_on_eval_python(self) -> None:
        # 这些要在评测机上跑；跑不起来 = 我们以为验过了、其实没验（2026-08-29 就是这样）。
        for path in DELIVERY_SCRIPTS:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), f"{path} 不存在")
                self.assertEqual(pep701_violations(path), [])

    # ---- 检查器本身的正反例：没有这四条，它可能既不叫也乱叫，我们照样看不出来

    def _violations_of(self, source: str) -> list[str]:
        fixture = ROOT / "tests" / "_pep701_fixture_tmp.py"
        fixture.write_text(source, encoding="utf-8")
        try:
            return pep701_violations(fixture)
        finally:
            fixture.unlink()

    def test_fires_on_multiline_fstring(self) -> None:
        hits = self._violations_of('x = 1\ny = f"a {1 if x else 2\n            + x}"\n')
        self.assertTrue(hits, "没抓到跨行 f-string —— 检查器没在工作")
        self.assertIn("跨行 f-string", hits[0])

    def test_fires_on_backslash_inside_expression(self) -> None:
        hits = self._violations_of('xs = ["a"]\ny = f"{chr(10).join(xs)!r}{\'\\n\'}"\n')
        self.assertTrue(hits, "没抓到表达式里的反斜杠")
        self.assertIn("含反斜杠", hits[0])

    def test_quiet_on_literal_backslash(self) -> None:
        """假阳性回归：字面量部分的转义在 3.11 合法，不许报警。初版就栽在这里。"""
        self.assertEqual(self._violations_of('x = 1\ny = f"\\n写出 {x}"\n'), [])
        self.assertEqual(self._violations_of('x = 1\ny = f"a{x}\\tb{x}\\n"\n'), [])

    def test_quiet_on_ordinary_nested_quotes(self) -> None:
        """假阳性回归：外双内单是所有版本都合法的写法，本仓库到处都是。"""
        self.assertEqual(self._violations_of('d = {"k": 1}\ny = f"{d[\'k\']}"\n'), [])

    def test_eval_python_version_matches_recorded_environment(self) -> None:
        # 版本不是随手写的：来自评测机实测产物。它变了，这个门禁的口径就要跟着变。
        recorded = json.loads(
            (ROOT / "outputs" / "cloud" / "delivery_cloud_py311_4t.json").read_text(
                encoding="utf-8"))["environment"]["python"]
        self.assertTrue(recorded.startswith(f"{EVAL_PYTHON[0]}.{EVAL_PYTHON[1]}."),
                        f"评测机实测 Python 是 {recorded}，与门禁口径 {EVAL_PYTHON} 不符")


if __name__ == "__main__":
    unittest.main()
