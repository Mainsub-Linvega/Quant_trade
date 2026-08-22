"""8/23 回补包 responder 列核查的四支判定。

这个脚本 8/23 当天只跑一次、且它的输出决定「要不要重开一条已经关掉的线」，
所以四支分支必须全部有用例 —— 尤其是**混合**那支：它是唯一会让人当场停下来的分支，
而 8/23 那天最容易发生的事就是赶工时把它当成噪声跳过去。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import check_backfill_responders as check  # noqa: E402

REHEARSAL = _REPO_ROOT / "outputs" / "data_audits" / "data_release_20260819_rehearsal.json"


def train_file(path: str, *, responders: int, target: bool = True,
               weight: bool = True) -> dict:
    columns = ["row_id", "time_id", "asset_id"]
    columns += [f"feature_{i:03d}" for i in range(323)]
    columns += [f"responder_{i:02d}" for i in range(responders)]
    if weight:
        columns.append("weight")
    if target:
        columns.append("target")
    return {"path": path, "columns": columns, "rows": 1_000_000}


def audit(touched: list[str], files: list[dict]) -> dict:
    return {"splits": {"train": files, "test": []},
            "comparison": {"changed": bool(touched),
                           "splits": {"train": {"added": touched, "removed": [],
                                                "modified": [], "row_delta": 0}}}}


def test_unchanged_train_split_keeps_the_axis_closed() -> None:
    payload = check.evaluate(audit([], [train_file("a.parquet", responders=47)]))
    assert payload["verdict"] == check.VERDICT_UNCHANGED
    assert payload["reopens_responder_line"] is False
    assert payload["reopen_targets"] == []


def test_backfill_with_responders_reopens_the_line() -> None:
    files = [train_file("old.parquet", responders=47), train_file("new.parquet", responders=47)]
    payload = check.evaluate(audit(["new.parquet"], files))
    assert payload["verdict"] == check.VERDICT_WITH_RESPONDERS
    assert payload["reopens_responder_line"] is True
    # 重开名单必须点名 2026-08-22 收口的三份产物 + 08-14 的母条件
    assert set(payload["reopen_targets"]) == set(check.REOPENS)
    assert "responder_stage_c_fill" in payload["reopen_targets"]


def test_backfill_without_responders_keeps_the_line_closed() -> None:
    files = [train_file("old.parquet", responders=47), train_file("new.parquet", responders=0)]
    payload = check.evaluate(audit(["new.parquet"], files))
    assert payload["verdict"] == check.VERDICT_WITHOUT_RESPONDERS
    assert payload["reopens_responder_line"] is False
    assert payload["per_file"][0]["n_responders"] == 0


def test_mixed_backfill_is_flagged_rather_than_guessed() -> None:
    """部分带、部分不带 ⟹ 不许按任一分支走。"""
    files = [train_file("a.parquet", responders=47), train_file("b.parquet", responders=0)]
    payload = check.evaluate(audit(["a.parquet", "b.parquet"], files))
    assert payload["verdict"] == check.VERDICT_INCONSISTENT
    assert payload["reopens_responder_line"] is False
    assert payload["n_with_responders"] == 1


def test_comparison_path_absent_from_splits_is_reported_not_ignored() -> None:
    payload = check.evaluate(audit(["ghost.parquet"], [train_file("a.parquet", responders=47)]))
    assert payload["paths_in_comparison_but_absent"] == ["ghost.parquet"]
    assert payload["verdict"] == check.VERDICT_INCONSISTENT


def test_responder_count_ignores_lookalike_column_names() -> None:
    """前缀是 `responder_`（**带下划线**）⟹ `responders_total` 这类列名不会被误算。

    钉住这条是因为它是个容易写错的边界：写成 `startswith("responder")` 就会把
    任何以 responder 开头的新列算进来，而判定「回补包带不带 responder」正是靠这个计数。
    """
    item = {"columns": ["responder_00", "responder_46", "responders_total", "feature_000"]}
    assert check.responder_count(item) == 2
    assert check.responder_count({"columns": ["responder_00"]}) == 1
    assert check.responder_count({"columns": ["target", "weight"]}) == 0
    assert check.responder_count({}) == 0


def test_mixed_verdict_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """混合分支必须真的以非零码退出 —— 只打印警告在赶工时会被忽略。"""
    files = [train_file("a.parquet", responders=47), train_file("b.parquet", responders=0)]
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit(["a.parquet", "b.parquet"], files)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_backfill_responders.py", "--audit", str(path)])
    with pytest.raises(SystemExit) as excinfo:
        check.main()
    assert "schema 不一致" in str(excinfo.value)


def test_clean_verdict_writes_both_artifacts(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    files = [train_file("old.parquet", responders=47), train_file("new.parquet", responders=47)]
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit(["new.parquet"], files)), encoding="utf-8")
    out = tmp_path / "verdict.json"
    monkeypatch.setattr(sys, "argv", ["check_backfill_responders.py", "--audit", str(path),
                                      "--output", str(out)])
    check.main()
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == check.VERDICT_WITH_RESPONDERS
    assert "触发" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_missing_audit_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["check_backfill_responders.py",
                                      "--audit", str(tmp_path / "nope.json")])
    with pytest.raises(SystemExit, match="找不到审计 JSON"):
        check.main()


@pytest.mark.skipif(not REHEARSAL.is_file(), reason="需要 08-19 演练审计")
def test_real_rehearsal_audit_reads_as_unchanged() -> None:
    """08-19 那次演练实测 comparison.changed=false ⟹ 本脚本必须读成「未变」。"""
    payload = check.evaluate(json.loads(REHEARSAL.read_text(encoding="utf-8")))
    assert payload["verdict"] == check.VERDICT_UNCHANGED
    assert payload["train_files_total"] == 9
