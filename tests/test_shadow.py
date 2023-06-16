import sqlite3

import pytest

from microscore.shadow import ShadowLogger


def test_log_and_summary(tmp_path):
    log = ShadowLogger(str(tmp_path / "s.db"))
    log.log("r1", "1", "2", [0.05, 0.2, 0.5], [0.06, 0.3, 0.2], [{"a": 1}, {"a": 2}, {"a": None}])
    log.log("r2", "1", "2", [0.1], [0.1])
    log.flush()
    s = log.summary()
    assert s["count"] == 4
    assert s["max_abs_diff"] == pytest.approx(0.3)
    # 0.05->0.06 same band, 0.2->0.3 review->decline, 0.5->0.2 decline->review, 0.1->0.1 same
    assert s["decision_disagreement_rate"] == pytest.approx(0.5)
    log.close()
    rows = sqlite3.connect(tmp_path / "s.db").execute(
        "SELECT request_id, row_index, champion_score, challenger_score, features FROM shadow_log ORDER BY id"
    ).fetchall()
    assert rows[0] == ("r1", 0, 0.05, 0.06, '{"a": 1}')
    assert rows[3][4] is None


def test_empty_summary(tmp_path):
    log = ShadowLogger(str(tmp_path / "e.db"))
    assert log.summary()["count"] == 0
    log.close()


def test_full_queue_drops_instead_of_blocking(tmp_path):
    log = ShadowLogger(str(tmp_path / "q.db"), max_queue=1)
    for i in range(500):
        log.log(f"r{i}", "1", "2", [0.1], [0.2])
    log.flush()
    assert log.summary()["count"] + log.dropped == 500
    log.close()
