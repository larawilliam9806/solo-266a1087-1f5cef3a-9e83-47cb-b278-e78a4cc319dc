from __future__ import annotations

import io
import threading

import pytest

import app as app_module
import db


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


def _seed(records=None):
    records = records or [
        db.Record("CNKI-1", "CNKI", "张伟", "清华计算机"),
        db.Record("CNKI-2", "CNKI", "李娜", "北大信科"),
        db.Record("CNKI-3", "CNKI", "王芳", "复旦数学"),
        db.Record("WF-1", "万方", "张伟", "清华 计算机系"),
        db.Record("WF-2", "万方", "李娜", "北大生命科学"),
    ]
    db.import_records(records)


def _csv(rows):
    buf = io.StringIO()
    buf.write("record_id,source,name,org\n")
    for r in rows:
        if isinstance(r, db.Record):
            buf.write(f"{r.record_id},{r.source},{r.name},{r.org}\n")
        else:
            buf.write(",".join(r) + "\n")
    return buf.getvalue()


def _preview(client, source, rows_or_text, filename="roster.csv"):
    text = rows_or_text if isinstance(rows_or_text, str) else _csv(rows_or_text)
    return client.post(
        "/refresh/preview",
        data={"source": source, "file": (io.BytesIO(text.encode("utf-8")), filename)},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def _preview_id_from(resp):
    body = resp.data.decode()
    marker = 'name="preview_id" value="'
    start = body.index(marker) + len(marker)
    return body[start:body.index('"', start)]


def _confirm(client, preview_id):
    return client.post("/refresh/confirm", data={"preview_id": preview_id}, follow_redirects=True)


def _groups():
    return {frozenset(r.record_id for r in g["records"]) for g in db.group_summary()}


# ---------------------------------------------------------------------------
# Basic deactivate / reactivate behaviour
# ---------------------------------------------------------------------------

def test_refresh_deactivates_absent_records_and_keeps_their_verdicts(client):
    _seed()
    v = db.decide("CNKI-1", "CNKI-2", "same")["verdict_id"]
    # Full roster for CNKI that drops CNKI-3 and updates CNKI-1.
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机系"),
        ("CNKI-2", "CNKI", "李娜", "北大信科"),
        ("CNKI-4", "CNKI", "赵敏", "上交电院"),
    ])
    assert resp.status_code == 200
    pid = _preview_id_from(resp)
    body = resp.data.decode()
    assert "将停用" in body and "CNKI-3" in body
    assert "将更新 / 恢复" in body and "将新增" in body

    resp = _confirm(client, pid)
    assert "已刷新来源".encode() in resp.data
    assert db.get_record("CNKI-3").enabled is False
    assert db.get_record("CNKI-1").org == "清华计算机系"
    assert db.get_record("CNKI-4").enabled is True
    # Other source untouched.
    assert db.get_record("WF-1").enabled is True and db.get_record("WF-1").org == "清华 计算机系"
    # Verdict retained even though a different endpoint is fine here.
    assert v in {vv.id for vv in db.list_verdicts()}
    assert frozenset({"CNKI-1", "CNKI-2"}) in _groups()


def test_header_only_csv_clears_the_source(client):
    _seed()
    db.decide("CNKI-1", "CNKI-2", "same")
    resp = _preview(client, "CNKI", "record_id,source,name,org\n")
    pid = _preview_id_from(resp)
    _confirm(client, pid)
    all_cnki = [r for r in db.list_records() if r.source == "CNKI"]
    assert all_cnki and all(not r.enabled for r in all_cnki)
    # Records persist, verdict persists, but no CNKI record is in the export.
    assert {r.record_id for r in db.list_records()} >= {"CNKI-1", "CNKI-2", "CNKI-3"}
    mapping = dict(db.export_rows())
    assert not ({"CNKI-1", "CNKI-2", "CNKI-3"} & set(mapping))
    assert mapping  # WF records still exported


def test_disabled_record_does_not_bridge_groups(client):
    _seed()
    db.decide("CNKI-1", "CNKI-2", "same")
    db.decide("CNKI-2", "CNKI-3", "same")
    assert frozenset({"CNKI-1", "CNKI-2", "CNKI-3"}) in _groups()

    # A CNKI roster omitting CNKI-2 (the bridge) deactivates it.
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ])
    _confirm(client, _preview_id_from(resp))

    assert frozenset({"CNKI-1"}) in _groups() and frozenset({"CNKI-3"}) in _groups()
    mapping = dict(db.export_rows())
    assert "CNKI-2" not in mapping
    # The dormant verdicts are still visible via the page.
    body = client.get("/").data.decode()
    assert "停用记录的判定" in body and "CNKI-2" in body
    # A disabled record cannot receive new judgements.
    with pytest.raises(db.ValidationError):
        db.decide("CNKI-2", "CNKI-1", "same")


def test_reactivation_restores_surviving_verdicts(client):
    _seed()
    db.decide("CNKI-1", "CNKI-2", "same")
    db.decide("CNKI-2", "CNKI-3", "same")
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ])
    _confirm(client, _preview_id_from(resp))
    assert frozenset({"CNKI-1"}) in _groups()

    # Record reappears -> original surviving "same" judgements become active again.
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机"),
        ("CNKI-2", "CNKI", "李娜", "北大信科"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ])
    body = resp.data.decode()
    assert "恢复" in body
    _confirm(client, _preview_id_from(resp))
    assert db.get_record("CNKI-2").enabled is True
    assert frozenset({"CNKI-1", "CNKI-2", "CNKI-3"}) in _groups()


def test_revoking_dormant_verdict_is_allowed_and_prevents_conflict(client):
    _seed()
    db.decide("CNKI-1", "CNKI-2", "same")
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ])
    _confirm(client, _preview_id_from(resp))
    dormant = [v for v in db.list_verdicts() if "CNKI-2" in (v.a, v.b)]
    assert dormant
    resp = client.post(f"/revoke/{dormant[0].id}", follow_redirects=True)
    assert "已撤回".encode() in resp.data
    assert not [v for v in db.list_verdicts() if v.id == dormant[0].id]


# ---------------------------------------------------------------------------
# Whole-batch rejections
# ---------------------------------------------------------------------------

def test_duplicate_ids_reject_whole_batch(client):
    _seed()
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华"),
        ("CNKI-8", "CNKI", "甲", "A"),
        ("CNKI-8", "CNKI", "乙", "B"),
    ])
    assert resp.status_code != 200 or "重复记录ID" in resp.data.decode()
    # Nothing changed at all.
    assert {r.record_id for r in db.list_records()} == {
        "CNKI-1", "CNKI-2", "CNKI-3", "WF-1", "WF-2",
    }
    assert db.get_record("CNKI-1").enabled is True


def test_cross_source_id_rejects_whole_batch(client):
    _seed()
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华"),
        ("WF-1", "CNKI", "张伟", "清华"),
    ])
    body = resp.data.decode()
    assert "已被来源“万方”占用" in body
    assert db.get_record("WF-1").source == "万方"


def test_source_mismatch_rejects_whole_batch(client):
    _seed()
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华"),
        ("CNKI-9", "万方", "某人", "某单位"),
    ])
    body = resp.data.decode()
    assert "与所选来源“CNKI”不一致" in body
    assert "CNKI-9" not in {r.record_id for r in db.list_records()}


def test_normal_import_rejects_disabled_id_and_source_change(client):
    _seed()
    # Deactivate CNKI-3 via whole-source refresh.
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机"),
        ("CNKI-2", "CNKI", "李娜", "北大信科"),
    ])
    _confirm(client, _preview_id_from(resp))
    assert db.get_record("CNKI-3").enabled is False

    body = _csv([("CNKI-3", "CNKI", "王芳", "复旦"), ("NEW-1", "CNKI", "新人", "某单位")])
    resp = client.post(
        "/import", data={"file": (io.BytesIO(body.encode()), "n.csv")},
        content_type="multipart/form-data", follow_redirects=True,
    )
    assert "已停用".encode() in resp.data
    assert db.get_record("CNKI-3").enabled is False
    assert "NEW-1" not in {r.record_id for r in db.list_records()}

    # Changing an id's source in a normal import is rejected for the whole batch.
    body = _csv([("CNKI-1", "万方", "张伟", "清华"), ("NEW-2", "CNKI", "新人", "某单位")])
    resp = client.post(
        "/import", data={"file": (io.BytesIO(body.encode()), "n2.csv")},
        content_type="multipart/form-data", follow_redirects=True,
    )
    assert "不能改为".encode() in resp.data
    assert db.get_record("CNKI-1").source == "CNKI"
    assert "NEW-2" not in {r.record_id for r in db.list_records()}


# ---------------------------------------------------------------------------
# Preview staleness and reactivation conflicts
# ---------------------------------------------------------------------------

def test_preview_is_invalidated_by_record_or_verdict_changes(client):
    _seed()
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机系"),
        ("CNKI-2", "CNKI", "李娜", "北大信科"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ])
    pid = _preview_id_from(resp)
    # A verdict change after the preview invalidates the confirmation.
    db.decide("CNKI-1", "CNKI-2", "same")
    resp = _confirm(client, pid)
    body = resp.data.decode()
    assert "预览已失效" in body
    assert db.get_record("CNKI-1").org == "清华计算机"  # not applied

    # Fresh preview on top of the changed state applies normally.
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机系"),
        ("CNKI-2", "CNKI", "李娜", "北大信科"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ])
    _confirm(client, _preview_id_from(resp))
    assert db.get_record("CNKI-1").org == "清华计算机系"


def test_reactivation_conflict_is_rejected_and_state_preserved(client):
    _seed()
    # While all records are enabled: CNKI-1 same WF-1, CNKI-2 different WF-1.
    db.decide("CNKI-1", "WF-1", "same")
    db.decide("CNKI-2", "WF-1", "different")

    # WF-1 disappears from 万方 -> disabled (its verdicts go dormant).
    resp = _preview(client, "万方", [("WF-2", "万方", "李娜", "北大生命科学")])
    _confirm(client, _preview_id_from(resp))
    assert db.get_record("WF-1").enabled is False

    # With WF-1 dormant, CNKI-1–CNKI-2 can be declared "same".
    assert db.decide("CNKI-1", "CNKI-2", "same")["ok"] is True

    # Reactivating WF-1 would put the "different" pair WF-1/CNKI-2 together.
    roster = [
        ("WF-1", "万方", "张伟", "清华 计算机系"),
        ("WF-2", "万方", "李娜", "北大生命科学"),
    ]
    resp = _preview(client, "万方", roster)
    assert "已判定为" in resp.data.decode()
    pid = _preview_id_from(resp)

    resp = _confirm(client, pid)
    body = resp.data.decode()
    assert "整源刷新被拒绝" in body
    assert "WF-1" in body and "CNKI-2" in body
    # Refresh-before state preserved: WF-1 stays disabled, no roster update.
    assert db.get_record("WF-1").enabled is False
    assert db.get_record("WF-1").org == "清华 计算机系"
    # Existing groupings and verdicts untouched.
    assert frozenset({"CNKI-1", "CNKI-2"}) in _groups()

    # After revoking the offending verdict, the same roster preview succeeds.
    diff = next(v for v in db.list_verdicts() if v.kind == "different")
    db.revoke(diff.id)
    resp = _preview(client, "万方", roster)
    _confirm(client, _preview_id_from(resp))
    assert db.get_record("WF-1").enabled is True
    assert frozenset({"CNKI-1", "CNKI-2", "WF-1"}) in _groups()


def test_confirm_twice_is_rejected(client):
    _seed()
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机"),
        ("CNKI-2", "CNKI", "李娜", "北大信科"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ])
    pid = _preview_id_from(resp)
    assert "已刷新来源".encode() in _confirm(client, pid).data
    resp = _confirm(client, pid)
    assert "预览不存在".encode() in resp.data


# ---------------------------------------------------------------------------
# Consistency, concurrency, persistence
# ---------------------------------------------------------------------------

def test_export_and_page_never_bridge_through_disabled_records(client):
    _seed()
    db.decide("CNKI-1", "CNKI-2", "same")
    db.decide("CNKI-2", "WF-1", "same")
    resp = _preview(client, "CNKI", [
        ("CNKI-1", "CNKI", "张伟", "清华计算机"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ])
    _confirm(client, _preview_id_from(resp))

    mapping = dict(db.export_rows())
    assert mapping["CNKI-1"] != mapping["WF-1"]
    view = db.snapshot()
    assert view["group_of"]["CNKI-1"] != view["group_of"]["WF-1"]
    json_body = client.get("/api/groups").get_json()
    assert "CNKI-2" not in json_body["mapping"]
    assert "CNKI-2" in json_body["disabled_record_ids"]


def test_concurrent_refresh_confirm_and_decision_keep_invariant(client):
    _seed()
    db.decide("CNKI-1", "WF-1", "same")
    db.decide("CNKI-2", "WF-1", "different")

    roster = [
        ("CNKI-1", "CNKI", "张伟", "清华计算机"),
        ("CNKI-2", "CNKI", "李娜", "北大信科"),
        ("CNKI-3", "CNKI", "王芳", "复旦数学"),
    ]
    resp = _preview(client, "CNKI", roster)
    # A concurrent decision after the preview makes the confirmation stale.
    db.decide("CNKI-1", "CNKI-3", "same")
    resp = _confirm(client, _preview_id_from(resp))
    assert "预览已失效".encode() in resp.data

    # Under concurrent attempts only complete, invariant-preserving states appear.
    errors = []

    roster_records = [db.Record(*row) for row in roster]

    def confirm_worker():
        try:
            p = db.create_refresh_preview("CNKI", roster_records)
            db.confirm_refresh(p["preview_id"])
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def decide_worker():
        try:
            db.decide("CNKI-1", "WF-1", "same")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    for _ in range(3):
        threads = [threading.Thread(target=confirm_worker), threading.Thread(target=decide_worker)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors
    view = db.snapshot()
    for pair in view["groups"]:
        assert not pair["different_to"]
    # End state must be one of the complete, legitimate outcomes: either the
    # post-preview decision merged CNKI-1 with WF-1, or the refresh ran.
    groups = _groups()
    if db.get_record("CNKI-3").enabled:
        assert frozenset({"CNKI-1", "CNKI-3"}) in groups or frozenset({"CNKI-1", "CNKI-3", "WF-1"}) in groups
    diff_pairs = [
        (v.a, v.b) for v in db.list_verdicts()
        if v.kind == "different" and db.get_record(v.a).enabled and db.get_record(v.b).enabled
    ]
    for a, b in diff_pairs:
        assert view["group_of"][a] != view["group_of"][b]


def test_disabled_state_and_verdicts_persist_across_reopen(client, tmp_path):
    path = str(tmp_path / "persist.db")
    db.DB_PATH = path
    db.init_db()
    db.import_records([
        db.Record("P1", "S", "甲", "A"),
        db.Record("P2", "S", "乙", "B"),
    ])
    db.decide("P1", "P2", "same")
    db.create_refresh_preview("S", [db.Record("P1", "S", "甲", "A")])
    preview_id = db.list_records()
    # Apply through the preview stored in the same DB.
    with db.connection() as conn:
        pid = conn.execute("SELECT id FROM refresh_previews").fetchone()["id"]
    db.confirm_refresh(pid)
    assert db.get_record("P2").enabled is False

    db.DB_PATH = path  # simulate restart
    assert db.get_record("P2").enabled is False
    assert [v.kind for v in db.list_verdicts()] == ["same"]
    assert "P2" not in dict(db.export_rows())
    assert "P2" in [r.record_id for r in db.snapshot()["disabled_records"]]
