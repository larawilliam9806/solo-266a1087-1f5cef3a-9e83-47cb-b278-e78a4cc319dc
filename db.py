"""SQLite persistence layer for the author-identity review app.

Design notes
------------
* One SQLite database file (``DB_PATH``) holds everything; WAL mode is used so
  that readers (e.g. a CSV export) never block writers and vice versa.
* ``records``  : rows imported from the CSV (unique record id, source, name,
  org).  A row may be *disabled* (``enabled=0``) after a whole-source refresh
  no longer lists it.  Disabled rows keep their data and all of their
  verdicts; they just stop taking part in grouping until they reappear.
* ``verdicts`` : *every* direct human judgement ("same person" / "different
  person") between two record ids.  Judgements are append-only, except that a
  user may revoke a specific judgement.  Crucially a "same" judgement is never
  folded into an existing group and thrown away: it stays as a real row so the
  grouping can be rebuilt from the surviving judgements after a revoke.
* Grouping is therefore *derived*, never stored: we rebuild connected
  components from the currently active "same" edges on every read.  Only
  verdicts whose **both** endpoints are enabled and not revoked take part, so
  a disabled record can never act as a bridge.  Re-enabling a record therefore
  automatically restores its surviving judgements.
* "Different" judgements are hard constraints.  A new "same" judgement is
  rejected if its two endpoints would end up in one component that also
  contains an active "different" edge.  The check and the insert happen inside
  a single ``BEGIN IMMEDIATE`` transaction, so two concurrent decisions cannot
  both slip past each other.
* Whole-source refresh works in two phases.  ``create_refresh_preview`` stores
  an immutable plan plus a fingerprint of all records/verdicts.  Confirmation
  only applies the plan if that fingerprint is still current, so any record or
  verdict change invalidates an old preview.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "data", "identity.db"))

# The whole app is served by one process (gunicorn with 1 worker / several
# threads), so a re-entrant lock serialises writers.  The SQLite transaction
# remains the source of truth and would also protect us against a second
# process.
_write_lock = threading.RLock()


class ConflictError(Exception):
    """The decision would put two records judged "different" into one group.

    ``decide()`` reports conflicts as a return value so the request can roll the
    transaction back cleanly; this exception type is kept for callers that
    prefer raising semantics and for symmetry with the other error types.
    """

    def __init__(self, conflict_pairs: list[tuple[str, str]], message: str = ""):
        self.conflict_pairs = conflict_pairs
        super().__init__(message or "conflicts with existing 'different' judgements")


class NotFoundError(Exception):
    pass


class ValidationError(Exception):
    pass


@dataclass(frozen=True)
class Record:
    record_id: str
    source: str
    name: str
    org: str
    enabled: bool = True


@dataclass(frozen=True)
class Verdict:
    id: int
    kind: str  # "same" | "different"
    a: str
    b: str
    created_at: str


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                source    TEXT NOT NULL,
                name      TEXT NOT NULL,
                org       TEXT NOT NULL,
                enabled   INTEGER NOT NULL DEFAULT 1,
                imported_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS verdicts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                kind       TEXT NOT NULL CHECK (kind IN ('same', 'different')),
                a          TEXT NOT NULL REFERENCES records(record_id),
                b          TEXT NOT NULL REFERENCES records(record_id),
                created_at TEXT NOT NULL,
                revoked_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_verdicts_active
                ON verdicts(revoked_at, kind);
            -- At most one *active* verdict per unordered record pair.  Rows are
            -- always stored with a < b, but min/max keeps legacy data safe too.
            CREATE UNIQUE INDEX IF NOT EXISTS uq_verdict_pair_active
                ON verdicts(min(a, b), max(a, b))
                WHERE revoked_at IS NULL;

            CREATE TABLE IF NOT EXISTS imports (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT,
                imported   INTEGER NOT NULL,
                updated    INTEGER NOT NULL,
                skipped    INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS refresh_previews (
                id          TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                plan_json   TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        # Migration for databases created before whole-source refresh existed.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(records)")}
        if "enabled" not in columns:
            conn.execute("ALTER TABLE records ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_records(rows: Iterable[Record], filename: str = "") -> dict:
    """Insert-or-update records in a single transaction.

    Record ids are globally unique and belong to exactly one source.  A normal
    import must never silently resurrect a disabled record (that is the
    whole-source refresh's job) nor move an id to another source; either
    condition rejects the whole batch.  Duplicate ids *within one file* keep
    the historical "last row wins" behaviour and are reported as skipped.
    """
    deduped: dict[str, Record] = {}
    duplicate_rows = 0
    for row in rows:
        if row.record_id in deduped:
            duplicate_rows += 1
        deduped[row.record_id] = row

    inserted = updated = 0
    with _write_lock, connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Validate every incoming id before writing anything, so a problem
            # on a later row cannot leave a half-imported batch behind.
            for rec in deduped.values():
                existing = conn.execute(
                    "SELECT source, enabled FROM records WHERE record_id=?", (rec.record_id,)
                ).fetchone()
                if existing is None:
                    continue
                if not existing["enabled"]:
                    raise ValidationError(
                        f"记录 {rec.record_id} 已停用，普通导入不能更新停用记录；"
                        f"请使用“整源刷新”恢复该来源。整批导入已拒绝，未写入任何记录。"
                    )
                if existing["source"] != rec.source:
                    raise ValidationError(
                        f"记录 {rec.record_id} 已属于来源“{existing['source']}”，"
                        f"不能改为“{rec.source}”。整批导入已拒绝，未写入任何记录。"
                    )

            for rec in deduped.values():
                cur = conn.execute("SELECT 1 FROM records WHERE record_id = ?", (rec.record_id,))
                if cur.fetchone():
                    conn.execute(
                        "UPDATE records SET source=?, name=?, org=? WHERE record_id=?",
                        (rec.source, rec.name, rec.org, rec.record_id),
                    )
                    updated += 1
                else:
                    conn.execute(
                        "INSERT INTO records (record_id, source, name, org, enabled) VALUES (?,?,?,?,1)",
                        (rec.record_id, rec.source, rec.name, rec.org),
                    )
                    inserted += 1
            conn.execute(
                "INSERT INTO imports (filename, imported, updated, skipped) VALUES (?,?,?,?)",
                (filename, inserted, updated, duplicate_rows),
            )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
    return {"inserted": inserted, "updated": updated, "skipped_duplicates": duplicate_rows}


def _row_to_record(row: sqlite3.Row) -> Record:
    return Record(row["record_id"], row["source"], row["name"], row["org"], bool(row["enabled"]))


def _row_to_verdict(row: sqlite3.Row) -> Verdict:
    return Verdict(row["id"], row["kind"], row["a"], row["b"], row["created_at"])


def list_records(search: str = "", enabled_only: bool = False) -> list[Record]:
    sql = "SELECT * FROM records"
    clauses = []
    params: list[str] = []
    if enabled_only:
        clauses.append("enabled = 1")
    if search:
        like = f"%{search}%"
        clauses.append("(record_id LIKE ? OR name LIKE ? OR org LIKE ?)")
        params.extend([like, like, like])
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY name, record_id"
    with connection() as conn:
        return [_row_to_record(r) for r in conn.execute(sql, params).fetchall()]


def list_sources() -> list[str]:
    with connection() as conn:
        return [r["source"] for r in conn.execute("SELECT DISTINCT source FROM records ORDER BY source")]


def get_record(record_id: str) -> Record:
    with connection() as conn:
        row = conn.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"unknown record id: {record_id}")
    return _row_to_record(row)


def list_verdicts(include_revoked: bool = False) -> list[Verdict]:
    sql = "SELECT * FROM verdicts"
    if not include_revoked:
        sql += " WHERE revoked_at IS NULL"
    sql += " ORDER BY id"
    with connection() as conn:
        return [_row_to_verdict(r) for r in conn.execute(sql).fetchall()]


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic representative keeps group ids stable.
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def _components(record_ids: Iterable[str], same_edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Map every *enabled* record id to a stable group key (smallest member id).

    Edges touching an id that is not in ``record_ids`` are ignored, which is
    how disabled records are kept from bridging groups.
    """
    ids = list(record_ids)
    id_set = set(ids)
    uf = _UnionFind(ids)
    for a, b in same_edges:
        if a in id_set and b in id_set:
            uf.union(a, b)
    return {rid: uf.find(rid) for rid in ids}


@dataclass
class GroupView:
    group_key: str
    member_ids: list[str]

    @property
    def label(self) -> str:
        return self.member_ids[0]


def _edges_from_verdicts(
    enabled: dict[str, bool], verdicts: Iterable[dict]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split active verdicts into (same_edges, different_pairs) that participate.

    A verdict participates only when both endpoints exist and are enabled.
    """
    same: list[tuple[str, str]] = []
    diff: list[tuple[str, str]] = []
    for v in verdicts:
        if v.get("revoked_at") is not None:
            continue
        a, b = v["a"], v["b"]
        if not (enabled.get(a) and enabled.get(b)):
            continue
        (same if v["kind"] == "same" else diff).append((a, b))
    return same, diff


def _state_conflicts(enabled: dict[str, bool], verdicts: Iterable[dict]) -> list[tuple[str, str]]:
    """Active 'different' pairs that share a component in the given state."""
    enabled_ids = [rid for rid, on in enabled.items() if on]
    same, diff = _edges_from_verdicts(enabled, verdicts)
    mapping = _components(enabled_ids, same)
    return sorted((a, b) for a, b in diff if mapping.get(a) == mapping.get(b))


def compute_groups() -> tuple[list[GroupView], dict[str, str]]:
    """Return (groups, record_id -> group_key) using participating verdicts."""
    data = snapshot()
    groups = [
        GroupView(g["group_key"], [r.record_id for r in g["records"]])
        for g in data["groups"]
    ]
    return groups, data["group_of"]


def group_summary() -> list[dict]:
    """Groups enriched with the record data, for display."""
    return snapshot()["groups"]


def _view_from_conn(conn: sqlite3.Connection) -> dict:
    """Build every read-model item from one consistent connection state."""
    records = [_row_to_record(r) for r in conn.execute("SELECT * FROM records")]
    active_rows = [
        _row_to_verdict(r)
        for r in conn.execute("SELECT * FROM verdicts WHERE revoked_at IS NULL ORDER BY id")
    ]
    verdict_rows = [
        dict(r)
        for r in conn.execute("SELECT * FROM verdicts WHERE revoked_at IS NULL ORDER BY id")
    ]

    records.sort(key=lambda r: (r.name, r.record_id))
    enabled = {r.record_id: r.enabled for r in records}
    enabled_records = [r for r in records if r.enabled]
    disabled_records = [r for r in records if not r.enabled]
    enabled_ids = {r.record_id for r in enabled_records}

    same_edges, diff_pairs = _edges_from_verdicts(enabled, verdict_rows)
    mapping = _components([r.record_id for r in enabled_records], same_edges)

    buckets: dict[str, list[str]] = defaultdict(list)
    for rid, key in mapping.items():
        buckets[key].append(rid)

    by_id = {r.record_id: r for r in records}
    participating = [
        v for v in active_rows if v.a in enabled_ids and v.b in enabled_ids
    ]
    participating_by_record: dict[str, list[Verdict]] = defaultdict(list)
    for v in participating:
        participating_by_record[v.a].append(v)
        participating_by_record[v.b].append(v)

    groups = []
    for key, members in sorted(buckets.items()):
        members = sorted(members)
        groups.append(
            {
                "group_key": key,
                "records": [by_id[m] for m in members],
                "verdicts": sorted(
                    {v.id: v for m in members for v in participating_by_record.get(m, [])}.values(),
                    key=lambda v: v.id,
                ),
                "different_to": _invariant_violations(members, diff_pairs),
            }
        )

    # Verdicts that are retained but currently dormant because at least one
    # endpoint is disabled.  They are shown on the page and can still be
    # revoked; they will take part again if the record reappears.
    dormant = [v for v in active_rows if not (enabled.get(v.a) and enabled.get(v.b))]
    dormant_by_record: dict[str, list[Verdict]] = defaultdict(list)
    for v in dormant:
        if not enabled.get(v.a):
            dormant_by_record[v.a].append(v)
        if not enabled.get(v.b):
            dormant_by_record[v.b].append(v)

    stats = {
        "records": len(enabled_records),
        "disabled": len(disabled_records),
        "groups": len(groups),
        "same": sum(1 for v in active_rows if v.kind == "same"),
        "different": sum(1 for v in active_rows if v.kind == "different"),
        "dormant": len(dormant),
    }

    return {
        "records": list(records),
        "all_records": records,
        "enabled_records": enabled_records,
        "disabled_records": disabled_records,
        "disabled_ids": {r.record_id for r in disabled_records},
        "groups": groups,
        "group_of": mapping,
        "verdicts": active_rows,
        "dormant_verdicts": dormant,
        "dormant_by_record": dict(dormant_by_record),
        "stats": stats,
    }


def snapshot(search: str = "") -> dict:
    """A consistent page-sized view, read from a single SQLite transaction.

    A concurrent refresh/import/decision/revoke therefore cannot make the page
    mix pre- and post-change records, verdicts or groups.
    """
    with connection() as conn:
        conn.execute("BEGIN")
        try:
            data = _view_from_conn(conn)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    if search:
        needle = search.casefold()
        data["records"] = [
            r
            for r in data["all_records"]
            if needle in r.record_id.casefold()
            or needle in r.name.casefold()
            or needle in r.org.casefold()
        ]
    else:
        data["records"] = list(data["all_records"])
    return data


def _invariant_violations(members: list[str], diff_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Active 'different' pairs whose two endpoints ended up in the same group.

    A correct run never produces one; the check exists so a hand-edited database
    or a future bug is surfaced in the UI instead of silently producing bad
    mappings.
    """
    member_set = set(members)
    return [(a, b) for a, b in diff_pairs if a in member_set and b in member_set]


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def _enabled_map(conn: sqlite3.Connection) -> dict[str, bool]:
    return {
        r["record_id"]: bool(r["enabled"])
        for r in conn.execute("SELECT record_id, enabled FROM records")
    }


def _active_verdict_dicts(conn: sqlite3.Connection) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT id, kind, a, b, created_at, revoked_at FROM verdicts "
            "WHERE revoked_at IS NULL ORDER BY id"
        )
    ]


def _same_conflict_pairs(
    conn: sqlite3.Connection, a: str, b: str
) -> list[tuple[str, str]]:
    """Active different-pairs that would end up inside the merged component.

    Returns the offending record pairs so the UI can name them.  Both records
    of every pair lie in the merged component (component(a) ∪ component(b)).
    The caller must already have revoked any previous verdict for (a, b).
    """
    enabled = _enabled_map(conn)
    verdicts = _active_verdict_dicts(conn)
    same, diff = _edges_from_verdicts(enabled, verdicts)
    enabled_ids = [rid for rid, on in enabled.items() if on]
    mapping = _components(enabled_ids, same + [(a, b)])
    merged_root = mapping[a]
    conflicts = []
    for x, y in diff:
        if mapping.get(x) == merged_root and mapping.get(y) == merged_root:
            conflicts.append((x, y))
    return conflicts


def _different_conflict_pairs(
    conn: sqlite3.Connection, a: str, b: str
) -> list[tuple[str, str]]:
    """A 'different' judgement is impossible if a and b already share a group.

    The caller must already have revoked any previous verdict for (a, b);
    otherwise the pair's own old 'same' edge would make this always fail.
    """
    enabled = _enabled_map(conn)
    verdicts = _active_verdict_dicts(conn)
    same, _ = _edges_from_verdicts(enabled, verdicts)
    enabled_ids = [rid for rid, on in enabled.items() if on]
    mapping = _components(enabled_ids, same)
    if mapping.get(a) == mapping.get(b):
        return [(a, b)]
    return []


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def decide(a: str, b: str, kind: str) -> dict:
    """Record a 'same' or 'different' judgement between two enabled records.

    Re-deciding a pair drops the previous *active* verdict for that pair while
    keeping it in the table as revoked history.  Revoking the old row happens
    *before* the conflict check, so switching a pair from "same" to
    "different" is judged against the remaining constraints only.

    The whole operation runs under ``BEGIN IMMEDIATE`` plus the process lock.
    Disabled records cannot receive new judgements; their stored judgements
    stay visible (and revocable) but are dormant.
    """
    if kind not in ("same", "different"):
        raise ValidationError(f"invalid verdict kind: {kind}")
    if a == b:
        raise ValidationError("a record cannot be judged against itself")

    with _write_lock, connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for rid in (a, b):
                row = conn.execute(
                    "SELECT enabled FROM records WHERE record_id=?", (rid,)
                ).fetchone()
                if row is None:
                    raise NotFoundError(f"unknown record id: {rid}")
                if not row["enabled"]:
                    raise ValidationError(
                        f"记录 {rid} 已停用，不能新增判定。请先通过“整源刷新”恢复该记录。"
                    )

            lo, hi = _pair_key(a, b)
            # Drop this pair's previous active verdict first: it must not take
            # part in the constraint check for the new decision.
            conn.execute(
                "UPDATE verdicts SET revoked_at=datetime('now') "
                "WHERE revoked_at IS NULL AND min(a,b)=? AND max(a,b)=?",
                (lo, hi),
            )
            if kind == "same":
                conflicts = _same_conflict_pairs(conn, a, b)
            else:
                conflicts = _different_conflict_pairs(conn, a, b)
            if conflicts:
                conn.execute("ROLLBACK")
                return {
                    "ok": False,
                    "reason": "conflict",
                    "conflict_pairs": conflicts,
                }

            cur = conn.execute(
                "INSERT INTO verdicts (kind, a, b, created_at) VALUES (?,?,?,datetime('now'))",
                (kind, lo, hi),
            )
            conn.execute("COMMIT")
            return {"ok": True, "verdict_id": cur.lastrowid, "kind": kind, "a": lo, "b": hi}
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise


def revoke(verdict_id: int) -> dict:
    """Mark one verdict revoked.  All other verdicts stay exactly as they are.

    Works for verdicts touching disabled records too, so an operator can clean
    up dormant judgements without reactivating the record.
    """
    with _write_lock, connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM verdicts WHERE id=?", (verdict_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"unknown verdict id: {verdict_id}")
            if row["revoked_at"] is not None:
                conn.execute("COMMIT")
                return {"ok": True, "already_revoked": True, "verdict_id": verdict_id}
            conn.execute(
                "UPDATE verdicts SET revoked_at=datetime('now') WHERE id=?", (verdict_id,)
            )
            conn.execute("COMMIT")
            return {"ok": True, "already_revoked": False, "verdict_id": verdict_id}
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise


# ---------------------------------------------------------------------------
# Whole-source refresh (preview + confirm)
# ---------------------------------------------------------------------------

def _fingerprint(conn: sqlite3.Connection) -> str:
    """Deterministic digest of every record and verdict row.

    Previews store this digest; confirmation recomputes it inside the write
    transaction.  Any record or verdict change between the two phases makes an
    old confirmation invalid.
    """
    h = hashlib.sha256()
    for row in conn.execute(
        "SELECT record_id, source, name, org, enabled FROM records ORDER BY record_id"
    ):
        h.update(
            json.dumps(
                [row["record_id"], row["source"], row["name"], row["org"], row["enabled"]],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        h.update(b"\n")
    for row in conn.execute(
        "SELECT id, kind, a, b, created_at, revoked_at FROM verdicts ORDER BY id"
    ):
        h.update(
            json.dumps(
                [row["id"], row["kind"], row["a"], row["b"], row["created_at"], row["revoked_at"]],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def _preview_counts(plan: dict) -> dict:
    return {
        "inserted": len(plan["inserts"]),
        "updated": len([u for u in plan["updates"] if u["action"] == "update"]),
        "reactivated": len([u for u in plan["updates"] if u["action"] == "reactivate"]),
        "deactivated": len(plan["deactivates"]),
        "unchanged": len(plan["unchanged"]),
    }


def create_refresh_preview(source: str, rows: Iterable[Record], filename: str = "") -> dict:
    """Validate a complete roster and persist an immutable refresh plan.

    Rules enforced here (all reject the *whole* batch):
      * every data row's ``source`` equals the selected source;
      * no record id appears twice in the file;
      * no record id is already owned by a different source.
    """
    source = (source or "").strip()
    if not source:
        raise ValidationError("请选择或填写要刷新的来源。")
    rows = list(rows)

    with _write_lock, connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = {
                r["record_id"]: dict(r)
                for r in conn.execute("SELECT * FROM records")
            }

            errors: list[str] = []
            seen: set[str] = set()
            dupes: list[str] = []
            for rec in rows:
                if rec.record_id in seen and rec.record_id not in dupes:
                    dupes.append(rec.record_id)
                seen.add(rec.record_id)
                if rec.source != source:
                    errors.append(
                        f"记录 {rec.record_id} 的来源为“{rec.source}”，"
                        f"与所选来源“{source}”不一致"
                    )
                owner = current.get(rec.record_id)
                if owner is not None and owner["source"] != source:
                    errors.append(
                        f"记录 {rec.record_id} 已被来源“{owner['source']}”占用，"
                        f"不能在本来源下使用"
                    )
            if dupes:
                preview = "、".join(dupes[:10])
                more = f" 等 {len(dupes)} 个" if len(dupes) > 10 else ""
                errors.insert(0, f"文件内存在重复记录ID：{preview}{more}")
            if errors:
                shown = "；".join(errors[:8])
                more = f"（另有 {len(errors) - 8} 处）" if len(errors) > 8 else ""
                raise ValidationError(f"整源刷新被拒绝，未做任何修改：{shown}{more}")

            incoming = {rec.record_id: rec for rec in rows}
            inserts: list[dict] = []
            updates: list[dict] = []
            unchanged: list[dict] = []

            for rec in sorted(rows, key=lambda r: r.record_id):
                owner = current.get(rec.record_id)
                if owner is None:
                    inserts.append(
                        {
                            "record_id": rec.record_id,
                            "source": source,
                            "name": rec.name,
                            "org": rec.org,
                        }
                    )
                    continue
                changed = owner["name"] != rec.name or owner["org"] != rec.org
                before = {
                    "name": owner["name"],
                    "org": owner["org"],
                    "enabled": bool(owner["enabled"]),
                }
                if not owner["enabled"]:
                    updates.append(
                        {
                            "record_id": rec.record_id,
                            "action": "reactivate",
                            "name": rec.name,
                            "org": rec.org,
                            "before": before,
                            "changed": changed,
                        }
                    )
                elif changed:
                    updates.append(
                        {
                            "record_id": rec.record_id,
                            "action": "update",
                            "name": rec.name,
                            "org": rec.org,
                            "before": before,
                            "changed": True,
                        }
                    )
                else:
                    unchanged.append(
                        {
                            "record_id": rec.record_id,
                            "name": rec.name,
                            "org": rec.org,
                        }
                    )

            deactivates = [
                {"record_id": rid, "name": row["name"], "org": row["org"]}
                for rid, row in sorted(current.items())
                if row["source"] == source
                and bool(row["enabled"])
                and rid not in incoming
            ]

            plan = {
                "source": source,
                "filename": filename,
                "inserts": inserts,
                "updates": updates,
                "unchanged": unchanged,
                "deactivates": deactivates,
            }
            counts = _preview_counts(plan)

            # Prospective conflict check: a reactivated record restores its
            # surviving verdicts, which may merge a "different" pair.
            post_enabled = {rid: bool(row["enabled"]) for rid, row in current.items()}
            for ins in inserts:
                post_enabled[ins["record_id"]] = True
            for upd in updates:
                post_enabled[upd["record_id"]] = True
            for dea in deactivates:
                post_enabled[dea["record_id"]] = False
            conflicts = _state_conflicts(post_enabled, _active_verdict_dicts(conn))

            preview_id = secrets.token_urlsafe(24)
            conn.execute(
                "DELETE FROM refresh_previews WHERE created_at < datetime('now','-1 day')"
            )
            conn.execute(
                "INSERT INTO refresh_previews (id, source, fingerprint, plan_json) "
                "VALUES (?,?,?,?)",
                (
                    preview_id,
                    source,
                    _fingerprint(conn),
                    json.dumps(plan, ensure_ascii=False),
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

    return {
        "preview_id": preview_id,
        "source": source,
        "plan": plan,
        "counts": counts,
        "conflicts": conflicts,
    }


def confirm_refresh(preview_id: str) -> dict:
    """Apply a previously previewed whole-source refresh atomically.

    The stored fingerprint is checked inside the write transaction.  If any
    record or verdict changed after the preview, the confirmation is refused
    and the user must preview again.  If applying the plan would put an active
    "different" pair into one group, the entire refresh is rolled back.
    """
    preview_id = (preview_id or "").strip()
    if not preview_id:
        raise ValidationError("缺少预览标识，请重新预览。")

    with _write_lock, connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM refresh_previews WHERE id=?", (preview_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise NotFoundError("预览不存在或已被使用，请重新预览。")

            if _fingerprint(conn) != row["fingerprint"]:
                conn.execute("DELETE FROM refresh_previews WHERE id=?", (preview_id,))
                conn.execute("COMMIT")
                return {"ok": False, "reason": "stale", "source": row["source"]}

            plan = json.loads(row["plan_json"])
            source = plan["source"]
            counts = _preview_counts(plan)

            for ins in plan["inserts"]:
                conn.execute(
                    "INSERT INTO records (record_id, source, name, org, enabled) "
                    "VALUES (?,?,?,?,1)",
                    (ins["record_id"], source, ins["name"], ins["org"]),
                )
            for upd in plan["updates"]:
                conn.execute(
                    "UPDATE records SET source=?, name=?, org=?, enabled=1 "
                    "WHERE record_id=?",
                    (source, upd["name"], upd["org"], upd["record_id"]),
                )
            for dea in plan["deactivates"]:
                conn.execute(
                    "UPDATE records SET enabled=0 "
                    "WHERE record_id=? AND source=? AND enabled=1",
                    (dea["record_id"], source),
                )

            # Re-check the complete invariant from the just-written state.  A
            # conflict means the whole batch is undone, keeping the pre-refresh
            # records, enabled flags and verdicts exactly as they were.
            records_enabled = _enabled_map(conn)
            conflicts = _state_conflicts(records_enabled, _active_verdict_dicts(conn))
            if conflicts:
                conn.execute("ROLLBACK")
                return {
                    "ok": False,
                    "reason": "conflict",
                    "source": source,
                    "conflict_pairs": conflicts,
                    "counts": counts,
                }

            conn.execute("DELETE FROM refresh_previews WHERE id=?", (preview_id,))
            conn.execute(
                "INSERT INTO imports (filename, imported, updated, skipped) VALUES (?,?,?,?)",
                (
                    f"refresh:{source}:{plan.get('filename', '')}",
                    counts["inserted"],
                    counts["updated"] + counts["reactivated"],
                    counts["deactivated"],
                ),
            )
            conn.execute("COMMIT")
            return {"ok": True, "source": source, "counts": counts}
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_rows() -> list[tuple[str, str]]:
    """(record_id, group_key) for every *enabled* record, one snapshot.

    A read transaction pins a consistent view of both tables, so a concurrent
    decision or refresh can never produce a CSV that mixes pre- and
    post-change groups.  Disabled records are omitted entirely.
    """
    with connection() as conn:
        conn.execute("BEGIN")
        try:
            records = [
                dict(r)
                for r in conn.execute("SELECT record_id, enabled FROM records")
            ]
            verdicts = _active_verdict_dicts(conn)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    enabled = {r["record_id"]: bool(r["enabled"]) for r in records}
    enabled_ids = [r["record_id"] for r in records if r["enabled"]]
    same, _ = _edges_from_verdicts(enabled, verdicts)
    mapping = _components(enabled_ids, same)
    return [(rid, mapping[rid]) for rid in sorted(mapping, key=lambda r: (mapping[r], r))]
