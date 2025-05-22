"""
core.db  (ver.2)
SQLite3 でシミュレーション結果を保存

runs       … 入力パラメータ（1行 / run）
summary    … run 全体の接触回数など集約値（1行 / run）
contacts   … 接触イベント（複数行 / run）
figures    … 生成したグラフ・動画のパス
"""
from __future__ import annotations
import sqlite3, json, pathlib, datetime as dt
from typing import Iterable, Sequence

# DB ファイルはプロジェクト直下 results.db に固定
DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "results.db"

DDL = {
"runs": """
CREATE TABLE IF NOT EXISTS runs(
    run_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    shape    TEXT NOT NULL,
    params   TEXT NOT NULL,
    created  TEXT NOT NULL
);""",
"summary": """
CREATE TABLE IF NOT EXISTS summary(
    run_id   INTEGER PRIMARY KEY,
    contact_count INTEGER,
    note     TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);""",
"contacts": """
CREATE TABLE IF NOT EXISTS contacts(
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER,
    sperm_idx INTEGER,
    step     INTEGER,
    time_s   REAL,
    x REAL, y REAL, z REAL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);""",
"figures": """
CREATE TABLE IF NOT EXISTS figures(
    fig_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER,
    kind     TEXT,
    path     TEXT,
    created  TEXT,
    note     TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);"""
}

def _conn():
    """DDL を保証したうえでコネクションを返す"""
    conn = sqlite3.connect(DB_PATH)
    for ddl in DDL.values(): conn.execute(ddl)
    return conn

# ---------- 保存ヘルパ -----------------
def save_run_meta(constants: dict) -> int:
    """runs に 1 行挿入して run_id を返す"""
    conn=_conn(); cur=conn.cursor()
    cur.execute("INSERT INTO runs(shape,params,created) VALUES(?,?,?)",
                (constants["shape"],
                 json.dumps(constants, separators=(',', ':')),
                 dt.datetime.now().isoformat(timespec='seconds')))
    run_id = cur.lastrowid
    conn.commit(); conn.close()
    return run_id

def save_summary(run_id: int, contact_count: int, note: str = ""):
    conn=_conn()
    conn.execute("""INSERT OR REPLACE INTO summary(run_id,contact_count,note)
                    VALUES(?,?,?)""",
                 (run_id, contact_count, note))
    conn.commit(); conn.close()

def save_contacts(run_id: int, rows: Iterable[Sequence]):
    """
    rows = [(sperm_idx, step, time_s, x, y, z), ...]  ← 6 要素
    """
    conn = _conn(); cur = conn.cursor()

    cur.executemany(
        "INSERT INTO contacts(run_id,sperm_idx,step,time_s,x,y,z) "
        "VALUES(?,?,?,?,?,?,?)",
        [(run_id, *r) for r in rows]          # ← run_id + 6 要素 = 7
    )
    conn.commit(); conn.close()


def save_figure(run_id: int, kind: str, path: str, note: str = ""):
    conn=_conn()
    conn.execute("""INSERT INTO figures(run_id,kind,path,created,note)
                    VALUES(?,?,?,?,?)""",
                 (run_id, kind, path,
                  dt.datetime.now().isoformat(timespec='seconds'),
                  note))
    conn.commit(); conn.close()
