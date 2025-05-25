#!/usr/bin/env python3
"""
CLI で結果を集計・可視化するユーティリティ
-------------------------------------------------
  python tools/analyze.py --list
  python tools/analyze.py --group-by volume,sperm_conc
  python tools/analyze.py --hist 12      # run_id=12 の接触時刻ヒスト
"""
from __future__ import annotations
import argparse, sqlite3, json, pathlib, sys
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "results.db"

# ----------------------------------------------------------------------
def _conn():
    return sqlite3.connect(DB_PATH)

def list_runs(limit:int=20):
    cur=_conn().cursor()
    cur.execute("SELECT run_id,shape,created FROM runs ORDER BY run_id DESC LIMIT ?",(limit,))
    rows=cur.fetchall()
    print(" run_id │ shape │ created")
    print("─"*32)
    for r in rows:
        print(f"{r[0]:>6} │ {r[1]:<5} │ {r[2]}")

def group_stats(cols: list[str]):
    sel = ", ".join([f"json_extract(params,'$.{c}')" for c in cols])
    # 修正 ↓  列番号を文字列にして join
    group = ", ".join(str(i + 1) for i in range(len(cols)))

    q = f"""
    SELECT {sel},
           AVG(contact_count),
           printf('±%.2f',
             CASE WHEN COUNT(*)>1
                  THEN sqrt(AVG(contact_count*contact_count)
                       -   AVG(contact_count)*AVG(contact_count))
                  ELSE 0 END),
           COUNT(*)
      FROM runs JOIN summary USING(run_id)
     GROUP BY {group}
     ORDER BY {group};
    """
    cur = _conn().cursor()
    cur.execute(q)
    rows = cur.fetchall()

    header = " | ".join(cols + ["mean", "sd", "n"])
    print(header); print("-" * len(header))
    for r in rows:
        print(" | ".join(map(str, r)))

def hist_run(run_id:int, bins:int=20):
    cur=_conn().cursor()
    cur.execute("SELECT time_s FROM contacts WHERE run_id=? ORDER BY time_s;",(run_id,))
    times=[t[0] for t in cur.fetchall()]
    if not times:
        print("No contacts for run",run_id); return
    times=np.array(times)
    plt.figure(figsize=(10, 4))
    plt.hist(times,bins=bins,color="orange",edgecolor="k")
    plt.title(f"Run {run_id} – Contact time histogram")
    plt.xlabel("Time (s)"); plt.ylabel("Count")
    out=pathlib.Path(f"hist_run{run_id}.png"); plt.savefig(out,dpi=120)
    print("Saved histogram →",out)

# ----------------------------------------------------------------------
def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--list",action="store_true",help="最新 20 件を一覧")
    p.add_argument("--group-by",metavar="COLS",
                   help="カンマ区切り列で平均±SD(contact_count) を計算")
    p.add_argument("--hist",type=int,metavar="RUN_ID",
                   help="run_id の接触時刻ヒストグラムを PNG 保存")
    args=p.parse_args(argv)

    if args.list:
        list_runs(); return
    if args.group_by:
        cols=args.group_by.split(",")
        group_stats(cols); return
    if args.hist is not None:
        hist_run(args.hist); return
    p.print_help()

if __name__=="__main__":
    main()
