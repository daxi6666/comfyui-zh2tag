# -*- coding: utf-8 -*-
"""添加词条（跨平台，可用于任意安装路径）

用法：
  python add_word.py                          # 交互模式
  python add_word.py 花海 "flower field" pos   # 直接添加
  python add_word.py 水印 watermark neg
  python add_word.py 亮 bright auto
"""
import argparse
import os
import sqlite3
import sys

sys.stdout.reconfigure(errors="replace") if hasattr(sys.stdout, "reconfigure") else None

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def resolve_db():
    env = os.environ.get("PT_DB_PATH")
    if env:
        return env
    p = os.path.join(_REPO, "data", "prompt_db.db")
    if os.path.exists(p):
        return p
    return os.path.join(_REPO, "data", "prompt_db.db")


DB = resolve_db()

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS prompt_map ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, zh TEXT UNIQUE, "
    "tags TEXT, hits INTEGER DEFAULT 0, created TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "polarity TEXT DEFAULT 'pos')"
)


def ensure(conn):
    conn.execute(SCHEMA)
    conn.commit()


def upsert(conn, zh, tags, polarity, category=None):
    cat = category or {"pos": "自定义", "neg": "负面", "auto": "语境"}.get(polarity, "自定义")
    cur = conn.cursor()
    cur.execute("SELECT tags, polarity FROM prompt_map WHERE zh = ?", (zh,))
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE prompt_map SET tags = ?, polarity = ?, category = ? WHERE zh = ?",
            (tags, polarity, cat, zh),
        )
        conn.commit()
        print(f"  已更新: {zh} -> {tags} [{polarity}]")
    else:
        cur.execute(
            "INSERT INTO prompt_map (category, zh, tags, polarity) VALUES (?,?,?,?)",
            (cat, zh, tags, polarity),
        )
        conn.commit()
        print(f"  已添加: {zh} -> {tags} [{polarity}]")


def show(conn, polarity=None):
    cur = conn.cursor()
    if polarity:
        cur.execute("SELECT count(*) FROM prompt_map WHERE polarity = ?", (polarity,))
    else:
        cur.execute("SELECT count(*) FROM prompt_map")
    total = cur.fetchone()[0]
    cur.execute("SELECT polarity, count(*) FROM prompt_map GROUP BY polarity")
    dist = ", ".join(f"{p}={n}" for p, n in cur.fetchall())
    print(f"\n  词库: {DB}")
    print(f"  总数: {total}    {dist}")


def main():
    ap = argparse.ArgumentParser(description="给词库添加词条")
    ap.add_argument("zh", nargs="?", help="中文短语")
    ap.add_argument("tags", nargs="?", help="英文标签")
    ap.add_argument("polarity", nargs="?", default="pos", choices=["pos", "neg", "auto"],
                    help="pos=正向 neg=负面 auto=按语境")
    ap.add_argument("--category", default=None, help="自定义分类名")
    ap.add_argument("--db", default=None, help="指定数据库路径")
    args = ap.parse_args()

    global DB
    if args.db:
        DB = args.db
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)

    conn = sqlite3.connect(DB)
    ensure(conn)

    if args.zh and args.tags:
        upsert(conn, args.zh.strip(), args.tags.strip(), args.polarity, args.category)
    else:
        print("添加词条（中文留空回车 = 退出）")
        print("类型: 1=正面  2=负面  3=语境（回车默认 1）")
        while True:
            try:
                zh = input("\n中文: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not zh:
                break
            tags = input("标签(英文): ").strip()
            if not tags:
                print("  标签为空，跳过")
                continue
            pol = input("类型 [1/2/3]: ").strip() or "1"
            polarity = {"1": "pos", "2": "neg", "3": "auto"}.get(pol, "pos")
            upsert(conn, zh, tags, polarity, args.category)

    show(conn)
    conn.close()


if __name__ == "__main__":
    main()
