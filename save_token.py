import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE=os.path.dirname(os.path.abspath(__file__))
dst=os.path.join(HERE,"gh_token.txt")
try:
    import tkinter
    r=tkinter.Tk(); r.withdraw()
    t=r.clipboard_get()
    r.destroy()
except Exception as e:
    print("  读取剪贴板失败:", e); sys.exit(1)
t=(t or "").strip()
if not t:
    print("  剪贴板是空的，请先复制 Token"); sys.exit(1)
# 去掉可能混进来的空白/换行
t="".join(t.split())
ok = t.startswith("ghp_") or t.startswith("github_pat_") or t.startswith("gho_")
with open(dst,"w",encoding="utf-8",newline="") as f:
    f.write(t)
print(f"  已写入: {dst}")
print(f"  长度: {len(t)} 字符")
print(f"  前缀: {t[:4]}…{t[-4:]}")
print(f"  格式: {'✅ 像 GitHub Token' if ok else '⚠️ 不像 GitHub Token（ghp_ 开头才对）'}")