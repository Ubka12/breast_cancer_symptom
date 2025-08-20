# backend/smoke_check_all.py
import json, urllib.request, urllib.error

BASE = "http://127.0.0.1:8000"

def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return r.status, r.read()

def post_json(path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())

def assert_(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("✔", msg)

def main():
    print("\n=== ROUTE CHECKS ===")
    for p in ["/","/about","/symptoms","/selfcheck","/support","/disclaimer","/index.html","/about.html"]:
        status, _ = get(p)
        assert_(status==200, f"GET {p} -> 200")

    print("\n=== API /check ===")

    status, r = post_json("/check", {"symptoms": ""})
    assert_(r["method"]=="none", "Empty -> method none")
    assert_(r["risk"]=="LOW", "Empty -> LOW")

    status, r = post_json("/check", {"symptoms":"bloody nipple discharge today"})
    assert_(r["method"]=="rule-based", "Bloody discharge -> rule-based")
    assert_(r["risk"]=="HIGH", "Bloody discharge -> HIGH")

    status, r = post_json("/check", {"symptoms":"new lump in my armpit"})
    assert_(r["method"]=="rule-based", "Lump armpit -> rule-based")
    assert_(r["risk"]=="HIGH", "Lump armpit -> HIGH")

    status, r = post_json("/check", {"symptoms":"breast skin looks dimpled and red"})
    assert_(r["method"]=="rule-based", "Skin dimpling/red -> rule-based")
    assert_(r["risk"]=="HIGH", "Skin dimpling/red -> HIGH")

    status, r = post_json("/check", {"symptoms":"my nipple looks pulled in / inverted"})
    assert_(r["method"]=="rule-based", "Nipple inversion -> rule-based")
    assert_(r["risk"]=="MEDIUM", "Nipple inversion -> MEDIUM")

    status, r = post_json("/check", {"symptoms":"change in size and shape of my breast"})
    assert_(r["method"]=="rule-based", "Size/shape -> rule-based")
    assert_(r["risk"]=="MEDIUM", "Size/shape -> MEDIUM")

    status, r = post_json("/check", {"symptoms":"persistent breast pain for two weeks"})
    assert_(r["method"]=="rule-based", "Persistent pain -> rule-based")
    assert_(r["risk"]=="MEDIUM", "Persistent pain -> MEDIUM")

    status, r = post_json("/check", {"symptoms":"mild tenderness"})
    assert_(r["method"]=="rule-based", "Tenderness -> rule-based")
    assert_(r["risk"]=="LOW", "Tenderness -> LOW")

    status, r = post_json("/check", {"symptoms":"my breast skin looks like orange peel"})
    assert_(r["method"]=="bert", "Orange peel -> bert")
    assert_(r["risk"]=="HIGH", "Orange peel -> HIGH")
    assert_(float(r.get("similarity_score",0)) >= 0.6, "Orange peel -> sim >= 0.6")

    status, r = post_json("/check", {"symptoms":"I feel off"})
    assert_(r["method"]=="llm", "Vague -> llm")
    assert_(r["risk"] in ("LOW","MEDIUM","HIGH"), "Vague -> risk present")

    print("\nALL CHECKS PASSED ✅")

if __name__ == "__main__":
    main()
