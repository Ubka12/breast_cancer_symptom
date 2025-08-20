import requests, json

URL = "http://127.0.0.1:8000/check"
cases = [
    ("EMPTY", ""),
    ("W4: bloody nipple discharge", "I noticed bloody nipple discharge today"),
    ("W4: lump in armpit", "new lump in my armpit"),
    ("W4: skin dimpling/red", "breast skin looks dimpled and red"),
    ("W3: nipple inversion", "my nipple looks pulled in / inverted"),
    ("W3: size/shape change", "I noticed a change in the size and shape of my breast"),
    ("W2: persistent pain", "persistent breast pain for two weeks"),
    ("W1: tenderness", "mild tenderness"),
    ("BERT: orange peel", "my breast skin looks like orange peel"),
    ("VAGUE: i feel odd", "I feel odd"),
]

for label, text in cases:
    r = requests.post(URL, json={"symptoms": text})
    try:
        j = r.json()
    except Exception:
        j = {"_error": r.text}
    print(f"\n=== {label} ===")
    print("INPUT:", text)
    print("STATUS:", r.status_code)
    print(json.dumps(j, indent=2))
