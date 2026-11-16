"""앱의 계산을 numpy/scipy 결과와 대조하고 화면을 캡처한다."""
import json, sys
import numpy as np
from scipy import stats
from playwright.sync_api import sync_playwright

from pathlib import Path
APP = (Path(__file__).resolve().parent.parent / "index.html").as_uri()
fails = []

def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1, abs(b))

def check(name, ok, detail=""):
    print(("OK   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fails.append(name)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1100, "height": 900})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(APP)

    # ---------- 반복 측정 ----------
    for data in [[1.42, 1.45, 1.40, 1.44, 1.43, 1.41], [2.31, 2.28, 2.35, 2.30, 2.33, 2.29],
                 [0.452, 0.447, 0.461, 0.455, 0.449], [101.3, 99.8, 100.4]]:
        r = pg.evaluate(f"analyzeRepeat({json.dumps(data)}, 'se', 0, 1)")
        a = np.array(data)
        check(f"mean {data[:2]}", close(r["mean"], a.mean()))
        check(f"sd {data[:2]}", close(r["sd"], a.std(ddof=1)))
        check(f"se {data[:2]}", close(r["se"], a.std(ddof=1) / np.sqrt(len(a))))

    # 반올림 규칙
    cases = [  # (u, k) -> (value, lsd)
        (0.00667, 1, 0.007, -3), (0.0096, 1, 0.01, -2), (0.0146, 2, 0.015, -3),
        (0.95, 1, 1.0, 0), (0.96, 1, 1.0, 0), (34.7, 1, 30, 1), (0.0125, 2, 0.013, -3),
    ]
    for u, k, val, lsd in cases:
        r = pg.evaluate(f"roundUncertainty({u}, {k})")
        check(f"roundUncertainty({u},{k})", close(r["value"], val) and r["lsd"] == lsd, str(r))
    for x, lsd, s in [(1.425, -2, "1.43"), (1.005, -2, "1.01"), (2.3100000000000005, -3, "2.310"),
                      (1.4249999999999998, -2, "1.43"), (1234.5, 1, "1230"), (-0.0455, -3, "-0.046")]:
        r = pg.evaluate(f"fixedAt({x}, {lsd})")
        check(f"fixedAt({x},{lsd})", r == s, r)

    # ---------- 직선 맞춤 ----------
    sets = [
        ([0.10, 0.20, 0.30, 0.40, 0.50, 0.60], [1.02, 2.05, 2.98, 4.03, 5.01, 5.96]),
        ([0.020, 0.041, 0.059, 0.081, 0.100, 0.121], [0.49, 0.98, 1.47, 1.96, 2.45, 2.94]),
        ([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [v * v for v in [0.90, 1.10, 1.27, 1.42, 1.55, 1.68]]),
    ]
    for xs, ys in sets:
        r = pg.evaluate(f"fitLine({json.dumps(xs)}, {json.dumps(ys)}, 'ab')")
        lr = stats.linregress(xs, ys)
        check("slope", close(r["a"], lr.slope))
        check("intercept", close(r["b"], lr.intercept))
        check("slope stderr", close(r["sa"], lr.stderr))
        check("intercept stderr", close(r["sb"], lr.intercept_stderr))
        check("R2", close(r["R2"], lr.rvalue ** 2))
        # 원점 통과
        r0 = pg.evaluate(f"fitLine({json.dumps(xs)}, {json.dumps(ys)}, 'a0')")
        X = np.array(xs)[:, None]; Y = np.array(ys)
        a0, res, *_ = np.linalg.lstsq(X, Y, rcond=None)
        ssr = float(((Y - a0[0] * X[:, 0]) ** 2).sum())
        sa0 = np.sqrt(ssr / (len(xs) - 1)) / np.sqrt((X[:, 0] ** 2).sum())
        check("origin slope", close(r0["a"], a0[0]))
        check("origin slope stderr", close(r0["sa"], sa0))

    # ---------- 유효숫자 세기 ----------
    for s, sf in [("0.00450", 3), ("1200", 2), ("1200.", 4), ("1.20e3", 3), ("58.44", 4),
                  ("0.2500", 4), ("100.0", 4), ("5", 1), ("0.5", 1), ("6.022e23", 4)]:
        r = pg.evaluate(f"countSig({json.dumps(s)})")
        check(f"countSig {s}", r["sf"] == sf, str(r))

    # ---------- 유효숫자 식 ----------
    exprs = [
        ("(51.05 - 48.126) / 58.44 / 0.2500", "0.200"),
        ("(53.94 - 50.218) / 74.55 / 0.1000", "0.499"),
        ("{4} × pi^2 × 0.500 ÷ 1.42^2", "9.79"),
        ("12.3 * 4.56", "56.1"),
        ("1.2 + 3.456", "4.7"),
        ("100.0 / 3.0", "33"),
        ("2.0 * 600", "1 × 10³"),
        ("9.99 * 1.0", "10."),  # 아래에서 따로 판단
        ("6.02e23 * 2.0", "1.2 × 10²⁴"),
        ("5.01 - 5.0", "0 (유효숫자 없음)"),
    ]
    for e, want in exprs:
        r = pg.evaluate(f"""(() => {{ const r = evaluateSig({json.dumps(e)}); const q = r.result;
            return {{ v: q.v, sf: q.sf, txt: q.exact ? 'exact' : (q.sf >= 1 ? formatSig(q.v, q.sf) : '0 (유효숫자 없음)') }}; }})()""")
        if e == "9.99 * 1.0":
            check(f"sig {e}", r["txt"] in ("10", "1.0 × 10¹"), str(r))
        else:
            check(f"sig {e}", r["txt"] == want, str(r))

    for e in ["(1+2", "3 +", "abc", "1/0"]:
        msg = pg.evaluate(f"(() => {{ try {{ evaluateSig({json.dumps(e)}); return 'no error'; }} catch (x) {{ return x.message; }} }})()")
        check(f"error for {e!r}", msg != "no error", msg)

    check("no page errors", not errors, "; ".join(errors))
    b.close()

print("\nFAILED:" if fails else "\nALL PASSED", fails if fails else "")
