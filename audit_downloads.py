"""
Audit which S&P 500 filings were successfully downloaded.

Run at any time — including while build_graph.py is still downloading.
Prints a per-company status table and a summary.

Usage:
  python audit_downloads.py
"""

from pathlib import Path
from build_graph import fetch_sp500_tickers

FILINGS_DIR = Path("/Volumes/T7 Shield/c-o-investigator/data/filings/sec-edgar-filings")
FORMS = ("DEF 14A", "10-K")


def check_ticker(ticker: str) -> dict[str, str]:
    """
    Check download status for one ticker across all form types.

    Parameters:
        ticker : str - Stock ticker symbol.

    Returns:
        dict[str, str] - Map of form type → status string ('ok N files', 'missing', 'empty').

    Examples:
        check_ticker('AAPL')
        # {'DEF 14A': 'ok 1 files', '10-K': 'ok 1 files'}
    """
    result = {}
    for form in FORMS:
        form_dir = FILINGS_DIR / ticker / form
        if not form_dir.exists():
            result[form] = "missing"
            continue
        filing_dirs = [d for d in form_dir.iterdir() if d.is_dir()]
        if not filing_dirs:
            result[form] = "empty dir"
            continue
        # Count actual HTML files across all filing subdirs
        html_files = []
        for fd in filing_dirs:
            html_files.extend(list(fd.glob("*.htm")) + list(fd.glob("*.html")))
        result[form] = f"ok ({len(html_files)} file{'s' if len(html_files) != 1 else ''})"
    return result


def main() -> None:
    tickers = fetch_sp500_tickers()
    print(f"\nAuditing {len(tickers)} S&P 500 tickers in {FILINGS_DIR}\n")
    print(f"{'Ticker':<8}  {'DEF 14A':<20}  {'10-K':<20}")
    print("-" * 52)

    counts: dict[str, dict[str, int]] = {
        form: {"ok": 0, "missing": 0, "empty dir": 0} for form in FORMS
    }
    failures: list[str] = []

    for ticker in sorted(tickers):
        status = check_ticker(ticker)
        def14a = status.get("DEF 14A", "missing")
        tenk = status.get("10-K", "missing")

        has_any_problem = any(not s.startswith("ok") for s in status.values())
        if has_any_problem:
            failures.append(ticker)

        print(f"{ticker:<8}  {def14a:<20}  {tenk:<20}")

        for form in FORMS:
            s = status.get(form, "missing")
            if s.startswith("ok"):
                counts[form]["ok"] += 1
            elif s == "empty dir":
                counts[form]["empty dir"] += 1
            else:
                counts[form]["missing"] += 1

    print("\n" + "=" * 52)
    print("SUMMARY")
    print("=" * 52)
    for form in FORMS:
        c = counts[form]
        total = len(tickers)
        print(f"{form:<10}  ok: {c['ok']}/{total}  |  missing: {c['missing']}  |  empty: {c['empty dir']}")

    if failures:
        print(f"\nTickers with at least one missing/empty form ({len(failures)}):")
        print(" ".join(failures))
    else:
        print("\nAll tickers have all forms downloaded.")


if __name__ == "__main__":
    main()
