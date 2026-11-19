"""Print rows as a plain aligned text table."""

from __future__ import annotations


def print_table(headers: list[str], rows: list[list[object]], *, empty: str = "(none)") -> None:
    if not rows:
        print(empty)
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)).rstrip())
