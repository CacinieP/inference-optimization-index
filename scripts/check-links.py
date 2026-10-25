#!/usr/bin/env python3
"""检查仓库内 Markdown 的外链是否可访问。

用法:
    python3 scripts/check-links.py               # 检查全部 md
    python3 scripts/check-links.py README.md     # 只检查指定文件
    python3 scripts/check-links.py --timeout 25

退出码:
    0  没有死链（403/429/超时归为 WARN，不算死链）
    1  存在死链（4xx/5xx）
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL_RE = re.compile(r"https?://[^\s<>\"'`)\]}]+")
TRIM = ".,;:!?、。，；：）)]}”\"'"
SKIP_HOSTS = (
    "img.shields.io",
    "shields.io",
    "badgen.net",
    "travis-ci",
    "127.0.0.1",
    "localhost",
    "0.0.0.0",
)
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def collect(paths: list[Path]) -> dict[str, set[str]]:
    urls: dict[str, set[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in URL_RE.finditer(text):
            url = match.group(0).rstrip(TRIM)
            if any(host in url for host in SKIP_HOSTS):
                continue
            urls.setdefault(url, set()).add(str(path))
    return urls


def probe(url: str, timeout: float) -> tuple[str, str]:
    """返回 (状态, 说明)。状态: OK / WARN / FAIL。"""
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": UA,
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(
            req, timeout=timeout, context=ssl.create_default_context()
        ) as resp:
            try:
                resp.read(2048)  # 只读头部片段：不支持 Range 的站点（如国内出版社）也能过
            except Exception:
                pass
            return "OK", str(resp.status)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 405, 429):
            return "WARN", f"{exc.code} 疑似反爬"
        return "FAIL", str(exc.code)
    except Exception as exc:  # 超时 / SSL / DNS
        return "WARN", f"{type(exc).__name__}: {str(exc)[:60]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="要检查的 md 文件，默认全部")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--show-warn", action="store_true", help="显示 WARN 明细")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    paths = (
        [Path(f).resolve() for f in args.files]
        if args.files
        else sorted(root.rglob("*.md"))
    )
    urls = collect(paths)
    print(f"扫描 {len(paths)} 个文件，发现 {len(urls)} 个外链\n")

    results: list[tuple[str, str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(probe, url, args.timeout): url for url in urls}
        for fut in concurrent.futures.as_completed(futures):
            url = futures[fut]
            results.append((*fut.result(), url))

    results.sort(key=lambda row: (row[0] != "FAIL", row[0] != "WARN", row[2]))
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]

    for status, detail, url in results:
        if status == "OK":
            continue
        if status == "WARN" and not args.show_warn:
            continue
        print(f"[{status}] {detail:<28} {url}")

    print("\n--- 汇总 ---")
    print(f"OK   : {len(results) - len(fails) - len(warns)}")
    print(f"WARN : {len(warns)}（403/429/超时，需人工复核）")
    print(f"FAIL : {len(fails)}（死链）")

    if fails:
        print("\n死链清单：")
        for _, detail, url in fails:
            print(f"  {detail:<12} {url}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())