"""Export DealRocket assistant dialogs from the support admin panel.

The admin session cookie is read from DEALROCKET_ADMIN_COOKIE and is never written to disk.
Dialogs are anonymous; the output stays in data/raw/, which is not committed.
"""

import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ADMIN_API = "https://support.dealrocket.ru/admin/api"


def get(path, cookie):
    request = Request(
        f"{ADMIN_API}/{path}", headers={"cookie": cookie, "accept": "application/json"}
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/dealrocket_admin")
    parser.add_argument("--pause", type=float, default=0.4, help="seconds between API calls")
    args = parser.parse_args()
    cookie = os.environ["DEALROCKET_ADMIN_COOKIE"]
    args.output.mkdir(parents=True, exist_ok=True)

    listing, page = [], 1
    while True:
        chunk = get(f"conversations?errors_only=false&page={page}&page_size=30", cookie)
        listing.extend(chunk["items"])
        if not chunk["items"] or len(listing) >= chunk["total"]:
            break
        page += 1
        time.sleep(args.pause)

    assistant = [item for item in listing if "assistant" in (item.get("modules") or [])]
    with (args.output / "conversations.jsonl").open("w", encoding="utf-8") as file:
        for item in assistant:
            detail = get(f"conversations/{item['id']}", cookie)
            file.write(json.dumps(detail, ensure_ascii=False) + "\n")
            time.sleep(args.pause)
    print(f"dialogs: {len(listing)}, assistant dialogs saved: {len(assistant)}")


if __name__ == "__main__":
    main()
