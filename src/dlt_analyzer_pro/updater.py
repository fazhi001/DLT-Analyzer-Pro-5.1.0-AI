from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import __version__
from .database import Database
from .models import Draw


SOURCE_NAME = "中国体彩网（国家体育总局体育彩票管理中心）"
SOURCE_PAGE_URL = "https://www.lottery.gov.cn/kj/kjlb.html?dlt"
API_URL = (
    "https://webapi.sporttery.cn/gateway/lottery/"
    "getHistoryPageListV1.qry"
)
GAME_NO = "85"
DEFAULT_START_ISSUE = "21001"
DEFAULT_END_ISSUE = "26081"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    f"Chrome/126.0 Safari/537.36 DLTAnalyzerPro/{__version__}"
)


class UpdateError(RuntimeError):
    """Raised when official draw data cannot be downloaded or validated."""


@dataclass(frozen=True, slots=True)
class UpdateResult:
    added: int
    updated: int
    fetched: int
    latest_remote_issue: str | None
    source_name: str
    checked_pages: int
    full_sync: bool = False


def build_api_url(page_no: int, page_size: int = 100) -> str:
    query = urlencode(
        {
            "gameNo": GAME_NO,
            "provinceId": "0",
            "pageSize": str(max(1, min(int(page_size), 100))),
            "isVerify": "1",
            "pageNo": str(max(1, int(page_no))),
        }
    )
    return f"{API_URL}?{query}"


def fetch_json(url: str, timeout: float = 20.0) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": SOURCE_PAGE_URL,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept-Encoding": "identity",
            "Connection": "close",
        },
    )
    try:
        with urlopen(
            request,
            timeout=timeout,
            context=ssl.create_default_context(),
        ) as response:
            raw = response.read()
    except HTTPError as exc:
        raise UpdateError(f"中国体彩网接口返回 HTTP {exc.code}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise UpdateError(f"无法连接中国体彩网：{reason}") from exc
    except TimeoutError as exc:
        raise UpdateError("连接中国体彩网超时") from exc

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            payload = json.loads(raw.decode(encoding))
            if not isinstance(payload, dict):
                raise UpdateError("中国体彩网接口返回格式异常")
            return payload
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            raise UpdateError("中国体彩网接口返回的不是有效 JSON") from exc
    raise UpdateError("中国体彩网接口字符编码无法识别")


def parse_api_record(record: dict) -> Draw:
    issue = str(record.get("lotteryDrawNum", "")).strip()
    result = str(record.get("lotteryDrawResult", "")).strip()
    draw_time = str(record.get("lotteryDrawTime", "")).strip()

    values = [int(value) for value in result.replace(",", " ").split() if value]
    if len(values) != 7:
        raise UpdateError(f"{issue or '未知期号'} 的开奖号码不是7个")

    parsed_date = None
    if draw_time:
        date_text = draw_time[:10]
        try:
            parsed_date = datetime.strptime(date_text, "%Y-%m-%d").date()
        except ValueError as exc:
            raise UpdateError(f"{issue} 的开奖日期无法识别：{draw_time}") from exc

    draw = Draw(
        issue=issue,
        draw_date=parsed_date,
        front=tuple(sorted(values[:5])),
        back=tuple(sorted(values[5:])),
    )
    draw.validate()
    return draw


def parse_api_payload(payload: dict) -> list[Draw]:
    if payload.get("success") is False:
        message = payload.get("errorMessage") or payload.get("message") or "未知错误"
        raise UpdateError(f"中国体彩网接口返回失败：{message}")

    value = payload.get("value")
    if not isinstance(value, dict):
        raise UpdateError("中国体彩网接口缺少 value 数据")

    records = value.get("list")
    if records is None:
        records = []
    if not isinstance(records, list):
        raise UpdateError("中国体彩网接口 list 数据格式异常")

    draws: list[Draw] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        status = record.get("lotteryDrawStatus")
        if status not in (None, 20, "20"):
            continue
        try:
            draws.append(parse_api_record(record))
        except (ValueError, UpdateError):
            continue
    return draws


class OfficialDrawUpdater:
    """Fetch DLT history from the JSON service used by the official page."""

    def __init__(
        self,
        database: Database,
        fetcher: Callable[[str], dict] = fetch_json,
    ) -> None:
        self.database = database
        self.fetcher = fetcher

    def update(self, max_pages: int = 30) -> UpdateResult:
        """Incrementally add draws newer than the latest local issue."""
        existing = {draw.issue for draw in self.database.all_draws()}
        latest_local = self.database.latest_issue()
        lower_bound = int(latest_local) + 1 if latest_local and latest_local.isdigit() else 1

        draws, latest_remote, checked = self._fetch_range(
            start_issue=lower_bound,
            end_issue=None,
            max_pages=max_pages,
        )
        added = sum(draw.issue not in existing for draw in draws)
        updated = len(draws) - added
        if draws:
            self.database.upsert_draws(draws)

        return UpdateResult(
            added=added,
            updated=updated,
            fetched=len(draws),
            latest_remote_issue=latest_remote,
            source_name=SOURCE_NAME,
            checked_pages=checked,
            full_sync=False,
        )

    def sync_range(
        self,
        start_issue: str = DEFAULT_START_ISSUE,
        end_issue: str = DEFAULT_END_ISSUE,
        max_pages: int = 100,
    ) -> UpdateResult:
        """
        Download and overwrite the requested official history range.

        This fills missing dates and corrects any local row whose issue already
        exists, while preserving draws outside the requested range.
        """
        start_number = self._issue_number(start_issue)
        end_number = self._issue_number(end_issue)
        if start_number > end_number:
            raise ValueError("开始期号不能大于结束期号")

        existing = {draw.issue for draw in self.database.all_draws()}
        draws, latest_remote, checked = self._fetch_range(
            start_issue=start_number,
            end_issue=end_number,
            max_pages=max_pages,
        )
        if not draws:
            raise UpdateError(
                f"中国体彩网没有返回 {start_issue}—{end_issue} 的开奖数据"
            )

        issues = {draw.issue for draw in draws}
        expected_start = str(start_number).zfill(5)
        expected_end = str(end_number).zfill(5)
        if expected_start not in issues or expected_end not in issues:
            raise UpdateError(
                f"官网数据范围不完整：实际取得 "
                f"{min(issues, key=int)}—{max(issues, key=int)}"
            )

        added = sum(draw.issue not in existing for draw in draws)
        updated = len(draws) - added
        self.database.upsert_draws(draws)

        return UpdateResult(
            added=added,
            updated=updated,
            fetched=len(draws),
            latest_remote_issue=latest_remote,
            source_name=SOURCE_NAME,
            checked_pages=checked,
            full_sync=True,
        )

    def _fetch_range(
        self,
        start_issue: int,
        end_issue: int | None,
        max_pages: int,
    ) -> tuple[list[Draw], str | None, int]:
        collected: dict[str, Draw] = {}
        latest_remote: str | None = None
        checked_pages = 0

        for page_no in range(1, max_pages + 1):
            payload = self.fetcher(build_api_url(page_no))
            checked_pages += 1
            page_draws = parse_api_payload(payload)
            if not page_draws:
                break

            if latest_remote is None:
                latest_remote = max((draw.issue for draw in page_draws), key=int)

            smallest = min(int(draw.issue) for draw in page_draws)
            for draw in page_draws:
                issue_number = int(draw.issue)
                if issue_number < start_issue:
                    continue
                if end_issue is not None and issue_number > end_issue:
                    continue
                collected[draw.issue] = draw

            if smallest <= start_issue:
                break

        result = sorted(collected.values(), key=lambda draw: int(draw.issue))
        return result, latest_remote, checked_pages

    @staticmethod
    def _issue_number(issue: str) -> int:
        text = str(issue).strip()
        if len(text) != 5 or not text.isdigit():
            raise ValueError(f"无效期号：{issue}")
        return int(text)
