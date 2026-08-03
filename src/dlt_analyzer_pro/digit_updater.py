from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from urllib.parse import urlencode

from .database import Database
from .models import DigitDraw
from .updater import API_URL, SOURCE_NAME, UpdateError, fetch_json


# ``35`` is the API identifier for 排列三.  The official history endpoint
# uses ``350133`` for 排列五; using 35 returns three-digit records that are
# correctly rejected by the PL5 parser but leaves an empty local history.
PL5_GAME_NO = "350133"
PL5_SOURCE_PAGE_URL = "https://www.lottery.gov.cn/kj/kjlb.html?plw"
DEFAULT_START_ISSUE = "04001"


@dataclass(frozen=True, slots=True)
class DigitUpdateResult:
    fetched: int
    pl5_added: int
    pl5_updated: int
    pl3_added: int
    pl3_updated: int
    latest_remote_issue: str | None
    checked_pages: int
    full_sync: bool
    source_name: str = SOURCE_NAME


def build_pl5_api_url(page_no: int, page_size: int = 100) -> str:
    query = urlencode(
        {
            "gameNo": PL5_GAME_NO,
            "provinceId": "0",
            "pageSize": str(max(1, min(int(page_size), 100))),
            "isVerify": "1",
            "pageNo": str(max(1, int(page_no))),
        }
    )
    return f"{API_URL}?{query}"


def parse_pl5_record(record: dict) -> DigitDraw:
    issue = str(record.get("lotteryDrawNum", "")).strip()
    result = str(record.get("lotteryDrawResult", "")).strip()
    draw_time = str(record.get("lotteryDrawTime", "")).strip()
    values = [int(value) for value in result.replace(",", " ").split() if value]
    if len(values) != 5:
        raise UpdateError(f"{issue or '未知期号'} 的排列5开奖号码不是5位")

    parsed_date = None
    if draw_time:
        try:
            parsed_date = datetime.strptime(draw_time[:10], "%Y-%m-%d").date()
        except ValueError as exc:
            raise UpdateError(f"{issue} 的开奖日期无法识别：{draw_time}") from exc

    draw = DigitDraw(game="pl5", issue=issue, draw_date=parsed_date, digits=tuple(values))
    draw.validate()
    return draw


def parse_pl5_payload(payload: dict) -> list[DigitDraw]:
    if payload.get("success") is False:
        message = payload.get("errorMessage") or payload.get("message") or "未知错误"
        raise UpdateError(f"中国体彩网接口返回失败：{message}")
    value = payload.get("value")
    if not isinstance(value, dict):
        raise UpdateError("中国体彩网排列5接口缺少 value 数据")
    records = value.get("list") or []
    if not isinstance(records, list):
        raise UpdateError("中国体彩网排列5接口 list 数据格式异常")

    draws: list[DigitDraw] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        status = record.get("lotteryDrawStatus")
        if status not in (None, 20, "20"):
            continue
        try:
            draws.append(parse_pl5_record(record))
        except (ValueError, UpdateError):
            continue
    return draws


def derive_pl3(draw: DigitDraw) -> DigitDraw:
    if draw.game != "pl5" or len(draw.digits) != 5:
        raise ValueError("排列3只能由有效的排列5记录生成")
    result = DigitDraw(
        game="pl3",
        issue=draw.issue,
        draw_date=draw.draw_date,
        digits=tuple(draw.digits[:3]),
    )
    result.validate()
    return result


class OfficialDigitUpdater:
    """Synchronize PL5 history and derive PL3 from the same official draw."""

    def __init__(
        self,
        database: Database,
        fetcher: Callable[[str], dict] = fetch_json,
    ) -> None:
        self.database = database
        self.fetcher = fetcher

    def update(self, max_pages: int = 30) -> DigitUpdateResult:
        latest = self.database.latest_digit_issue("pl5")
        lower_bound = int(latest) + 1 if latest and latest.isdigit() else 1
        return self._sync(lower_bound, max_pages=max_pages, full_sync=False)

    def sync_all(
        self,
        start_issue: str = DEFAULT_START_ISSUE,
        max_pages: int = 100,
    ) -> DigitUpdateResult:
        text = str(start_issue).strip()
        if not text.isdigit():
            raise ValueError(f"无效期号：{start_issue}")
        return self._sync(int(text), max_pages=max_pages, full_sync=True)

    def _sync(
        self,
        start_issue: int,
        max_pages: int,
        full_sync: bool,
    ) -> DigitUpdateResult:
        existing_pl5 = {draw.issue for draw in self.database.all_digit_draws("pl5")}
        existing_pl3 = {draw.issue for draw in self.database.all_digit_draws("pl3")}
        collected: dict[str, DigitDraw] = {}
        latest_remote: str | None = None
        checked_pages = 0

        for page_no in range(1, max(1, int(max_pages)) + 1):
            payload = self.fetcher(build_pl5_api_url(page_no))
            checked_pages += 1
            page_draws = parse_pl5_payload(payload)
            if not page_draws:
                break
            if latest_remote is None:
                latest_remote = max((draw.issue for draw in page_draws), key=int)
            smallest = min(int(draw.issue) for draw in page_draws)
            for draw in page_draws:
                if int(draw.issue) >= start_issue:
                    collected[draw.issue] = draw
            if smallest <= start_issue:
                break

        pl5_draws = sorted(collected.values(), key=lambda item: int(item.issue))
        if full_sync and not pl5_draws:
            raise UpdateError("中国体彩网没有返回排列5历史数据")
        pl3_draws = [derive_pl3(draw) for draw in pl5_draws]

        pl5_added = sum(draw.issue not in existing_pl5 for draw in pl5_draws)
        pl3_added = sum(draw.issue not in existing_pl3 for draw in pl3_draws)
        if pl5_draws:
            self.database.upsert_digit_draws(pl5_draws)
            self.database.upsert_digit_draws(pl3_draws)

        return DigitUpdateResult(
            fetched=len(pl5_draws),
            pl5_added=pl5_added,
            pl5_updated=len(pl5_draws) - pl5_added,
            pl3_added=pl3_added,
            pl3_updated=len(pl3_draws) - pl3_added,
            latest_remote_issue=latest_remote,
            checked_pages=checked_pages,
            full_sync=full_sync,
        )
