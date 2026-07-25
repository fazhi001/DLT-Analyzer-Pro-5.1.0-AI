from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import __version__

# 中国大陆全年固定使用 UTC+8，无夏令时。使用固定偏移可避免 Windows
# 环境缺少 IANA 时区数据库时 ZoneInfo("Asia/Shanghai") 加载失败。
BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
UTC = timezone.utc

# 优先使用中国体彩网的响应时间校准；失败时软件继续使用本机时钟。
DEFAULT_TIME_SYNC_URLS = (
    "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?gameNo=85&provinceId=0&pageSize=1&isVerify=1&pageNo=1",
    "https://www.lottery.gov.cn/kj/kjlb.html?dlt",
)


def now_beijing() -> datetime:
    """Return an aware datetime in Beijing time based on the system clock."""
    return datetime.now(UTC).astimezone(BEIJING_TZ)


def format_beijing_now() -> str:
    return now_beijing().strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True, slots=True)
class NetworkTimeResult:
    utc_time: datetime
    source: str
    round_trip_seconds: float


class RealtimeBeijingClock:
    """
    A continuously ticking Beijing clock.

    The clock starts from the local system UTC time. After a successful HTTP
    time synchronization it advances from a monotonic clock, so changing the
    Windows time zone or manually adjusting the wall clock will not make the
    displayed time jump until the next synchronization.
    """

    def __init__(
        self,
        *,
        initial_utc: datetime | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._monotonic_fn = monotonic_fn
        base = initial_utc or datetime.now(UTC)
        if base.tzinfo is None:
            base = base.replace(tzinfo=UTC)
        self._base_utc = base.astimezone(UTC)
        self._base_monotonic = self._monotonic_fn()
        self._network_synced = False
        self._source = "本机时钟"
        self._last_sync_utc: datetime | None = None

    def utc_now(self) -> datetime:
        elapsed = max(0.0, self._monotonic_fn() - self._base_monotonic)
        return self._base_utc + timedelta(seconds=elapsed)

    def beijing_now(self) -> datetime:
        return self.utc_now().astimezone(BEIJING_TZ)

    def format_now(self) -> str:
        return self.beijing_now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def network_synced(self) -> bool:
        return self._network_synced

    @property
    def source(self) -> str:
        return self._source

    @property
    def last_sync_utc(self) -> datetime | None:
        return self._last_sync_utc

    def apply_network_time(self, result: NetworkTimeResult) -> None:
        value = result.utc_time
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        self._base_utc = value.astimezone(UTC)
        self._base_monotonic = self._monotonic_fn()
        self._network_synced = True
        self._source = result.source
        self._last_sync_utc = self._base_utc

    def source_label(self) -> str:
        return "网络校时" if self._network_synced else "本机时钟"


def fetch_http_network_time(
    url: str,
    *,
    timeout: float = 8.0,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> NetworkTimeResult:
    """Read the HTTP Date header and compensate approximately for latency."""
    request = Request(
        url,
        headers={
            "User-Agent": f"Mozilla/5.0 DLTAnalyzerPro/{__version__}",
            "Accept": "application/json,text/html,*/*",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
            "Connection": "close",
        },
        method="GET",
    )
    started = monotonic_fn()
    try:
        with urlopen(request, timeout=timeout) as response:
            date_header = response.headers.get("Date")
            # Read only a small amount; the Date header is all that is needed.
            response.read(1)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(str(reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("连接超时") from exc
    finished = monotonic_fn()

    if not date_header:
        raise RuntimeError("服务器未返回 Date 时间头")
    parsed = parsedate_to_datetime(date_header)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    round_trip = max(0.0, finished - started)
    corrected = parsed + timedelta(seconds=round_trip / 2.0)
    return NetworkTimeResult(
        utc_time=corrected,
        source=url,
        round_trip_seconds=round_trip,
    )


def synchronize_network_time(
    urls: Iterable[str] = DEFAULT_TIME_SYNC_URLS,
    *,
    timeout: float = 8.0,
    fetcher: Callable[..., NetworkTimeResult] = fetch_http_network_time,
) -> NetworkTimeResult:
    """Try time sources in order and return the first successful sample."""
    errors: list[str] = []
    for url in urls:
        try:
            return fetcher(url, timeout=timeout)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("；".join(errors) if errors else "没有可用的网络时间源")


def parse_database_timestamp(value: str | None) -> datetime | None:
    """
    Parse SQLite timestamps.

    SQLite CURRENT_TIMESTAMP is UTC and is normally stored as
    ``YYYY-MM-DD HH:MM:SS`` without a timezone suffix. A timezone-aware ISO
    timestamp is also accepted for forward compatibility.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def format_database_timestamp_beijing(value: str | None) -> str:
    """Convert a UTC database timestamp to Beijing time for display."""
    parsed = parse_database_timestamp(value)
    if parsed is None:
        return ""
    return parsed.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")
