from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Draw:
    issue: str
    draw_date: date | None
    front: tuple[int, int, int, int, int]
    back: tuple[int, int]

    def validate(self) -> None:
        if not self.issue.strip():
            raise ValueError("期号不能为空")
        if len(self.front) != 5 or len(set(self.front)) != 5:
            raise ValueError(f"前区必须是5个不重复号码：{self.front}")
        if len(self.back) != 2 or len(set(self.back)) != 2:
            raise ValueError(f"后区必须是2个不重复号码：{self.back}")
        if not all(1 <= n <= 35 for n in self.front):
            raise ValueError(f"前区号码必须在1至35之间：{self.front}")
        if not all(1 <= n <= 12 for n in self.back):
            raise ValueError(f"后区号码必须在1至12之间：{self.back}")
        if tuple(sorted(self.front)) != self.front:
            raise ValueError("前区号码必须升序")
        if tuple(sorted(self.back)) != self.back:
            raise ValueError("后区号码必须升序")


@dataclass(frozen=True, slots=True)
class Prediction:
    front: tuple[int, int, int, int, int]
    back: tuple[int, int]
    score: float
    strategy: str


@dataclass(frozen=True, slots=True)
class BacktestDetail:
    issue: str
    model_front_hits: int
    model_back_hits: int
    random_front_hits: int
    random_back_hits: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    evaluated: int
    model_front_average: float
    model_back_average: float
    random_front_average: float
    random_back_average: float
    details: tuple[BacktestDetail, ...]


@dataclass(frozen=True, slots=True)
class DigitDraw:
    game: str
    issue: str
    draw_date: date | None
    digits: tuple[int, ...]

    def validate(self) -> None:
        game = self.game.lower().strip()
        expected = 3 if game == "pl3" else 5 if game == "pl5" else 0
        if expected == 0:
            raise ValueError(f"不支持的排列玩法：{self.game}")
        if not self.issue.strip():
            raise ValueError("期号不能为空")
        if len(self.digits) != expected:
            raise ValueError(f"{game} 必须包含 {expected} 位号码：{self.digits}")
        if not all(0 <= number <= 9 for number in self.digits):
            raise ValueError(f"排列号码必须在0至9之间：{self.digits}")

    @property
    def number_text(self) -> str:
        return "".join(str(number) for number in self.digits)


@dataclass(frozen=True, slots=True)
class DigitPrediction:
    game: str
    digits: tuple[int, ...]
    score: float
    strategy: str
    model_mode: str = "统计融合"

    @property
    def number_text(self) -> str:
        return "".join(str(number) for number in self.digits)


@dataclass(frozen=True, slots=True)
class DigitBacktestDetail:
    issue: str
    model_hits: int
    random_hits: int
    exact_model: bool
    exact_random: bool


@dataclass(frozen=True, slots=True)
class DigitBacktestResult:
    game: str
    evaluated: int
    model_average_hits: float
    random_average_hits: float
    model_exact_hits: int
    random_exact_hits: int
    position_model_rates: tuple[float, ...]
    position_random_rates: tuple[float, ...]
    details: tuple[DigitBacktestDetail, ...]
