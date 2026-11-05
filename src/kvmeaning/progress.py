import time


def _hms(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Progress:
    def __init__(self, label: str, total: int):
        self.label, self.total = label, total
        self.k, self.worked, self.start = 0, 0, time.time()
        self._last = self.start

    def step(self, item: str, note: str = "", skipped: bool = False) -> str:
        now = time.time()
        dt, self._last = now - self._last, now
        self.k += 1
        if not skipped:
            self.worked += 1
        elapsed = now - self.start
        # Skipped (resumed) items take no time, so they are left out of the rate.
        eta = elapsed / self.worked * (self.total - self.k) if self.worked else None
        line = (f"[{time.strftime('%H:%M:%S')}] {self.label} {self.k}/{self.total} {item}"
                f"{' skip (complete)' if skipped else f' {dt:.1f}s'}{' ' + note if note else ''}"
                f" elapsed={_hms(elapsed)} eta={'-' if eta is None else _hms(eta)}")
        print(line, flush=True)
        return line
