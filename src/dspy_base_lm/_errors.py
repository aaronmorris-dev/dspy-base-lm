"""Error normalization for the custom provider boundary."""

from collections.abc import Callable
from types import TracebackType
from typing import Literal

import dspy


class ProviderErrorBoundary:
    """Preserve DSPy errors and normalize unknown provider failures."""

    def __init__(self, normalize: Callable[[Exception], dspy.LMUnexpectedError]) -> None:
        self._normalize: Callable[[Exception], dspy.LMUnexpectedError] = normalize

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        _ = exception_type, traceback
        if exception is None or isinstance(exception, dspy.LMError):
            return False
        if isinstance(exception, Exception):
            raise self._normalize(exception) from exception
        return False


class TrainingJobErrorBoundary:
    """Resolve a DSPy training job with a provider failure."""

    def __init__(self, job: dspy.TrainingJob) -> None:
        self._job: dspy.TrainingJob = job

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        _ = exception_type, traceback
        if isinstance(exception, Exception):
            self._job.set_result(exception)
            return True
        return False
