from typing import Callable
import numpy as np

IntensitiesFn = Callable[[int, np.ndarray], np.ndarray]


def identity(step: int, full_resources: np.ndarray) -> np.ndarray:
    """Intensitites are always set to full_resources."""
    return full_resources


def weakly_intensities(
    step: int,
    full_resources: np.ndarray,
    weekly_intensities: np.ndarray = [0.6, 0.7, 0.8, 0.8, 0.9, 1],
) -> np.ndarray:
    """Changes intensities every week for 6 weeks in a row as follows (in default):
    WEEK 1: 60%
    WEEK 2: 70%
    WEEK 3: 80%
    WEEK 4: 80%
    WEEK 5: 90%
    WEEK 6: 100%
    """

    week_id = int((step - (step % 7)) / 7)
    # print(f"week_id: {week_id}")
    # print(
    #    f"step: {step} resources: {full_resources} and intensities: {weekly_intensities[week_id] * full_resources}"
    # )
    return weekly_intensities[week_id] * full_resources


INTENSITIES: dict[str, IntensitiesFn] = {
    "identity": identity,
    "weakly": weakly_intensities,
}
