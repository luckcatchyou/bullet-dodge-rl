import math


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def norm(x, y):
    length = math.hypot(x, y)
    return (0, 0) if length == 0 else (x / length, y / length)


def dist_circle_to_rect(cx, cy, rx, ry, rw, rh):
    closest_x = clamp(cx, rx - rw / 2, rx + rw / 2)
    closest_y = clamp(cy, ry - rh / 2, ry + rh / 2)
    return math.hypot(cx - closest_x, cy - closest_y)


def custom_linear_schedule(initial_lr: float, final_lr: float, total_steps: int, decay_steps: int):
    def schedule(progress_remaining: float) -> float:
        current_progress = 1.0 - progress_remaining
        current_steps = current_progress * total_steps

        if current_steps >= decay_steps:
            return final_lr

        decay_ratio = current_steps / decay_steps
        return initial_lr + decay_ratio * (final_lr - initial_lr)

    return schedule
