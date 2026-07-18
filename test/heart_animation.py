"""Terminal heart animation"""

import math
import time
import os
import sys


def heart(x, y):
    """Heart shape equation"""
    return (x**2 + y**2 - 1)**3 - x**2 * y**3


def render_frame(scale, beat, t):
    """Render one frame of the beating heart"""
    rows = []
    s = scale * beat
    for j in range(15, -15, -1):
        row = ""
        for i in range(-30, 31):
            x = i / (s * 2)
            y = j / s
            if heart(x, y) <= 0:
                # Color gradient based on distance from center
                d = math.sqrt(x**2 + y**2)
                if d < 0.3:
                    row += "\033[91m@\033[0m"
                elif d < 0.6:
                    row += "\033[91m#\033[0m"
                elif d < 0.9:
                    row += "\033[31m*\033[0m"
                else:
                    row += "\033[31m.\033[0m"
            else:
                row += " "
        rows.append(row)
    return rows


def sparkle_positions(t, width=61, height=30):
    """Generate random sparkle positions around the heart"""
    import random
    random.seed(int(t * 3))
    positions = []
    for _ in range(8):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        positions.append((y, x))
    return positions


def main():
    fps = 15
    frame_time = 1.0 / fps
    t = 0

    os.system("clear" if os.name != "nt" else "cls")
    sys.stdout.write("\033[?25l")  # hide cursor

    try:
        while True:
            # Heartbeat effect
            beat = 1.0 + 0.08 * math.sin(t * 4) * math.exp(-((t * 4 % (2 * math.pi) - math.pi) ** 2))
            beat += 0.03 * math.sin(t * 8)

            frame = render_frame(10, beat, t)
            sparkles = sparkle_positions(t)

            # Build output
            sys.stdout.write("\033[H")  # move to top-left
            for y, line in enumerate(frame):
                sys.stdout.write(line + "\n")

            # Messages
            messages = [
                "  Love is in the code  ",
                "  while(alive) { love++; }  ",
                "  git commit -m 'I love you'  ",
                "  import heart; heart.beat()  ",
            ]
            idx = int(t / 3) % len(messages)
            msg = messages[idx]

            # Fade effect
            phase = (t % 3) / 3
            if phase < 0.15:
                alpha = phase / 0.15
            elif phase > 0.85:
                alpha = (1 - phase) / 0.15
            else:
                alpha = 1.0

            if alpha > 0.5:
                sys.stdout.write(f"\n\033[95m{msg:^61}\033[0m\n")
            else:
                sys.stdout.write(f"\n{' ':^61}\n")

            # Floating hearts
            hearts_line = ""
            for i in range(7):
                offset = math.sin(t * 2 + i * 1.3) * 3
                pos = int(8 * i + offset + 5)
                hearts_line += " " * max(0, pos - len(hearts_line))
                symbols = ["\033[91m<3\033[0m", "\033[95m<3\033[0m", "\033[31m<3\033[0m"]
                hearts_line += symbols[i % 3]
            sys.stdout.write(hearts_line + "\n")

            sys.stdout.flush()
            time.sleep(frame_time)
            t += frame_time

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # show cursor


if __name__ == "__main__":
    main()
