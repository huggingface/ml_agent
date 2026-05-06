"""Braille-character canvas for high-resolution terminal graphics.

Each terminal cell maps to a 2x4 dot grid using Unicode braille characters
(U+2800–U+28FF), giving 2× horizontal and 4× vertical resolution.
"""

# Braille dot positions:  (0,0) (1,0)    dots 1,4
#                         (0,1) (1,1)    dots 2,5
#                         (0,2) (1,2)    dots 3,6
#                         (0,3) (1,3)    dots 7,8
_DOT_MAP = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)


class BrailleCanvas:
    """A pixel canvas that renders to braille characters."""

    def __init__(self, term_width: int, term_height: int):
        self.term_width = term_width
        self.term_height = term_height
        self.pixel_width = term_width * 2
        self.pixel_height = term_height * 4
        self._buf = bytearray(term_width * term_height)

    def clear(self) -> None:
        for i in range(len(self._buf)):
            self._buf[i] = 0

    def set_pixel(self, x: int, y: int) -> None:
        if 0 <= x < self.pixel_width and 0 <= y < self.pixel_height:
            cx, rx = divmod(x, 2)
            cy, ry = divmod(y, 4)
            self._buf[cy * self.term_width + cx] |= _DOT_MAP[ry][rx]

    def render(self) -> list[str]:
        lines = []
        for row in range(self.term_height):
            offset = row * self.term_width
            line = "".join(
                chr(0x2800 + self._buf[offset + col]) for col in range(self.term_width)
            )
            lines.append(line)
        return lines


# ── Bitmap font (5×7 uppercase + digits) ──────────────────────────────

_FONT: dict[str, list[str]] = {}


def _define_font() -> None:
    """Define a simple 5×7 bitmap font for uppercase ASCII."""
    glyphs = {
        "A": [" ## ", "#  #", "#  #", "####", "#  #", "#  #", "#  #"],
        "B": ["### ", "#  #", "#  #", "### ", "#  #", "#  #", "### "],
        "C": [" ## ", "#  #", "#   ", "#   ", "#   ", "#  #", " ## "],
        "D": ["### ", "#  #", "#  #", "#  #", "#  #", "#  #", "### "],
        "E": ["####", "#   ", "#   ", "### ", "#   ", "#   ", "####"],
        "F": ["####", "#   ", "#   ", "### ", "#   ", "#   ", "#   "],
        "G": [" ## ", "#  #", "#   ", "# ##", "#  #", "#  #", " ###"],
        "H": ["#  #", "#  #", "#  #", "####", "#  #", "#  #", "#  #"],
        "I": ["###", " # ", " # ", " # ", " # ", " # ", "###"],
        "J": ["  ##", "  # ", "  # ", "  # ", "  # ", "# # ", " #  "],
        "K": ["#  #", "# # ", "##  ", "##  ", "# # ", "#  #", "#  #"],
        "L": ["#   ", "#   ", "#   ", "#   ", "#   ", "#   ", "####"],
        "M": ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"],
        "N": ["#  #", "## #", "## #", "# ##", "# ##", "#  #", "#  #"],
        "O": [" ## ", "#  #", "#  #", "#  #", "#  #", "#  #", " ## "],
        "P": ["### ", "#  #", "#  #", "### ", "#   ", "#   ", "#   "],
        "Q": [" ## ", "#  #", "#  #", "#  #", "# ##", "#  #", " ## "],
        "R": ["### ", "#  #", "#  #", "### ", "# # ", "#  #", "#  #"],
        "S": [" ## ", "#  #", "#   ", " ## ", "   #", "#  #", " ## "],
        "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
        "U": ["#  #", "#  #", "#  #", "#  #", "#  #", "#  #", " ## "],
        "V": ["#   #", "#   #", "#   #", " # # ", " # # ", "  #  ", "  #  "],
        "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
        "X": ["#  #", "#  #", " ## ", " ## ", " ## ", "#  #", "#  #"],
        "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
        "Z": ["####", "   #", "  # ", " #  ", "#   ", "#   ", "####"],
        " ": ["  ", "  ", "  ", "  ", "  ", "  ", "  "],
        "0": [" ## ", "#  #", "#  #", "#  #", "#  #", "#  #", " ## "],
        "1": [" # ", "## ", " # ", " # ", " # ", " # ", "###"],
        "2": [" ## ", "#  #", "   #", "  # ", " #  ", "#   ", "####"],
        "3": [" ## ", "#  #", "   #", " ## ", "   #", "#  #", " ## "],
        "4": ["#  #", "#  #", "#  #", "####", "   #", "   #", "   #"],
        "5": ["####", "#   ", "### ", "   #", "   #", "#  #", " ## "],
        "6": [" ## ", "#   ", "### ", "#  #", "#  #", "#  #", " ## "],
        "7": ["####", "   #", "  # ", " #  ", " #  ", " #  ", " #  "],
        "8": [" ## ", "#  #", "#  #", " ## ", "#  #", "#  #", " ## "],
        "9": [" ## ", "#  #", "#  #", " ###", "   #", "   #", " ## "],
    }
    _FONT.update(glyphs)


_define_font()


def text_to_pixels(text: str, scale: int = 1) -> list[tuple[int, int]]:
    """Convert text string to a list of (x, y) pixel positions using bitmap font."""
    pixels = []
    cursor_x = 0
    for ch in text.upper():
        glyph = _FONT.get(ch)
        if glyph is None:
            cursor_x += 4 * scale
            continue
        for row_idx, row in enumerate(glyph):
            for col_idx, cell in enumerate(row):
                if cell == "#":
                    for sy in range(scale):
                        for sx in range(scale):
                            pixels.append(
                                (cursor_x + col_idx * scale + sx, row_idx * scale + sy)
                            )
        glyph_width = max(len(r) for r in glyph)
        cursor_x += (glyph_width + 1) * scale
    return pixels
