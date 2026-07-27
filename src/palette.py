"""Single source of truth for demo and figure colors.

Three colors carry the whole story, so they are chosen for separability rather
than for taste and they are validated rather than eyeballed:

  * ground-truth tumor        blue
  * distortion-optimal tumor  orange   (the model that erases)
  * tumor-aware tumor         aqua     (the model that preserves)

Healthy tissue ("no tumor") is deliberately NOT a fourth hue. It is a neutral
glass shell, so every saturated pixel on screen means tumor.

Uncertainty answers a different question (how much, not which one), so it gets a
sequential one-hue ramp instead of a categorical hue. The ramp is violet because
blue is already spent on the ground truth and an adjacent panel showing two
different meanings in one hue is the confusion we are trying to avoid.

Validation (OKLab dE x100, colorblind simulation, WCAG contrast):
  categorical trio, all pairs, dark viewport #0C1220
      worst CVD dE 9.4 (deutan), worst normal-vision dE 20.9, all >= 3:1  PASS
  categorical trio, all pairs, light card #F1F1F0
      worst CVD dE 9.2 (deutan), worst normal-vision dE 24.0             PASS
      orange 2.83:1 and aqua 2.49:1 fall below 3:1, so every swatch on the
      page must carry a visible text label. It does; do not remove them.
  uncertainty ramp: lightness strictly monotonic across all 7 steps        PASS

Red and green are avoided as a pair on purpose. The previous palette used
#35C089 against #E8694A, which sits at dE 8.0 for deutan and 5.0 for tritan,
so a red-green colorblind viewer could not reliably tell "tumor preserved" from
"tumor erased" (the single most important distinction in the demo).
"""

from __future__ import annotations

# --- categorical: the three tumor versions -----------------------------------
# Fixed slot order (blue, orange, aqua). Never reassign these to other entities
# and never cycle in a fourth hue; fold extra versions into small multiples.

# On the dark 3D viewport.
DARK = {
    "true": "#3987e5",         # ground-truth tumor
    "distortion": "#d95926",   # distortion-optimal prediction (erases)
    "tumor_aware": "#199e70",  # tumor-aware prediction (preserves)
}

# On the light HTML card.
LIGHT = {
    "true": "#2a78d6",
    "distortion": "#eb6834",
    "tumor_aware": "#1baf7a",
}

# --- neutral: healthy tissue, "no tumor" -------------------------------------
BG_DARK = "#0C1220"        # deep navy viewport
BRAIN_NEUTRAL = "#C9CEDA"  # glass brain shell; carries no identity
BRAIN_OPACITY = 0.17

# --- sequential: magnitude, not identity -------------------------------------
# Both ramps run light to dark, evenly spaced in OKLab lightness. They appear
# only in views that carry no categorical mark, so they cannot be confused with
# the three tumor hues above.

# Uncertainty. Violet, because blue is spent on the ground truth.
UNCERTAINTY_RAMP = [
    "#dbd9ff", "#bfbbff", "#a59df9", "#8d80eb", "#7666d5", "#6050b5", "#4a3c92",
]

# Reconstruction error against the true HR image. Blue, the default sequential
# hue, which reads as distance from the truth.
ERROR_RAMP = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]


def uncertainty_colors(on_dark: bool = True) -> list[str]:
    """Ramp ordered so that low uncertainty recedes into the background.

    On the dark viewport that means low = dark violet and high = bright violet.
    On a light page it is the reverse. Getting this backwards makes a confident
    reconstruction the loudest thing on screen, which inverts the message.
    """
    return list(reversed(UNCERTAINTY_RAMP)) if on_dark else list(UNCERTAINTY_RAMP)


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, colors)


def uncertainty_cmap(on_dark: bool = True, name: str = "uncertainty"):
    """The same ramp as a matplotlib colormap, for PyVista and for figures."""
    return _cmap(uncertainty_colors(on_dark), name)


def error_cmap(name: str = "error"):
    """Sequential blue for |error| panels on a light figure background."""
    return _cmap(ERROR_RAMP, name)


# --- matplotlib figures on white -------------------------------------------- #
# Figures sit on a white axes background, so they use the light variants.
FIG = dict(LIGHT)
FIG["low_res"] = "#6A6E73"  # the degraded input: neutral, it is not a model
