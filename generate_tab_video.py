#!/usr/bin/env python3
"""
Guitar Pro one-measure-per-line tab playhead renderer (v2 detector).

What it does
------------
1. Reads timing from a Guitar Pro MusicXML export.
2. Reads one or more Guitar Pro SVG page exports.
3. Detects each 6-line tablature staff automatically.
4. Treats each detected tab staff as ONE measure/system.
5. Detects the actual left/right barline clusters for every measure separately,
   so repeat barlines can change the usable playhead bounds.
6. Renders a video in which the correct measure is shown and a playhead moves
   across it with no Premiere keyframing.

The first thing to run is --analyze-only. It creates annotated PNGs showing
exactly what the script thinks each measure and its playhead bounds are.

Dependencies:
    python3 -m pip install pillow numpy cairosvg imageio-ffmpeg

Example:
    python3 generate_tab_video.py song.musicxml page1.svg page2.svg \
        --analyze-only --debug-dir tab_debug

Then render:
    python3 generate_tab_video.py song.musicxml page1.svg page2.svg \
        --output tab_playhead.mov --transparent --fps 60 \
        --width 3840 --height 720

Notes / current limitations
---------------------------
- Designed for ONE measure per line/system.
- Designed for a 6-string tablature staff. --strings can change this.
- Basic forward/backward repeats are expanded. Volta endings, D.S./D.C./Coda
  are detected and reported as unsupported rather than silently guessed.
- Timing comes from MusicXML. If the final performance is not locked to the
  Guitar Pro tempo map, align/sync separately or add timing anchors in a later
  revision.
"""

from __future__ import annotations

import argparse
import bisect
import io
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET

try:
    import numpy as np
    from PIL import Image, ImageColor, ImageDraw, ImageFont
    import cairosvg
except ImportError as exc:
    print(
        "Missing dependency. Install with:\n"
        "  python3 -m pip install pillow numpy cairosvg imageio-ffmpeg\n",
        file=sys.stderr,
    )
    raise


# ----------------------------- data models -----------------------------

@dataclass
class MeasureTiming:
    index: int
    number: str
    duration_quarters: float
    duration_seconds: float
    repeat_forward: bool = False
    repeat_backward: bool = False
    repeat_times: int = 2
    has_ending: bool = False
    is_empty: bool = False
    has_time_signature: bool = False
    note_onset_seconds: list[float] = field(default_factory=list)


@dataclass
class StaffGeometry:
    measure_index: int
    page_index: int
    system_index_on_page: int
    string_ys: list[int]
    spacing: float
    staff_left: int
    staff_right: int
    left_cluster_left: int
    left_cluster_right: int
    right_cluster_left: int
    right_cluster_right: int
    play_left: float
    play_right: float
    crop_box: tuple[int, int, int, int]
    note_breakpoints: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class RenderedMeasure:
    rgba: np.ndarray
    play_left: float
    play_right: float
    play_top: float
    play_bottom: float
    spacing_px: float
    note_breakpoints: list[tuple[float, float]]


# ----------------------------- utilities -----------------------------

def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_child(el: ET.Element, name: str) -> Optional[ET.Element]:
    for child in el:
        if local_name(child.tag) == name:
            return child
    return None


def child_text(el: ET.Element, name: str, default: Optional[str] = None) -> Optional[str]:
    child = first_child(el, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def grouped_runs(indices: Iterable[int]) -> list[tuple[int, int]]:
    vals = list(indices)
    if not vals:
        return []
    runs = []
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
        else:
            runs.append((start, prev))
            start = prev = v
    runs.append((start, prev))
    return runs


def close_small_false_gaps(signal: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False runs bounded by True if the gap is <= max_gap."""
    out = signal.copy()
    false_idx = np.flatnonzero(~signal)
    for a, b in grouped_runs(false_idx):
        if a == 0 or b == len(signal) - 1:
            continue
        if (b - a + 1) <= max_gap and signal[a - 1] and signal[b + 1]:
            out[a : b + 1] = True
    return out


def longest_true_run(signal: np.ndarray) -> tuple[int, int]:
    idx = np.flatnonzero(signal)
    runs = grouped_runs(idx)
    if not runs:
        raise RuntimeError("Could not find a continuous staff-line region.")
    return max(runs, key=lambda r: r[1] - r[0])


def parse_color(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    rgb = ImageColor.getrgb(value)
    if len(rgb) == 4:
        return rgb
    return rgb[0], rgb[1], rgb[2], alpha


def invert_to_transparent(img: Image.Image, fg_color: str) -> Image.Image:
    """Turn a page rendered as dark ink on a light/opaque background into a
    straight-alpha RGBA image where ink density becomes opacity and the
    background becomes fully transparent. The visible ink is recolored to
    ``fg_color`` (default white), so what remains reads as bright line art
    with no background, suitable for overlay on top of other footage.
    """
    gray = analysis_gray(img).astype(np.float32)
    alpha_out = np.clip(255.0 - gray, 0.0, 255.0)
    fg = np.array(ImageColor.getrgb(fg_color)[:3], dtype=np.float32)
    out = np.empty((*gray.shape, 4), dtype=np.uint8)
    out[..., 0] = fg[0]
    out[..., 1] = fg[1]
    out[..., 2] = fg[2]
    out[..., 3] = alpha_out.astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


# ----------------------------- MusicXML -----------------------------

def parse_musicxml(path: Path, part_id: Optional[str] = None) -> tuple[list[MeasureTiming], list[str]]:
    root = ET.parse(path).getroot()
    if local_name(root.tag) != "score-partwise":
        raise RuntimeError("Only score-partwise MusicXML is currently supported.")

    parts = [p for p in root if local_name(p.tag) == "part"]
    if not parts:
        raise RuntimeError("No <part> found in MusicXML.")

    if part_id:
        matching = [p for p in parts if p.attrib.get("id") == part_id]
        if not matching:
            ids = ", ".join(p.attrib.get("id", "?") for p in parts)
            raise RuntimeError(f"Part {part_id!r} not found. Available part ids: {ids}")
        part = matching[0]
    else:
        part = parts[0]

    divisions = 1
    beats = 4
    beat_type = 4
    current_tempo = 125.0
    saw_tempo = False
    warnings: list[str] = []
    result: list[MeasureTiming] = []

    for mi, measure in enumerate(c for c in part if local_name(c.tag) == "measure"):
        number = measure.attrib.get("number", str(mi + 1))
        cursor_div = 0
        max_cursor_div = 0
        tempo_events: list[tuple[float, float]] = []  # quarter offset, bpm

        repeat_forward = False
        repeat_backward = False
        repeat_times = 2
        has_ending = False
        seen_notes = 0
        seen_non_rest = 0
        measure_has_time_signature = False
        note_onsets: list[float] = []

        for child in measure:
            tag = local_name(child.tag)

            if tag == "note":
                seen_notes += 1
                if first_child(child, "rest") is None:
                    seen_non_rest += 1

            if tag == "attributes":
                div_txt = child_text(child, "divisions")
                if div_txt:
                    divisions = max(1, int(float(div_txt)))
                time_el = first_child(child, "time")
                if time_el is not None:
                    b = child_text(time_el, "beats")
                    bt = child_text(time_el, "beat-type")
                    if b and bt:
                        try:
                            beats = int(b)
                            beat_type = int(bt)
                            measure_has_time_signature = True
                        except ValueError:
                            warnings.append(f"Measure {number}: unusual time signature {b}/{bt}; keeping prior signature.")

            elif tag == "direction":
                offset_txt = child_text(child, "offset", "0") or "0"
                try:
                    offset_div = float(offset_txt)
                except ValueError:
                    offset_div = 0.0

                for desc in child.iter():
                    if local_name(desc.tag) == "sound" and "tempo" in desc.attrib:
                        try:
                            bpm = float(desc.attrib["tempo"])
                            qoff = max(0.0, (cursor_div + offset_div) / divisions)
                            tempo_events.append((qoff, bpm))
                            saw_tempo = True
                        except ValueError:
                            pass
                    if local_name(desc.tag) == "per-minute" and desc.text:
                        try:
                            bpm = float(desc.text.strip())
                            qoff = max(0.0, (cursor_div + offset_div) / divisions)
                            tempo_events.append((qoff, bpm))
                            saw_tempo = True
                        except ValueError:
                            pass

            elif tag == "sound" and "tempo" in child.attrib:
                try:
                    bpm = float(child.attrib["tempo"])
                    tempo_events.append((cursor_div / divisions, bpm))
                    saw_tempo = True
                except ValueError:
                    pass

            elif tag == "note":
                if first_child(child, "grace") is not None:
                    continue
                duration_txt = child_text(child, "duration", "0") or "0"
                try:
                    dur = int(float(duration_txt))
                except ValueError:
                    dur = 0
                is_chord = first_child(child, "chord") is not None
                if not is_chord:
                    # Track rest onsets too, not just real notes: a rest still
                    # gets its own visible glyph in the rendered tab, so its
                    # onset time is needed to correctly match MusicXML timing
                    # to on-page glyph runs (see detect_note_glyph_runs).
                    note_onsets.append(cursor_div / divisions)
                    cursor_div += dur
                    max_cursor_div = max(max_cursor_div, cursor_div)

            elif tag == "backup":
                duration_txt = child_text(child, "duration", "0") or "0"
                try:
                    cursor_div -= int(float(duration_txt))
                except ValueError:
                    pass
                cursor_div = max(0, cursor_div)

            elif tag == "forward":
                duration_txt = child_text(child, "duration", "0") or "0"
                try:
                    cursor_div += int(float(duration_txt))
                    max_cursor_div = max(max_cursor_div, cursor_div)
                except ValueError:
                    pass

            elif tag == "barline":
                for desc in child:
                    dtag = local_name(desc.tag)
                    if dtag == "repeat":
                        direction = desc.attrib.get("direction")
                        if direction == "forward":
                            repeat_forward = True
                        elif direction == "backward":
                            repeat_backward = True
                            try:
                                repeat_times = max(2, int(desc.attrib.get("times", "2")))
                            except ValueError:
                                repeat_times = 2
                    elif dtag == "ending":
                        has_ending = True

        nominal_quarters = beats * (4.0 / beat_type)
        duration_quarters = (max_cursor_div / divisions) if max_cursor_div > 0 else nominal_quarters

        # Ignore impossible tempo events and collapse duplicate offsets to the last tempo at that offset.
        event_map: dict[float, float] = {}
        for qoff, bpm in tempo_events:
            if 0.0 <= qoff <= duration_quarters and bpm > 0:
                event_map[qoff] = bpm
        events = sorted(event_map.items())

        def _onset_seconds(q: float) -> float:
            sec = 0.0
            prev = 0.0
            tempo_local = current_tempo
            for eq, bpm in events:
                if eq >= q:
                    break
                if eq > prev:
                    sec += (eq - prev) * 60.0 / tempo_local
                tempo_local = bpm
                prev = eq
            if q > prev:
                sec += (q - prev) * 60.0 / tempo_local
            return sec

        note_onset_seconds = [_onset_seconds(q) for q in sorted(set(note_onsets))]

        seconds = 0.0
        prev_q = 0.0
        tempo = current_tempo
        for qoff, bpm in events:
            if qoff > prev_q:
                seconds += (qoff - prev_q) * 60.0 / tempo
            tempo = bpm
            prev_q = qoff
        if duration_quarters > prev_q:
            seconds += (duration_quarters - prev_q) * 60.0 / tempo
        current_tempo = tempo

        result.append(
            MeasureTiming(
                index=mi,
                number=number,
                duration_quarters=duration_quarters,
                duration_seconds=seconds,
                repeat_forward=repeat_forward,
                repeat_backward=repeat_backward,
                repeat_times=repeat_times,
                has_ending=has_ending,
                is_empty=(seen_notes == 0 or seen_non_rest == 0),
                has_time_signature=measure_has_time_signature,
                note_onset_seconds=note_onset_seconds,
            )
        )

    if not saw_tempo:
        warnings.append("No tempo marking was found in MusicXML; defaulting to 125 BPM.")
    if any(m.has_ending for m in result):
        warnings.append(
            "Volta/ending brackets were found. This version does NOT expand first/second endings correctly. "
            "Use --no-expand-repeats for a straight-through test, or send me the file so repeat playback can be added."
        )

    # Words that often imply non-linear navigation.
    nav_words = []
    for el in part.iter():
        if local_name(el.tag) == "words" and el.text:
            t = el.text.upper()
            if any(k in t for k in ("D.S.", "D.C.", "CODA", "SEGNO", "TO CODA")):
                nav_words.append(el.text.strip())
    if nav_words:
        warnings.append("Navigation text detected (D.S./D.C./Coda/Segno); this version does not expand those jumps.")

    return result, warnings


def expand_basic_repeats(measures: list[MeasureTiming]) -> list[int]:
    """Expand ordinary forward/backward repeats. Does not support volta endings."""
    if any(m.has_ending for m in measures):
        raise RuntimeError(
            "Cannot safely expand repeats because volta/ending brackets exist. "
            "Run with --no-expand-repeats for now."
        )

    seq: list[int] = []
    i = 0
    repeat_start = 0
    backward_visits: dict[int, int] = {}
    safety = max(1000, len(measures) * 20)

    while 0 <= i < len(measures):
        if len(seq) > safety:
            raise RuntimeError("Repeat expansion exceeded safety limit; the repeat structure may be nested/unsupported.")

        m = measures[i]
        seq.append(i)

        if m.repeat_forward:
            repeat_start = i

        if m.repeat_backward:
            target_plays = max(2, m.repeat_times)
            plays_so_far = backward_visits.get(i, 1)
            if plays_so_far < target_plays:
                backward_visits[i] = plays_so_far + 1
                i = repeat_start
                continue

        i += 1

    return seq


# ----------------------------- SVG analysis -----------------------------

def render_svg(path: Path, scale: float) -> Image.Image:
    png = cairosvg.svg2png(url=str(path), scale=scale)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def analysis_gray(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:4] / 255.0
    on_white = rgb * alpha + 255.0 * (1.0 - alpha)
    return (0.2126 * on_white[..., 0] + 0.7152 * on_white[..., 1] + 0.0722 * on_white[..., 2]).astype(np.uint8)


def detect_line_centers(mask: np.ndarray, row_fraction: float) -> list[int]:
    h, w = mask.shape
    counts = mask.sum(axis=1)
    candidates = np.flatnonzero(counts >= max(20, int(w * row_fraction)))
    centers = []
    for a, b in grouped_runs(candidates):
        centers.append((a + b) // 2)
    return centers


def detect_tab_groups(
    mask: np.ndarray,
    strings: int,
    row_fraction: float,
    spacing_tolerance: float,
) -> list[list[int]]:
    """Detect tablature staves without rejecting them merely because another
    similarly spaced horizontal line sits just above/below the TAB staff.

    Guitar Pro commonly renders standard notation above TAB.  In some layouts
    the gap from the notation staff to the TAB staff is close enough to the TAB
    string spacing that the old detector treated the real 6-line TAB staff as
    part of a 7-line run and rejected it.

    We now collect every plausible ``strings``-line window, then resolve
    overlapping candidates.  When two candidates share line centers, prefer
    the lower one because Guitar Pro's TAB staff is below the standard staff.
    Regularity is used as a tie-breaker.
    """
    centers = detect_line_centers(mask, row_fraction)
    candidates: list[tuple[list[int], float]] = []

    def regular_window(window: list[int]) -> tuple[Optional[list[int]], float]:
        """Return a normalized ``strings``-line window and fit error.

        Primary mode expects exactly regular spacing. Fallback mode allows one
        weak/missing line center by accepting one near-double gap and inserting
        a synthetic center at the midpoint of that gap.
        """
        diffs = np.diff(window).astype(float)
        med = float(np.median(diffs)) if len(diffs) else 0.0
        if med < 3.0:
            return None, 999.0

        rel = np.abs(diffs - med) / med
        max_dev = float(np.max(rel)) if len(rel) else 999.0
        if max_dev <= spacing_tolerance:
            return window, max_dev

        # Salvage case: one line was too faint to detect, producing ~2x spacing
        # at one interval while the remaining intervals still look regular.
        if strings >= 4:
            double_idx = [
                i
                for i, d in enumerate(diffs)
                if abs(d - 2.0 * med) <= max(med * 0.42, 2.0)
            ]
            normal = [i for i, d in enumerate(diffs) if abs(d - med) <= max(med * spacing_tolerance, 2.0)]
            if len(double_idx) == 1 and len(normal) >= len(diffs) - 2:
                j = double_idx[0]
                recovered = list(window)
                recovered.insert(j + 1, int(round((window[j] + window[j + 1]) / 2.0)))
                if len(recovered) == strings:
                    rdiffs = np.diff(recovered).astype(float)
                    rmed = float(np.median(rdiffs)) if len(rdiffs) else 0.0
                    if rmed >= 3.0:
                        rdev = float(np.max(np.abs(rdiffs - rmed) / rmed))
                        if rdev <= max(spacing_tolerance * 1.15, 0.24):
                            return recovered, rdev

        return None, max_dev

    for i in range(0, len(centers) - strings + 1):
        window = centers[i : i + strings]
        normalized, error = regular_window(window)
        if normalized is not None:
            candidates.append((normalized, error))

    if not candidates:
        return []

    # Resolve windows that share one or more detected line centers.  A common
    # case is a 7-line-looking run caused by the bottom notation line plus six
    # TAB lines.  Both 6-line slices look mathematically valid; the lower slice
    # is the actual TAB staff.
    groups: list[list[int]] = []
    cluster: list[tuple[list[int], float]] = [candidates[0]]

    def overlaps(a: list[int], b: list[int]) -> bool:
        return bool(set(a) & set(b))

    def choose(cluster_items: list[tuple[list[int], float]]) -> list[int]:
        # Prefer the lowest candidate (largest bottom Y); if equal, choose the
        # most evenly-spaced one.
        return min(cluster_items, key=lambda item: (-item[0][-1], item[1]))[0]

    for cand in candidates[1:]:
        if any(overlaps(cand[0], existing[0]) for existing in cluster):
            cluster.append(cand)
        else:
            groups.append(choose(cluster))
            cluster = [cand]
    groups.append(choose(cluster))

    # Defensive de-duplication and page order.
    unique: list[list[int]] = []
    seen = set()
    for g in sorted(groups, key=lambda ys: ys[0]):
        key = tuple(g)
        if key not in seen:
            unique.append(g)
            seen.add(key)
    return unique


def detect_staff_extent(mask: np.ndarray, ys: list[int], spacing: float) -> tuple[int, int]:
    h, w = mask.shape
    radius = max(1, int(round(spacing * 0.12)))
    per_line = []
    for y in ys:
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        per_line.append(mask[y0:y1, :].any(axis=0))
    coverage = np.vstack(per_line).sum(axis=0)
    active = coverage >= max(2, len(ys) - 2)
    active = close_small_false_gaps(active, max_gap=max(8, int(spacing * 2.2)))
    left, right = longest_true_run(active)
    return left, right


def vertical_runs(mask: np.ndarray, ys: list[int], left: int, right: int, spacing: float) -> list[tuple[int, int]]:
    h, w = mask.shape
    y0 = max(0, int(ys[0] - spacing * 0.30))
    y1 = min(h, int(ys[-1] + spacing * 0.30) + 1)
    roi = mask[y0:y1, left : right + 1]
    counts = roi.sum(axis=0)
    threshold = max(4, int((y1 - y0) * 0.58))
    xs = np.flatnonzero(counts >= threshold)
    return [(a + left, b + left) for a, b in grouped_runs(xs)]


def boundary_clusters(
    runs: list[tuple[int, int]], staff_left: int, staff_right: int, spacing: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    width = staff_right - staff_left
    near = max(spacing * 7.0, width * 0.10)
    # Keep this tight: genuine double-barline / repeat-dot strokes sit within
    # about one spacing-unit of the anchor barline. A wider gap risks
    # absorbing an unrelated nearby glyph (e.g. a brush/strum arrow or a
    # chord's own fret numbers) into the "furniture" cluster, which then
    # causes the first note's play_left to be skipped past its true start.
    cluster_gap = spacing * 1.0

    # Left: anchor on the earliest strong vertical stroke near the staff start,
    # then absorb neighboring strokes belonging to double/repeat barlines.
    left_candidates = [r for r in runs if r[0] <= staff_left + near]
    if left_candidates:
        anchor = min(left_candidates, key=lambda r: abs(r[0] - staff_left))
        cluster = [r for r in left_candidates if r[0] <= anchor[1] + cluster_gap]
        lcluster = (min(r[0] for r in cluster), max(r[1] for r in cluster))
    else:
        lcluster = (staff_left, staff_left)

    right_candidates = [r for r in runs if r[1] >= staff_right - near]
    if right_candidates:
        anchor = min(right_candidates, key=lambda r: abs(staff_right - r[1]))
        cluster = [r for r in right_candidates if r[1] >= anchor[0] - cluster_gap]
        rcluster = (min(r[0] for r in cluster), max(r[1] for r in cluster))
    else:
        rcluster = (staff_right, staff_right)

    return lcluster, rcluster


def detect_note_glyph_runs(
    mask: np.ndarray,
    ys: list[int],
    spacing: float,
    search_left: int,
    search_right: int,
    expect_time_signature: bool = False,
    expect_repeat_dots: bool = False,
) -> list[tuple[int, int]]:
    """
    Find every glyph run (fret number, chord, articulation mark, etc.) within
    [search_left, search_right), left to right. Each run is one "column" of
    simultaneous notes -- a chord's connecting stem keeps its notes in a
    single run -- so runs correspond 1:1 with distinct note-onset times in the
    measure. This is what lets the playhead move at the true, uneven
    note-to-note pace Guitar Pro renders (some notes sit visually closer
    together than others) instead of a constant pace across the whole
    measure, and (via the first run) lets it start exactly on the first note
    instead of at a fixed offset from the barline.

    A plain tab string line only puts ink in a thin band right at its own row.
    A fret number/glyph sitting on that string is taller than the line, so it
    also puts ink just above and/or below the line's own thin band. Looking
    for that "extra" ink isolates real note content from the bare staff line.

    A time signature (or clef) drawn at the start of a system is also taller
    than the plain line, so it would otherwise be mistaken for a note. It
    CANNOT be told apart from a real note by height/shape alone: this
    renderer draws a connecting stem between simultaneous notes in a chord,
    and that stem is just as tall as a time signature's numerals. Instead,
    when the caller (using MusicXML data) tells us this measure declares a
    time signature via `expect_time_signature`, we treat the very first ink
    run in the search window as the time signature glyph -- but only if it
    genuinely looks tall/furniture-like -- and skip past it before collecting
    note runs. This is safe because a courtesy time signature is always the
    first thing drawn after the barline, so only the first run ever needs
    this treatment.

    A repeat-start barline draws two small dots right next to the double bar
    line, sitting in the gaps between string lines. Unlike a time signature,
    dots are SHORT, not tall, so the height check above would never flag
    them -- and they sit close enough to the barline/first note that they
    can end up fused with leftover barline ink into what looks like a
    plausible "first run". When the caller tells us this measure starts with
    a repeat (`expect_repeat_dots`), we check whether that first ink run is
    dot-sized (tiny even over a much taller window than a note would ever
    need) and only then skip past it, since a repeat's dots are always the
    first thing drawn after the barline (before any courtesy time signature
    and well before the first real note). If that first run isn't dot-sized
    -- e.g. the barline cluster detection already absorbed a strum-arrow's
    full-height stroke, leaving little to no gap before the real first
    note/chord -- it's left alone so a real note is never skipped.
    """
    if search_right <= search_left:
        return []
    h, w = mask.shape
    line_radius = max(1, int(round(spacing * 0.16)))
    glyph_radius = max(line_radius + 2, int(round(spacing * 0.42)))
    ext_top = max(0, ys[0] - int(round(spacing * 0.55)))
    ext_bot = min(h, ys[-1] + int(round(spacing * 0.55)) + 1)

    def tallest_blob(x0: int, x1: int) -> int:
        tallest = 0
        for x in range(x0, x1 + 1):
            col = mask[ext_top:ext_bot, x]
            ink_idx = np.flatnonzero(col)
            if len(ink_idx) == 0:
                continue
            blobs = grouped_runs(ink_idx)
            tallest = max(tallest, max(b - a + 1 for a, b in blobs))
        return tallest

    # A much wider vertical window than ext_top/ext_bot -- large enough to
    # capture a strum-direction arrow's true height (these are drawn well
    # above/below the staff itself), used only to tell a real note apart
    # from genuine (always tiny) repeat dots below.
    wide_top = max(0, ys[0] - int(round(spacing * 2.0)))
    wide_bot = min(h, ys[-1] + int(round(spacing * 2.0)) + 1)

    def wide_tallest_blob(x0: int, x1: int) -> int:
        tallest = 0
        for x in range(x0, x1 + 1):
            col = mask[wide_top:wide_bot, x]
            ink_idx = np.flatnonzero(col)
            if len(ink_idx) == 0:
                continue
            blobs = grouped_runs(ink_idx)
            tallest = max(tallest, max(b - a + 1 for a, b in blobs))
        return tallest

    effective_left = search_left
    if expect_time_signature or expect_repeat_dots:
        # Ink outside the thin per-string line bands, using the full staff
        # height rather than the narrow glyph_radius band, gives true blank
        # gaps between distinct glyphs (a plain string line alone never
        # shows up here).
        excl = np.zeros(ext_bot - ext_top, dtype=bool)
        for y in ys:
            a = max(0, y - line_radius - ext_top)
            b = min(ext_bot - ext_top, y + line_radius + 1 - ext_top)
            excl[a:b] = True
        sub = mask[ext_top:ext_bot, :].copy()
        sub[excl, :] = False
        extra_ink = sub.any(axis=0)

        # Repeat dots are drawn before any courtesy time signature, so check
        # for them first. Repeat dots are always tiny (a fraction of the
        # staff spacing tall), even measured over the wide window above --
        # but a real note can end up as this same "first run" too (e.g. a
        # wide chord/strum-arrow combo with little to no gap after the
        # barline), and those are unmistakably much taller. Only skip when
        # the run genuinely looks dot-sized, so a real first note is never
        # mistaken for furniture.
        if expect_repeat_dots:
            idx = np.flatnonzero(extra_ink[effective_left:search_right])
            if len(idx):
                run_start, run_end = grouped_runs((effective_left + i) for i in idx)[0]
                if wide_tallest_blob(run_start, run_end) <= spacing * 0.8:
                    effective_left = run_end + 1

        if expect_time_signature:
            idx = np.flatnonzero(extra_ink[effective_left:search_right])
            if len(idx):
                run_start, run_end = grouped_runs((effective_left + i) for i in idx)[0]
                run_width = run_end - run_start + 1
                # A genuine time-signature glyph is a narrow stack of digits
                # (roughly a couple of numerals wide at most), tall only
                # because the numerator/denominator span most of the staff
                # height. A chord's own connecting stem is just as tall but,
                # thanks to Guitar Pro staggering each string's fret digit
                # horizontally, considerably wider. Requiring the run to
                # also be narrow keeps a genuinely first-note chord from
                # being mistaken for the time signature and skipped.
                if tallest_blob(run_start, run_end) > spacing * 1.3 and run_width <= spacing * 2.2:
                    effective_left = run_end + 1

    note_cols = np.zeros(w, dtype=bool)
    for y in ys:
        top = max(0, y - glyph_radius)
        bot = min(h, y + glyph_radius + 1)
        line_top = max(top, y - line_radius)
        line_bot = min(bot, y + line_radius + 1)
        if line_top > top:
            note_cols |= mask[top:line_top, :].any(axis=0)
        if bot > line_bot:
            note_cols |= mask[line_bot:bot, :].any(axis=0)

    # Guitar Pro horizontally staggers the individual digits of a
    # simultaneous chord by a few pixels so overlapping fret-number glyphs
    # stay legible, which otherwise fragments one onset's ink into several
    # tiny runs separated by gaps of only a handful of pixels. Genuine gaps
    # between two different onsets are far larger (roughly a full glyph
    # width or more), so closing only small gaps merges the former without
    # bridging the latter.
    note_cols = close_small_false_gaps(note_cols, max_gap=max(3, int(round(spacing * 0.55))))

    # The left barline-cluster boundary (used to build search_left) can
    # accidentally absorb part of the very first note/chord's own ink -- a
    # chord spanning several strings draws a solid connecting stem that, near
    # the barline, can look just as tall/dense as barline ink to the boundary
    # detector. When that happens, search_left lands in the middle of the
    # real glyph instead of before it, so only its rightmost sliver (often
    # just 1-2px) is ever seen, misplacing play_left and confusing the
    # run-count/onset pairing downstream. Detect this by checking whether ink
    # is already active right at the boundary (a genuine first note always
    # has a clear blank gap before it -- that gap is what makes search_left a
    # safe starting point in the first place) and, if so, walk left through
    # the contiguous ink to recover the glyph's true start. Bounded so this
    # can never walk back into the barline itself, since a real blank gap
    # always separates the two.
    if 0 < effective_left < w and note_cols[effective_left]:
        recover_floor = max(0, effective_left - max(4, int(round(spacing * 2.0))))
        x = effective_left
        while x > recover_floor and note_cols[x - 1]:
            x -= 1
        effective_left = x

    if effective_left >= search_right:
        return []
    idx = np.flatnonzero(note_cols[effective_left:search_right])
    if len(idx) == 0:
        return []
    runs = [(effective_left + a, effective_left + b) for a, b in grouped_runs(idx)]

    # The right barline's own antialiased edge can bleed a couple of columns
    # past the strict barline-cluster boundary used as search_right, showing
    # up here as a spurious extra "run" that just touches the edge of the
    # search window. A real last note almost always sits with a visible gap
    # before the barline, so a run ending right at the boundary is barline
    # bleed-through, not a note, and is dropped.
    edge_guard = max(2, int(round(spacing * 0.12)))
    if runs and (search_right - runs[-1][1]) <= edge_guard:
        runs = runs[:-1]

    return runs


def detect_first_note_x(
    mask: np.ndarray,
    ys: list[int],
    spacing: float,
    search_left: int,
    search_right: int,
    expect_time_signature: bool = False,
    expect_repeat_dots: bool = False,
) -> Optional[int]:
    """Return the x-coordinate of the first note glyph, or None if no note
    content is found. See detect_note_glyph_runs for the detection rationale."""
    runs = detect_note_glyph_runs(
        mask, ys, spacing, search_left, search_right, expect_time_signature, expect_repeat_dots
    )
    return runs[0][0] if runs else None


def detect_page_geometries(
    img: Image.Image,
    page_index: int,
    starting_measure_index: int,
    strings: int,
    threshold: int,
    row_fraction: float,
    spacing_tolerance: float,
    inner_padding_spaces: float,
    crop_above_spaces: float,
    crop_below_spaces: float,
) -> list[StaffGeometry]:
    gray = analysis_gray(img)
    mask = gray < threshold
    groups = detect_tab_groups(mask, strings, row_fraction, spacing_tolerance)
    result: list[StaffGeometry] = []
    h, w = mask.shape

    for si, ys in enumerate(groups):
        spacing = float(np.median(np.diff(ys)))
        staff_left, staff_right = detect_staff_extent(mask, ys, spacing)
        runs = vertical_runs(mask, ys, staff_left, staff_right, spacing)
        lcluster, rcluster = boundary_clusters(runs, staff_left, staff_right, spacing)

        pad = spacing * inner_padding_spaces
        note_search_left = lcluster[1] + max(2, int(round(spacing * 0.05)))
        first_note_x = detect_first_note_x(mask, ys, spacing, note_search_left, rcluster[0])
        if first_note_x is not None:
            play_left = float(first_note_x)
        else:
            play_left = float(lcluster[1] + pad)
        play_right = float(rcluster[0] - pad)
        if play_right <= play_left + spacing * 4:
            # Defensive fallback if vertical-line detection grabbed something nonsensical.
            play_left = float(staff_left + pad)
            play_right = float(staff_right - pad)

        x0 = max(0, int(staff_left - spacing * 2.0))
        x1 = min(w, int(staff_right + spacing * 2.0) + 1)
        y0 = max(0, int(ys[0] - spacing * crop_above_spaces))
        y1 = min(h, int(ys[-1] + spacing * crop_below_spaces) + 1)

        # Never let a system crop swallow the neighboring measure. This matters
        # when there is lots of requested space above the TAB for standard
        # notation, dynamics, etc.
        if si > 0:
            prev_ys = groups[si - 1]
            boundary_above = int((prev_ys[-1] + ys[0]) / 2)
            y0 = max(y0, boundary_above)
        if si + 1 < len(groups):
            next_ys = groups[si + 1]
            boundary_below = int((ys[-1] + next_ys[0]) / 2)
            y1 = min(y1, boundary_below)

        result.append(
            StaffGeometry(
                measure_index=starting_measure_index + len(result),
                page_index=page_index,
                system_index_on_page=si,
                string_ys=list(ys),
                spacing=spacing,
                staff_left=staff_left,
                staff_right=staff_right,
                left_cluster_left=lcluster[0],
                left_cluster_right=lcluster[1],
                right_cluster_left=rcluster[0],
                right_cluster_right=rcluster[1],
                play_left=play_left,
                play_right=play_right,
                crop_box=(x0, y0, x1, y1),
            )
        )

    return result


def find_internal_split(mask: np.ndarray, geom: StaffGeometry) -> Optional[tuple[int, int, int]]:
    """
    Look for a single strong internal barline inside an already-detected staff,
    strictly between its left/right barline clusters.

    This is only used to recover cases where Guitar Pro placed two (usually
    empty/silent) measures on a single rendered system, which violates this
    tool's one-measure-per-system assumption. Returns (center_x, run_left,
    run_right) for the strongest qualifying internal vertical stroke, or None.
    """
    ys = geom.string_ys
    y0, y1 = ys[0], ys[-1]
    height = y1 - y0 + 1
    if height <= 0:
        return None

    x_start = geom.left_cluster_right + 2
    x_end = geom.right_cluster_left - 2
    if x_end <= x_start:
        return None

    roi = mask[y0 : y1 + 1, x_start : x_end + 1]
    counts = roi.sum(axis=0)
    threshold = height * 0.90
    xs = np.flatnonzero(counts >= threshold)
    if len(xs) == 0:
        return None

    runs = grouped_runs(xs)
    best = max(runs, key=lambda r: counts[r[0] : r[1] + 1].mean())
    run_left = int(best[0] + x_start)
    run_right = int(best[1] + x_start)
    return (run_left + run_right) // 2, run_left, run_right


def reconcile_measure_counts(
    geoms: list[StaffGeometry],
    measures: list[MeasureTiming],
    pages: list[Image.Image],
    ink_threshold: int,
    inner_padding_spaces: float,
) -> list[StaffGeometry]:
    """
    Recover the common case where Guitar Pro rendered two consecutive, entirely
    empty/silent measures on a single system. Detection elsewhere assumes one
    measure per system, so such a system is found once but must count as two
    measures. Only measures MusicXML confirms are empty are ever split, so this
    cannot misfire on systems that contain real note content.
    """
    if len(geoms) >= len(measures):
        return geoms

    masks: dict[int, np.ndarray] = {}

    def mask_for_page(page_index: int) -> np.ndarray:
        if page_index not in masks:
            gray = analysis_gray(pages[page_index])
            masks[page_index] = gray < ink_threshold
        return masks[page_index]

    result: list[StaffGeometry] = []
    mi = 0
    gi = 0
    while mi < len(measures) and gi < len(geoms):
        remaining_measures = len(measures) - mi
        remaining_geoms = len(geoms) - gi
        can_split = (
            remaining_measures > remaining_geoms
            and mi + 1 < len(measures)
            and measures[mi].is_empty
            and measures[mi + 1].is_empty
        )
        if can_split:
            g = geoms[gi]
            mask = mask_for_page(g.page_index)
            split = find_internal_split(mask, g)
            if split is not None:
                center, run_left, run_right = split
                pad = g.spacing * inner_padding_spaces
                left_half = StaffGeometry(
                    measure_index=0,
                    page_index=g.page_index,
                    system_index_on_page=g.system_index_on_page,
                    string_ys=g.string_ys,
                    spacing=g.spacing,
                    staff_left=g.staff_left,
                    staff_right=g.staff_right,
                    left_cluster_left=g.left_cluster_left,
                    left_cluster_right=g.left_cluster_right,
                    right_cluster_left=run_left,
                    right_cluster_right=run_right,
                    play_left=g.play_left,
                    play_right=float(run_left - pad),
                    crop_box=g.crop_box,
                )
                right_half = StaffGeometry(
                    measure_index=0,
                    page_index=g.page_index,
                    system_index_on_page=g.system_index_on_page,
                    string_ys=g.string_ys,
                    spacing=g.spacing,
                    staff_left=g.staff_left,
                    staff_right=g.staff_right,
                    left_cluster_left=run_left,
                    left_cluster_right=run_right,
                    right_cluster_left=g.right_cluster_left,
                    right_cluster_right=g.right_cluster_right,
                    play_left=float(run_right + pad),
                    play_right=g.play_right,
                    crop_box=g.crop_box,
                )
                result.append(left_half)
                result.append(right_half)
                mi += 2
                gi += 1
                continue

        result.append(geoms[gi])
        mi += 1
        gi += 1

    for idx, g in enumerate(result):
        g.measure_index = idx

    return result


def refine_furniture_starts(
    geoms: list[StaffGeometry],
    measures: list[MeasureTiming],
    pages: list[Image.Image],
    ink_threshold: int,
) -> None:
    """
    detect_first_note_x's time-signature/repeat-dot skip needs to know, per
    measure, whether MusicXML declares a time signature or a repeat start
    there. That mapping is only reliable once geoms and measures are
    guaranteed to align 1:1 (same length, same order), which is only true
    after reconcile_measure_counts has run -- empty-measure splitting shifts
    indices before that point. So the initial detect_page_geometries pass
    always runs without that awareness, and this second pass re-detects
    play_left just for the (typically rare) measures that actually declare a
    time signature and/or start with a repeat, mutating geoms in place.
    """
    masks: dict[int, np.ndarray] = {}

    def mask_for_page(page_index: int) -> np.ndarray:
        if page_index not in masks:
            gray = analysis_gray(pages[page_index])
            masks[page_index] = gray < ink_threshold
        return masks[page_index]

    for g, m in zip(geoms, measures):
        if not m.has_time_signature and not m.repeat_forward:
            continue
        mask = mask_for_page(g.page_index)
        note_search_left = g.left_cluster_right + max(2, int(round(g.spacing * 0.05)))
        first_note_x = detect_first_note_x(
            mask,
            g.string_ys,
            g.spacing,
            note_search_left,
            g.right_cluster_left,
            m.has_time_signature,
            m.repeat_forward,
        )
        if first_note_x is not None and first_note_x > g.play_left:
            g.play_left = float(first_note_x)


def compute_note_breakpoints(
    geoms: list[StaffGeometry],
    measures: list[MeasureTiming],
    pages: list[Image.Image],
    ink_threshold: int,
) -> tuple[int, int]:
    """
    Build a per-measure time -> x-position curve so the playhead moves at the
    real, uneven note-to-note pace Guitar Pro renders instead of a constant
    pace across the whole measure. For each measure, MusicXML's note onset
    times (already tempo-adjusted, in seconds from the measure start) are
    matched 1:1, in order, to the detected on-page glyph runs (each run is one
    note or simultaneous chord -- see detect_note_glyph_runs). When the counts
    don't match (ties, ornaments, or a merged run defeat a clean pairing),
    that measure falls back to a straight two-point line from play_left to
    play_right, i.e. the previous constant-pace behavior.

    Must run after reconcile_measure_counts and only once len(geoms) ==
    len(measures), for the same indexing reason as refine_furniture_starts:
    empty-measure splitting shifts indices before that point.

    Returns (matched, eligible) measure counts for a one-line summary.
    """
    masks: dict[int, np.ndarray] = {}

    def mask_for_page(page_index: int) -> np.ndarray:
        if page_index not in masks:
            gray = analysis_gray(pages[page_index])
            masks[page_index] = gray < ink_threshold
        return masks[page_index]

    eps = 1e-6
    matched = 0
    eligible = 0

    for g, m in zip(geoms, measures):
        onsets = m.note_onset_seconds
        breakpoints: list[tuple[float, float]] = []

        if onsets:
            eligible += 1
            mask = mask_for_page(g.page_index)
            note_search_left = g.left_cluster_right + max(2, int(round(g.spacing * 0.05)))
            runs = detect_note_glyph_runs(
                mask,
                g.string_ys,
                g.spacing,
                note_search_left,
                g.right_cluster_left,
                m.has_time_signature,
                m.repeat_forward,
            )
            if len(runs) == len(onsets) + 1:
                # A long/tied note is sometimes followed by a small
                # hammer-on/pull-off curl or slur-tail mark that sits just
                # clear of its note glyph (too far away for the small-gap
                # merge in detect_note_glyph_runs to bridge), showing up as
                # its own spurious extra run. A real fret-number glyph is
                # never anywhere near as narrow as one of these marks, so if
                # dropping the single narrowest run recovers an exact count
                # match, treat that as the fix -- but only when it's a clear
                # outlier (much narrower than the rest), to avoid discarding
                # a genuine (if unusually narrow) note elsewhere.
                widths = [b - a + 1 for a, b in runs]
                narrowest_i = min(range(len(runs)), key=lambda i: widths[i])
                rest = widths[:narrowest_i] + widths[narrowest_i + 1 :]
                if rest and widths[narrowest_i] * 2 <= min(rest):
                    runs = runs[:narrowest_i] + runs[narrowest_i + 1 :]
                elif len(runs) > 2:
                    # No narrow outlier -- this is often instead a single
                    # onset's glyph rendered in two adjacent pieces with a
                    # gap just over detect_note_glyph_runs' own small-gap
                    # merge threshold (e.g. an artificial-harmonic fret
                    # number followed a few pixels later by its "<n>" pitch
                    # annotation). The gap between that pair is always a
                    # clear outlier vs. the real gaps between distinct
                    # onsets elsewhere in the same measure, so merge the two
                    # runs straddling the single smallest gap when it is
                    # unambiguously smaller than every other gap.
                    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
                    smallest_i = min(range(len(gaps)), key=lambda i: gaps[i])
                    other_gaps = gaps[:smallest_i] + gaps[smallest_i + 1 :]
                    if other_gaps and gaps[smallest_i] * 2 <= min(other_gaps):
                        merged = (runs[smallest_i][0], runs[smallest_i + 1][1])
                        runs = runs[:smallest_i] + [merged] + runs[smallest_i + 2 :]
            if len(runs) == len(onsets):
                matched += 1
                breakpoints = [(t, float(r[0])) for t, r in zip(onsets, runs)]

        if not breakpoints:
            breakpoints = [(0.0, g.play_left), (m.duration_seconds, g.play_right)]
        else:
            if breakpoints[0][0] > eps:
                breakpoints.insert(0, (0.0, g.play_left))
            if breakpoints[-1][0] < m.duration_seconds - eps:
                breakpoints.append((m.duration_seconds, g.play_right))

        g.note_breakpoints = breakpoints

    return matched, eligible


def apply_geometry_overrides(geoms: list[StaffGeometry], path: Optional[Path]) -> None:
    """
    Optional JSON overrides. Keys are 1-based sequential measure numbers.

    Example:
      {
        "12": {"left_offset": 8, "right_offset": -4},
        "33": {"play_left": 245.0, "play_right": 1820.0}
      }

    Coordinates refer to the rasterized SVG analysis image, i.e. after --svg-scale.
    """
    if not path:
        return
    data = json.loads(path.read_text())
    for key, vals in data.items():
        idx = int(key) - 1
        if idx < 0 or idx >= len(geoms):
            raise RuntimeError(f"Override measure {key} is out of range.")
        g = geoms[idx]
        if "play_left" in vals:
            g.play_left = float(vals["play_left"])
        if "play_right" in vals:
            g.play_right = float(vals["play_right"])
        g.play_left += float(vals.get("left_offset", 0))
        g.play_right += float(vals.get("right_offset", 0))


def write_debug_pages(
    pages: list[Image.Image],
    geoms: list[StaffGeometry],
    debug_dir: Path,
    measure_numbers: list[str],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    by_page: dict[int, list[StaffGeometry]] = {}
    for g in geoms:
        by_page.setdefault(g.page_index, []).append(g)

    for pi, page in enumerate(pages):
        img = page.convert("RGBA").copy()
        draw = ImageDraw.Draw(img, "RGBA")
        for g in by_page.get(pi, []):
            x0, y0, x1, y1 = g.crop_box
            ytop = g.string_ys[0]
            ybot = g.string_ys[-1]
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 165, 0, 220), width=3)
            draw.rectangle(
                (g.left_cluster_left, ytop, g.left_cluster_right, ybot),
                fill=(30, 144, 255, 100),
            )
            draw.rectangle(
                (g.right_cluster_left, ytop, g.right_cluster_right, ybot),
                fill=(30, 144, 255, 100),
            )
            for _, bx in g.note_breakpoints[1:-1]:
                draw.line((bx, ytop, bx, ybot), fill=(160, 32, 240, 160), width=2)
            draw.line((g.play_left, y0, g.play_left, y1), fill=(0, 190, 80, 255), width=4)
            draw.line((g.play_right, y0, g.play_right, y1), fill=(220, 50, 50, 255), width=4)
            label_num = measure_numbers[g.measure_index] if g.measure_index < len(measure_numbers) else str(g.measure_index + 1)
            label = f"measure {label_num}  L={g.play_left:.1f}  R={g.play_right:.1f}"
            label_x = max(x0, int(g.play_left) - 4)
            draw.rectangle((label_x, max(0, y0 - 24), label_x + min(620, len(label) * 9), y0), fill=(255, 255, 255, 220))
            draw.text((label_x + 4, max(0, y0 - 21)), label, fill=(0, 0, 0, 255))
        out = debug_dir / f"page_{pi + 1:03d}_detected.png"
        img.save(out)

    # Also dump coordinates for easy inspection / future overrides.
    payload = []
    for g in geoms:
        payload.append(
            {
                "sequential_measure": g.measure_index + 1,
                "musicxml_measure": measure_numbers[g.measure_index] if g.measure_index < len(measure_numbers) else None,
                "page": g.page_index + 1,
                "system_on_page": g.system_index_on_page + 1,
                "staff_left": int(g.staff_left),
                "staff_right": int(g.staff_right),
                "left_bar_cluster": [int(g.left_cluster_left), int(g.left_cluster_right)],
                "right_bar_cluster": [int(g.right_cluster_left), int(g.right_cluster_right)],
                "play_left": g.play_left,
                "play_right": g.play_right,
                "crop_box": [int(v) for v in g.crop_box],
                "spacing": float(g.spacing),
            }
        )
    (debug_dir / "geometry.json").write_text(json.dumps(payload, indent=2))


# ----------------------------- rendering -----------------------------

def compute_global_scale(
    geoms: list[StaffGeometry],
    out_width: int,
    out_height: int,
    margin: int,
    crop_above_spaces: float,
    crop_below_spaces: float,
) -> float:
    """
    Compute a single scale factor shared by every measure.

    Per-measure crop height can be slightly smaller than the nominal
    crop-above/crop-below request when a system sits close to a neighboring
    system on the page (see the boundary clamping in detect_page_geometries).
    Using each measure's own (possibly clamped) crop height to size the video
    frame made the whole tab image change size/position measure to measure.
    Instead we size against the nominal, unclamped height so the resulting
    scale is identical for every measure; only real per-measure crop WIDTH
    differences (i.e. actual note content) are allowed to vary.
    """
    max_w = max(1, out_width - 2 * margin)
    max_h = max(1, out_height - 2 * margin)
    scales = []
    for g in geoms:
        cw = g.crop_box[2] - g.crop_box[0]
        nominal_ch = g.spacing * (crop_above_spaces + (len(g.string_ys) - 1) + crop_below_spaces)
        scales.append(min(max_w / cw, max_h / max(1.0, nominal_ch)))
    return min(scales)


def prepare_measure_canvas(
    page: Image.Image,
    geom: StaffGeometry,
    out_width: int,
    out_height: int,
    margin: int,
    background: str,
    scale: float,
    crop_above_spaces: float,
) -> RenderedMeasure:
    x0, y0, x1, y1 = geom.crop_box
    crop = page.crop(geom.crop_box).convert("RGBA")
    cw, ch = crop.size
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    crop = crop.resize((nw, nh), Image.Resampling.LANCZOS)

    if background.lower() == "transparent":
        canvas = Image.new("RGBA", (out_width, out_height), (0, 0, 0, 0))
    else:
        canvas = Image.new("RGBA", (out_width, out_height), parse_color(background))

    # Anchor every frame at the same fixed point rather than centering each
    # crop's own (possibly clamped) bounding box: pin the left edge at the
    # margin, and pin the nominal (unclamped) top of the crop at the margin
    # too. If this particular system's crop was clamped tighter than nominal
    # (close neighboring system), the missing space is added back here as a
    # blank buffer so the staff lines still land on the same pixel row.
    virtual_y0 = geom.string_ys[0] - geom.spacing * crop_above_spaces
    ox = margin
    oy = int(round(margin - (virtual_y0 - y0) * scale))
    canvas.alpha_composite(crop, (ox, oy))

    play_left = ox + (geom.play_left - x0) * scale
    play_right = ox + (geom.play_right - x0) * scale
    note_breakpoints = [(t, ox + (x - x0) * scale) for t, x in geom.note_breakpoints] or [
        (0.0, play_left),
        (1.0, play_right),
    ]

    # Limit the playhead's vertical extent to the staff itself: the line runs
    # from just above the top string line down to the bottom string line,
    # matching a clean NLE-style playhead (e.g. Final Cut Pro) rather than
    # overshooting past the staff on both ends.
    spacing_px = geom.spacing * scale
    top_y = oy + (geom.string_ys[0] - y0) * scale
    bottom_y = oy + (geom.string_ys[-1] - y0) * scale
    play_top = top_y - spacing_px * 0.18
    play_bottom = bottom_y

    return RenderedMeasure(np.array(canvas), play_left, play_right, play_top, play_bottom, spacing_px, note_breakpoints)


def build_timeline(measures: list[MeasureTiming], sequence: list[int], lead_in: float, tail: float):
    items = []
    t = lead_in
    for idx in sequence:
        dur = measures[idx].duration_seconds
        items.append((idx, t, t + dur))
        t += dur
    return items, t + tail


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def interp_breakpoints(breakpoints: list[tuple[float, float]], elapsed: float) -> float:
    """Piecewise-linear lookup of x-position at time `elapsed` (seconds from
    measure start) given a sorted list of (time, x) breakpoints."""
    if len(breakpoints) == 1:
        return breakpoints[0][1]
    times = [bp[0] for bp in breakpoints]
    idx = bisect.bisect_right(times, elapsed) - 1
    idx = max(0, min(idx, len(breakpoints) - 2))
    t0, x0 = breakpoints[idx]
    t1, x1 = breakpoints[idx + 1]
    if t1 <= t0:
        return x0
    frac = max(0.0, min(1.0, (elapsed - t0) / (t1 - t0)))
    return x0 + frac * (x1 - x0)


def draw_playhead_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    tip_y: float,
    spacing_px: float,
    fill: tuple[int, int, int, int],
) -> None:
    """Draw a small flag-shaped marker whose point sits at (x, tip_y), the top
    of the playhead line -- similar to a Final Cut Pro-style playhead handle
    sitting just above the timeline content."""
    half_w = spacing_px * 0.55
    corner_cut = half_w * 0.35
    rect_h = spacing_px * 0.55
    tip_h = spacing_px * 0.35
    rect_top = tip_y - tip_h - rect_h
    rect_bottom = tip_y - tip_h

    points = [
        (x - half_w + corner_cut, rect_top),
        (x + half_w - corner_cut, rect_top),
        (x + half_w, rect_top + corner_cut),
        (x + half_w, rect_bottom),
        (x, tip_y),
        (x - half_w, rect_bottom),
        (x - half_w, rect_top + corner_cut),
    ]
    draw.polygon(points, fill=fill)


def render_video(
    output: Path,
    prepared: list[RenderedMeasure],
    timeline: list[tuple[int, float, float]],
    total_duration: float,
    fps: float,
    width: int,
    height: int,
    playhead_color: str,
    playhead_width: int,
    playhead_opacity: int,
    background: str,
    audio: Optional[Path],
):
    transparent = background.lower() == "transparent"
    if transparent and output.suffix.lower() != ".mov":
        raise RuntimeError("Transparent output should use a .mov filename (ProRes 4444).")

    ffmpeg = get_ffmpeg_exe()
    input_pix = "rgba"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", input_pix,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio:
        cmd += ["-i", str(audio)]

    if transparent:
        cmd += ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p"]

    if audio:
        cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "320k", "-shortest"]
    cmd += [str(output)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    starts = [item[1] for item in timeline]
    nframes = int(math.ceil(total_duration * fps))
    ph_rgba = parse_color(playhead_color, max(0, min(255, playhead_opacity)))

    first_idx = timeline[0][0]
    last_idx = timeline[-1][0]

    try:
        for frame_no in range(nframes):
            t = frame_no / fps
            pos = bisect.bisect_right(starts, t) - 1

            if pos < 0:
                measure_idx = first_idx
                elapsed = 0.0
            elif pos >= len(timeline):
                measure_idx = last_idx
                elapsed = timeline[-1][2] - timeline[-1][1]
            else:
                measure_idx, start, end = timeline[pos]
                elapsed = max(0.0, min(end - start, t - start))

            base = prepared[measure_idx]
            frame_img = Image.fromarray(base.rgba.copy(), mode="RGBA")
            draw = ImageDraw.Draw(frame_img, "RGBA")
            x = interp_breakpoints(base.note_breakpoints, elapsed)
            draw.line((x, base.play_top, x, base.play_bottom), fill=ph_rgba, width=playhead_width)
            draw_playhead_marker(draw, x, base.play_top, base.spacing_px, ph_rgba)
            proc.stdin.write(frame_img.tobytes())

            if frame_no % max(1, int(fps * 5)) == 0:
                pct = 100.0 * frame_no / max(1, nframes)
                print(f"Rendering: {pct:5.1f}%", end="\r", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
        print(" " * 40, end="\r")
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc}")


# ----------------------------- CLI -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a moving playhead over Guitar Pro SVG tab, one measure per line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "musicxml",
        nargs="?",
        type=Path,
        default=None,
        help="Guitar Pro MusicXML export (default: auto-detected single file in --xml-dir)",
    )
    p.add_argument(
        "svg",
        nargs="*",
        type=Path,
        default=None,
        help="SVG page(s), in page order; shell globs are fine (default: all *.svg files in --svg-dir)",
    )
    p.add_argument("-o", "--output", type=Path, default=Path("tab.mov"))
    p.add_argument("--xml-dir", type=Path, default=Path("xml"), help="Folder to auto-discover the MusicXML file from when it is not given positionally")
    p.add_argument("--svg-dir", type=Path, default=Path("svg"), help="Folder to auto-discover SVG pages from when none are given positionally")
    p.add_argument("--part", help="MusicXML part id to use for timing (default: first part)")
    p.add_argument("--strings", type=int, default=6, help="Number of tablature string lines")
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--margin", type=int, default=36)
    p.add_argument("--background", default="transparent", help="CSS color like white/#111111, or transparent")
    p.add_argument("--transparent", action="store_true", help="Shortcut for --background transparent")
    p.add_argument(
        "--no-invert-colors",
        dest="invert_colors",
        action="store_false",
        help="Disable ink recoloring / transparency inversion; render dark ink on the page background as-is",
    )
    p.set_defaults(invert_colors=True)
    p.add_argument("--foreground-color", default="white", help="Ink color used when inversion is enabled (the default)")
    p.add_argument("--playhead-color", default="#E53935")
    p.add_argument("--playhead-width", type=int, default=6)
    p.add_argument("--playhead-opacity", type=int, default=128)
    p.add_argument("--lead-in", type=float, default=0.0, help="Seconds before measure 1 starts")
    p.add_argument("--tail", type=float, default=0.0, help="Seconds to hold after the last measure")
    p.add_argument("--audio", type=Path, help="Optional audio file to mux for a preview")
    p.add_argument("--no-expand-repeats", action="store_true", help="Play written measures once, ignoring repeat navigation")

    # Detection knobs. Defaults are intentionally conservative.
    p.add_argument("--svg-scale", type=float, default=2.0, help="Raster scale used only for SVG analysis/rendering")
    p.add_argument("--ink-threshold", type=int, default=205, help="0-255; pixels darker than this count as ink")
    p.add_argument("--line-row-fraction", type=float, default=0.20, help="Minimum page-width fraction for a staff-line candidate")
    p.add_argument("--spacing-tolerance", type=float, default=0.20, help="Allowed relative variation between tab string-line spacing")
    p.add_argument("--inner-padding", type=float, default=1.35, help="Playhead inset from detected inner barline edge, in tab-string spacings")
    p.add_argument("--crop-above", type=float, default=11.0, help="Crop above top tab line, in tab-string spacings")
    p.add_argument("--crop-below", type=float, default=4.0, help="Crop below bottom tab line, in tab-string spacings")

    p.add_argument("--analyze-only", action="store_true", help="Only detect measures/bounds and write debug PNGs; do not render video")
    p.add_argument("--debug-dir", type=Path, default=Path("tab_debug"))
    p.add_argument("--overrides", type=Path, help="Optional JSON file with per-measure playhead-bound corrections")
    return p


def resolve_musicxml_path(explicit: Optional[Path], xml_dir: Path) -> Path:
    if explicit is not None:
        return explicit
    if not xml_dir.exists():
        raise FileNotFoundError(f"No MusicXML file given and --xml-dir {xml_dir} does not exist.")
    candidates = sorted(
        [p for p in xml_dir.iterdir() if p.suffix.lower() in (".xml", ".musicxml")],
        key=natural_key,
    )
    if not candidates:
        raise FileNotFoundError(f"No .xml/.musicxml file found in {xml_dir}.")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(f"Expected exactly one MusicXML file in {xml_dir}, found: {names}")
    return candidates[0]


def resolve_svg_paths(explicit: list[Path], svg_dir: Path) -> list[Path]:
    if explicit:
        return sorted(explicit, key=natural_key)
    if not svg_dir.exists():
        raise FileNotFoundError(f"No SVG files given and --svg-dir {svg_dir} does not exist.")
    candidates = sorted(svg_dir.glob("*.svg"), key=natural_key)
    if not candidates:
        raise FileNotFoundError(f"No .svg files found in {svg_dir}.")
    return candidates


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.transparent:
        args.background = "transparent"

    args.musicxml = resolve_musicxml_path(args.musicxml, args.xml_dir)
    svg_paths = resolve_svg_paths(args.svg, args.svg_dir)
    for path in [args.musicxml, *svg_paths]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.audio and not args.audio.exists():
        raise FileNotFoundError(args.audio)

    print(f"MusicXML input: {args.musicxml}")
    print(f"SVG pages ({len(svg_paths)}): {', '.join(p.name for p in svg_paths)}")

    measures, warnings = parse_musicxml(args.musicxml, args.part)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"MusicXML: {len(measures)} written measures")

    pages: list[Image.Image] = []
    geoms: list[StaffGeometry] = []
    for pi, svg_path in enumerate(svg_paths):
        print(f"Analyzing SVG page {pi + 1}: {svg_path.name}")
        page = render_svg(svg_path, args.svg_scale)
        pages.append(page)
        found = detect_page_geometries(
            page,
            page_index=pi,
            starting_measure_index=len(geoms),
            strings=args.strings,
            threshold=args.ink_threshold,
            row_fraction=args.line_row_fraction,
            spacing_tolerance=args.spacing_tolerance,
            inner_padding_spaces=args.inner_padding,
            crop_above_spaces=args.crop_above,
            crop_below_spaces=args.crop_below,
        )
        print(f"  detected {len(found)} one-measure tab systems")
        geoms.extend(found)

    if len(geoms) < len(measures):
        deficit_before = len(measures) - len(geoms)
        geoms = reconcile_measure_counts(
            geoms, measures, pages, args.ink_threshold, args.inner_padding
        )
        recovered = deficit_before - (len(measures) - len(geoms))
        if recovered > 0:
            print(
                f"Recovered {recovered} measure(s) that shared a system with another empty measure."
            )

    if len(geoms) == len(measures):
        refine_furniture_starts(geoms, measures, pages, args.ink_threshold)
        matched, eligible = compute_note_breakpoints(geoms, measures, pages, args.ink_threshold)
        if eligible:
            print(
                f"Note-level playhead timing: {matched}/{eligible} measures matched per-note detail "
                "(the rest fall back to a constant pace across the measure)."
            )

    apply_geometry_overrides(geoms, args.overrides)
    write_debug_pages(pages, geoms, args.debug_dir, [m.number for m in measures])
    print(f"Debug output: {args.debug_dir.resolve()}")

    if len(geoms) != len(measures):
        print(
            f"\nSTOP: detected {len(geoms)} tab systems but MusicXML contains {len(measures)} measures.\n"
            f"Open {args.debug_dir}/page_###_detected.png and check what was detected.\n"
            "This mismatch must be fixed before rendering so measure-to-system mapping cannot drift.",
            file=sys.stderr,
        )
        return 2

    if args.analyze_only:
        print("Analysis complete. Green = playhead start; red = playhead end; blue = detected barline clusters.")
        return 0

    if args.no_expand_repeats:
        sequence = list(range(len(measures)))
    else:
        sequence = expand_basic_repeats(measures)

    timeline, total_duration = build_timeline(measures, sequence, args.lead_in, args.tail)
    print(f"Playback sequence: {len(sequence)} measure visits")
    print(f"Video duration: {total_duration:.3f} s")

    render_pages = pages
    if args.invert_colors:
        render_pages = [invert_to_transparent(p, args.foreground_color) for p in pages]

    scale = compute_global_scale(geoms, args.width, args.height, args.margin, args.crop_above, args.crop_below)
    prepared: list[RenderedMeasure] = []
    for g in geoms:
        prepared.append(
            prepare_measure_canvas(
                render_pages[g.page_index], g, args.width, args.height, args.margin, args.background, scale, args.crop_above
            )
        )

    render_video(
        output=args.output,
        prepared=prepared,
        timeline=timeline,
        total_duration=total_duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        playhead_color=args.playhead_color,
        playhead_width=args.playhead_width,
        playhead_opacity=args.playhead_opacity,
        background=args.background,
        audio=args.audio,
    )
    print(f"Wrote: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)