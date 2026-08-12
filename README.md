# Guitar Pro Tab Video Exporter

Turns a Guitar Pro song into a video of its tablature with a moving playhead,
synced to the song's tempo/timing — one measure per line, ready to overlay
on a playthrough video (transparent background, white ink, by default).

## How it works

`generate_tab_video.py`:
1. Reads note timing (tempo, measure durations, repeats) from a MusicXML export.
2. Rasterizes the SVG page exports and automatically detects each 6-line TAB
   staff, treating every detected staff as one measure/system.
3. Renders a video where the correct measure is shown for its duration, with
   a vertical playhead line sweeping across it — no manual keyframing needed.

## Setup

Requires a working Python + native `cairo` install (needed by `cairosvg`).
On macOS, the system Python has SIP restrictions that break `cairosvg`'s
`dlopen` of Homebrew's `libcairo`, so use a Homebrew Python instead:

```bash
brew install cairo python@3.13
python3.13 -m venv .venv313
source .venv313/bin/activate
pip install pillow numpy cairosvg imageio-ffmpeg
```

`ffmpeg` itself is not required separately — `imageio-ffmpeg` bundles it.

## Usage

1. In Guitar Pro, export the song as **MusicXML**.
2. In Guitar Pro, export the song as **SVG** (one file per page).
3. Put the `.xml`/`.musicxml` file in the `xml/` folder (exactly one file).
4. Put all the `.svg` page files in the `svg/` folder.
5. Run:
   ```bash
   source .venv313/bin/activate
   python generate_tab_video.py
   ```

This auto-discovers the MusicXML file from `xml/` and all SVG pages from
`svg/`, then writes a transparent, white-ink `tab.mov` in the project root.

Before doing a full render, it's worth sanity-checking detection first:

```bash
python generate_tab_video.py --analyze-only
```

This writes annotated PNGs to `tab_debug/` (green = playhead start, red =
playhead end, blue = detected barline clusters, orange = crop box per
measure) without rendering any video. If the detected measure count doesn't
match the MusicXML measure count, the script stops and tells you so — check
the debug images before rendering.

## Useful options

| Option | Default | Purpose |
|---|---|---|
| `-o, --output` | `tab.mov` | Output video path (must be `.mov` if transparent) |
| `--xml-dir` / `--svg-dir` | `xml/` / `svg/` | Folders to auto-discover input files from |
| `--no-invert-colors` | off | Keep normal dark-ink-on-page-background instead of the default transparent/white-ink overlay style |
| `--foreground-color` | `white` | Ink color used when inversion is enabled |
| `--background` | `transparent` | Set to a CSS color (e.g. `white`) for an opaque background instead |
| `--fps`, `--width`, `--height` | `60`, `3840`, `720` | Output video specs |
| `--playhead-color`, `--playhead-width`, `--playhead-opacity` | `#E53935`, `6`, `230` | Playhead line styling |
| `--audio` | — | Optional audio file to mux in for previewing |
| `--no-expand-repeats` | off | Play written measures once, ignoring repeat signs |
| `--overrides` | — | JSON file with per-measure playhead-bound corrections (see `apply_geometry_overrides` in the script) |

Run `python generate_tab_video.py --help` for the full list, including the
lower-level detection tuning knobs (`--ink-threshold`, `--crop-above`,
`--crop-below`, etc.) — the defaults have already been tuned against a real
song and shouldn't normally need changing.

## Notes / limitations

- Assumes one measure per rendered system/line and a 6-string TAB staff
  (`--strings` can change the string count).
- Basic forward/backward repeats are expanded automatically. Volta (1st/2nd)
  endings and D.S./D.C./Coda navigation are detected and reported as
  unsupported rather than silently guessed at.
- Full (non-`--analyze-only`) renders can take several minutes, and
  transparent ProRes 4444 `.mov` output can be several GB for a full song.