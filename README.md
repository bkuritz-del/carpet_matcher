# Carpet Pattern Candidate Matcher

This local Python tool recursively indexes machine-design BMP files and ranks them against a customer photograph. It compares grayscale structure, repeat frequencies, direction, and coarse layout rather than literal color. Each result identifies the machine type (the folder immediately above `Bitmap files`) and the BMP filename/pattern name.

## Windows desktop interface

Run the Gooey interface with:

```powershell
python carpet_gui.py --query "C:\Photos\customer-carpet.jpg"
```

The pattern library defaults to `G:\Design\Design Machine Pattern Files\_1_TEMPLATE DESIGN LIBRARY`. Choose a customer photograph and run the matcher. The folder can still be changed with the picker when needed. The program creates `carpet-index.npz` automatically on the first run; select **Rebuild index** after BMP files are added or changed. If `G:` is unavailable, connect to the company network/VPN before running the program.

Every push and pull request runs the **Build Windows EXE** GitHub Actions workflow. Download `CarpetPatternMatcher-Windows` from the workflow run's **Artifacts** section and extract `CarpetPatternMatcher.exe`. The executable does not require Python on the user's computer.

## Quick start

Use Python 3.10+ with Pillow and NumPy installed.

```powershell
python carpet_matcher.py index "G:\Design\Design Machine Pattern Files" --output carpet-index.npz
python carpet_matcher.py match customer-photo.jpg --index carpet-index.npz --top 10
```

Results are displayed in this form:

```text
 1.  84.28  Machine: 1.10g EGL (K) Designs  Pattern: k1137  File: G:\...\Bitmap files\k1137.BMP
```

For a photo containing walls, furniture, or a phone-app border, restrict matching to a carpet-only rectangle. Coordinates are fractions of image width and height:

```powershell
python carpet_matcher.py match customer-photo.jpg --index carpet-index.npz --crop 0.05,0.15,0.95,0.95
```

Add `--json` to produce machine-readable results. Rebuild the index after BMPs are added or changed.

## What the score means

The score is a relative structural-similarity ranking, not a probability or proof of identity. Machine BMP colors encode construction information and do not directly reproduce installed-carpet color and texture. Review the top candidates visually and confirm using product metadata or a physical sample.

Accuracy will improve substantially with a labeled test set containing known customer photos and their correct BMPs. Those examples can later train the ranking weights or a compact machine-learning model.
