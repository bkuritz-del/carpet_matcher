"""Gooey desktop interface for the carpet pattern candidate matcher."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from gooey import Gooey, GooeyParser

from carpet_matcher import build_index, match


DEFAULT_LIBRARY = r"G:\Design\Design Machine Pattern Files\_1_TEMPLATE DESIGN LIBRARY"


@Gooey(
    program_name="Carpet Pattern Matcher",
    program_description="Find likely machine patterns for a customer carpet photograph.",
    default_size=(900, 720),
    required_cols=1,
    optional_cols=1,
    clear_before_run=True,
    show_success_modal=False,
    show_failure_modal=True,
    progress_regex=r"^\[(?P<current>\d+)/(?P<total>\d+)\]",
    progress_expr="current / total * 100",
)
def main() -> int:
    parser = GooeyParser(description="Index the bitmap library and rank likely matching patterns.")
    required = parser.add_argument_group("Files")
    required.add_argument(
        "--library",
        default=DEFAULT_LIBRARY,
        metavar="Pattern library folder",
        help="Master folder containing machine-type folders and their Bitmap files subfolders.",
        widget="DirChooser",
    )
    required.add_argument(
        "--query",
        required=True,
        metavar="Customer carpet image",
        help="Photograph to identify.",
        widget="FileChooser",
        gooey_options={"wildcard": "Image files (*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff)|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"},
    )

    options = parser.add_argument_group("Options")
    options.add_argument(
        "--index",
        default="carpet-index.npz",
        metavar="Search index file",
        help="Reusable index. It is created automatically when it does not exist.",
        widget="FileSaver",
        gooey_options={"wildcard": "Carpet index (*.npz)|*.npz"},
    )
    options.add_argument(
        "--rebuild",
        action="store_true",
        metavar="Rebuild index",
        help="Rescan all BMP files, including newly added or changed patterns.",
        widget="CheckBox",
    )
    options.add_argument(
        "--top",
        type=int,
        default=10,
        choices=range(1, 51),
        metavar="Number of candidates",
        widget="Dropdown",
    )
    options.add_argument(
        "--crop",
        default="",
        metavar="Optional carpet crop",
        help="left,top,right,bottom fractions; example: 0.05,0.15,0.95,0.95",
    )
    options.add_argument(
        "--save-results",
        default="",
        metavar="Optional results file",
        help="Save the ranked results as JSON.",
        widget="FileSaver",
        gooey_options={"wildcard": "JSON files (*.json)|*.json"},
    )
    options.add_argument(
        "--preview",
        default="aligned-query-preview.png",
        metavar="Aligned preview image",
        help="Shows the rotated and cropped customer image that is actually searched.",
        widget="FileSaver",
        gooey_options={"wildcard": "PNG image (*.png)|*.png"},
    )
    args = parser.parse_args()

    library = Path(args.library)
    query = Path(args.query)
    index = Path(args.index)
    try:
        if not library.is_dir():
            raise FileNotFoundError(
                f"Pattern library is unavailable: {library}. "
                "Connect to the company network/VPN or choose another folder."
            )
        if args.rebuild or not index.exists():
            print("Building the pattern index. This can take several minutes the first time.")
            build_index(library, index)
            print()
        print("Finding the primary carpet direction and aligning the image...", flush=True)
        preview = Path(args.preview) if args.preview else None
        results = match(index, query, args.crop or None, args.top, preview)
        print("Comparing the aligned image with indexed patterns...\n", flush=True)
        print("MOST LIKELY PATTERNS")
        print("=" * 78)
        for result in results:
            print(f"#{result['rank']}   Score: {result['score']:.2f}")
            print(f"Machine type: {result['machine_type']}")
            print(f"Pattern name: {result['pattern_name']}")
            print(f"File: {result['path']}")
            print("-" * 78)
        print("\nScores are candidate rankings, not probabilities. Confirm against product records or a physical sample.")
        if args.save_results:
            destination = Path(args.save_results)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nSaved results to {destination}")
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
