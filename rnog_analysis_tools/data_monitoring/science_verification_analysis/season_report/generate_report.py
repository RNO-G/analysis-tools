"""Run the existing SVA and create a compact HTML report from its output.

This module deliberately invokes ``science_verification_analysis_main.py`` as
a subprocess.  It does not import or modify the SVA entry point, which keeps
the season-report feature isolated and easy to merge.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable

try:
    from .heatmaps import generate_health_heatmaps
except ImportError:  # Direct execution: python season_report/generate_report.py
    from heatmaps import generate_health_heatmaps


PROJECT_DIR = Path(__file__).resolve().parents[1]
SVA_SCRIPT = PROJECT_DIR / "science_verification_analysis_main.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Science Verification Analysis and build its HTML season report."
    )
    parser.add_argument("-st", "--station_id", type=int, help="Station to analyze")
    parser.add_argument("--data_location", default="uchicago")
    parser.add_argument("-ex", "--exclude-runs", nargs="+", type=int, default=[])
    parser.add_argument(
        "--debug_plot",
        action="store_true",
        help=(
            "Accepted for command compatibility; this SVA branch generates its "
            "diagnostic plots automatically and has no debug CLI flag"
        ),
    )
    parser.add_argument(
        "--existing-result",
        type=Path,
        help="Build the report from an existing SVA result without running SVA",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--runs", nargs="+", type=int)
    selection.add_argument("--run_range", nargs=2, type=int)
    selection.add_argument("--time_range", nargs=2)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.existing_result:
        result_dir = args.existing_result.expanduser().resolve()
    else:
        if args.station_id is None:
            parser.error("--station_id is required unless --existing-result is used")
        if not any((args.runs, args.run_range, args.time_range)):
            parser.error("one of --runs, --run_range, or --time_range is required")
        results_base = result_base_for(args.data_location)
        result_dir = run_sva(args, results_base)

    validate_result_directory(result_dir)
    report_dir = result_dir.parent / "season_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(result_dir)
    combined_csv, status_csv = collect_per_run_health(
        runs=metadata["runs"],
        station_id=metadata["station"],
        data_location=args.data_location,
        results_base=result_dir.parent,
        report_dir=report_dir,
        retained_result=result_dir,
    )
    heatmap_paths = generate_health_heatmaps(combined_csv, report_dir / "plots")
    report_path = report_dir / "index.html"
    report_path.write_text(
        render_report(result_dir, report_path, heatmap_paths, combined_csv, status_csv),
        encoding="utf-8",
    )
    print(f"Season report written to {report_path}")
    print(f"SVA source directory: {result_dir}")
    return report_path


def result_base_for(data_location: str) -> Path:
    if data_location == "uchicago":
        return PROJECT_DIR / "outputs"
    if data_location == "desy":
        return Path("/pnfs/ifh.de/acs/radio/diskonly/NuRadioMC/science_verification_analysis")
    # This mirrors the current SVA behavior for a custom data location.
    return Path("/pnfs/ifh.de/acs/radio/diskonly/NuRadioMC/science_verification_analysis")


def run_sva(args: argparse.Namespace, results_base: Path) -> Path:
    results_base.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in result_directories(results_base)}
    command = [
        sys.executable,
        str(SVA_SCRIPT),
        "-st",
        str(args.station_id),
        "--data_location",
        args.data_location,
    ]
    if args.runs:
        command.extend(["--runs", *(str(run) for run in args.runs)])
    elif args.run_range:
        command.extend(["--run_range", *(str(run) for run in args.run_range)])
    else:
        command.extend(["--time_range", *args.time_range])
    if args.exclude_runs:
        command.extend(["--exclude-runs", *(str(run) for run in args.exclude_runs)])
    if args.debug_plot:
        print(
            "Note: --debug_plot is not forwarded because the current SVA branch "
            "generates diagnostic plots automatically."
        )

    print("Running SVA:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)

    after = {path.resolve() for path in result_directories(results_base)}
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        candidates = [
            path for path in after
            if f"station-{args.station_id}_" in path.name
        ]
        if not candidates:
            raise RuntimeError(f"SVA completed but no result directory was found under {results_base}")
        result = max(candidates, key=lambda path: path.stat().st_mtime)
        if len(created) > 1:
            print(f"Warning: found {len(created)} new result directories; using {result}")
        return result
    return created[0]


def collect_per_run_health(runs: list[int], station_id: int, data_location: str,
                           results_base: Path, report_dir: Path,
                           retained_result: Path) -> tuple[Path, Path]:
    """Run SVA once per run, retain its CSV rows, and remove its output."""
    data_dir = report_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    combined_path = data_dir / "channel_health_by_run.csv"
    status_path = data_dir / "run_processing_status.csv"
    fieldnames, rows = read_combined_health(combined_path)
    requested = {int(run) for run in runs}
    if rows and "Station" not in fieldnames:
        fieldnames, rows = [], []
    rows = [
        row for row in rows
        if int(row["Station"]) == station_id and int(row["Run"]) in requested
    ]
    completed = {int(row["Run"]) for row in rows}
    statuses = read_status_rows(status_path)
    statuses = {
        run: row for run, row in statuses.items()
        if row.get("Station") == str(station_id) and run in requested
    }

    for position, run in enumerate(runs, start=1):
        if int(run) in completed:
            print(f"Per-run SVA {position}/{len(runs)}: run {run} already collected; skipping")
            continue
        print(f"Per-run SVA {position}/{len(runs)}: run {run}")
        before = {path.resolve() for path in result_directories(results_base)}
        command = [
            sys.executable, str(SVA_SCRIPT), "-st", str(station_id),
            "--data_location", data_location, "--runs", str(run), "--summary-only",
        ]
        error = ""
        try:
            subprocess.run(command, cwd=PROJECT_DIR, check=True)
        except subprocess.CalledProcessError as exception:
            error = f"SVA exited with status {exception.returncode}"

        after = {path.resolve() for path in result_directories(results_base)}
        created = sorted(after - before, key=lambda path: path.stat().st_mtime)
        temporary = [path for path in created if path.resolve() != retained_result.resolve()]

        if not error:
            try:
                if len(temporary) != 1:
                    raise RuntimeError(f"Expected one per-run result directory, found {len(temporary)}")
                new_fields, new_rows = extract_validation_rows(temporary[0], station_id, int(run))
                if fieldnames and new_fields != fieldnames:
                    raise ValueError("Per-run validation-summary columns changed")
                fieldnames = fieldnames or new_fields
                rows = [row for row in rows if int(row["Run"]) != int(run)] + new_rows
                rows.sort(key=lambda row: (int(row["Run"]), int(row["Channel"])))
                write_csv_atomic(combined_path, fieldnames, rows)
                verify_run_in_combined(combined_path, int(run), len(new_rows))
                completed.add(int(run))
                statuses[int(run)] = {
                    "Station": str(station_id), "Run": str(run), "Status": "complete",
                    "Rows": str(len(new_rows)), "Message": "",
                }
            except Exception as exception:  # Preserve progress and continue with later runs.
                error = str(exception)

        if error:
            print(f"Warning: run {run} was not collected: {error}", file=sys.stderr)
            statuses[int(run)] = {
                "Station": str(station_id), "Run": str(run), "Status": "failed", "Rows": "0", "Message": error,
            }
        write_status_atomic(status_path, statuses)

        for path in temporary:
            remove_temporary_result(path, results_base, station_id, int(run))

    if not combined_path.exists():
        raise RuntimeError("No per-run validation summaries were collected")
    return combined_path, status_path


def extract_validation_rows(result_dir: Path, station_id: int,
                            run: int) -> tuple[list[str], list[dict[str, str]]]:
    matches = sorted((result_dir / "channel_health_summary").glob("validation_summary_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one validation summary in {result_dir}, found {len(matches)}")
    with matches[0].open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "Channel" not in reader.fieldnames:
            raise ValueError(f"Invalid validation summary: {matches[0]}")
        fields = ["Station", "Run", *reader.fieldnames]
        rows = [
            {"Station": str(station_id), "Run": str(run), **{key: value for key, value in row.items()}}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"Validation summary contains no channels: {matches[0]}")
    channels = [int(row["Channel"]) for row in rows]
    if len(channels) != len(set(channels)):
        raise ValueError(f"Validation summary contains duplicate channels: {matches[0]}")
    return fields, rows


def read_combined_health(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def read_status_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as source:
        return {int(row["Run"]): row for row in csv.DictReader(source)}


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_status_atomic(path: Path, statuses: dict[int, dict[str, str]]) -> None:
    rows = [statuses[run] for run in sorted(statuses)]
    write_csv_atomic(path, ["Station", "Run", "Status", "Rows", "Message"], rows)


def verify_run_in_combined(path: Path, run: int, expected_rows: int) -> None:
    _, rows = read_combined_health(path)
    actual = sum(int(row["Run"]) == run for row in rows)
    if actual != expected_rows:
        raise RuntimeError(
            f"Combined CSV verification failed for run {run}: expected {expected_rows} rows, found {actual}"
        )


def remove_temporary_result(path: Path, results_base: Path,
                            station_id: int, run: int) -> None:
    resolved = path.resolve()
    expected_parent = results_base.resolve()
    expected_fragment = f"station-{station_id}_run{run}-run{run}_"
    if resolved.parent != expected_parent or expected_fragment not in resolved.name:
        raise RuntimeError(f"Refusing to remove unexpected result directory: {resolved}")
    shutil.rmtree(resolved)
    print(f"Removed temporary per-run result: {resolved.name}")


def result_directories(results_base: Path) -> Iterable[Path]:
    if not results_base.exists():
        return []
    return (
        path for path in results_base.iterdir()
        if path.is_dir() and path.name != "season_report"
    )


def validate_result_directory(result_dir: Path) -> None:
    if not result_dir.is_dir():
        raise FileNotFoundError(f"SVA result directory does not exist: {result_dir}")
    if not (result_dir / "plots" / "standard_plots").is_dir():
        raise ValueError(f"No plots/standard_plots directory found in {result_dir}")


def read_metadata(result_dir: Path) -> dict[str, object]:
    readmes = list(result_dir.glob("README_shifters.txt"))
    if not readmes:
        raise FileNotFoundError(f"README_shifters.txt was not found in {result_dir}")
    text = readmes[0].read_text(encoding="utf-8")
    header = re.search(r"Station\s+(\d+),\s+Runs:\s*\[([^]]*)\]", text)
    interval = re.search(r"Time Range:\s*(.+?)\s+to\s+(.+?)\s*$", text, re.MULTILINE)
    if not header or not interval:
        raise ValueError(f"Could not parse station/run/time metadata from {readmes[0]}")
    runs = [int(value.strip()) for value in header.group(2).split(",") if value.strip()]

    analyzed_count = len(runs)
    logs = list((result_dir / "logs").glob("*.log"))
    if logs:
        log_text = logs[0].read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Successfully read and combined data from (\d+) runs", log_text)
        if match:
            analyzed_count = int(match.group(1))
    return {
        "station": int(header.group(1)),
        "runs": runs,
        "first_run": min(runs) if runs else None,
        "last_run": max(runs) if runs else None,
        "analyzed_count": analyzed_count,
        "start_time": interval.group(1).strip(),
        "end_time": interval.group(2).strip(),
    }


def find_standard_plot(result_dir: Path, pattern: str) -> Path:
    matches = sorted((result_dir / "plots" / "standard_plots").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one standard plot matching {pattern!r}, found {len(matches)}"
        )
    return matches[0]


def link_from(report_path: Path, target: Path) -> str:
    return Path(os.path.relpath(target, report_path.parent)).as_posix()


def pdf_panel(title: str, plot: Path, report_path: Path) -> str:
    link = html.escape(link_from(report_path, plot), quote=True)
    return (
        '<section class="plot-card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<object data="{link}" type="application/pdf">'
        f'<p>PDF preview unavailable. <a href="{link}">Open the plot</a>.</p>'
        "</object>"
        f'<a class="open-link" href="{link}">Open full plot</a>'
        "</section>"
    )


def image_panel(title: str, plot: Path, report_path: Path) -> str:
    link = html.escape(link_from(report_path, plot), quote=True)
    return (
        '<section class="health-card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<a href="{link}"><img src="{link}" alt="{html.escape(title, quote=True)}"></a>'
        "</section>"
    )


def plot_directory_links(result_dir: Path, report_path: Path) -> str:
    plots_root = result_dir / "plots"
    sections = []
    for directory in sorted(path for path in plots_root.rglob("*") if path.is_dir()):
        files = sorted(path for path in directory.iterdir() if path.is_file())
        if not files:
            continue
        directory_label = directory.relative_to(result_dir).as_posix()
        links = "".join(
            f'<li><a href="{html.escape(link_from(report_path, path), quote=True)}">'
            f"{html.escape(path.relative_to(result_dir).as_posix())}</a></li>"
            for path in files
        )
        sections.append(
            f'<details><summary>{html.escape(directory_label)} ({len(files)})</summary>'
            f"<ul>{links}</ul></details>"
        )
    return "".join(sections)


def render_report(result_dir: Path, report_path: Path,
                  heatmap_paths: dict[str, Path], combined_csv: Path,
                  status_csv: Path) -> str:
    metadata = read_metadata(result_dir)
    deep = find_standard_plot(result_dir, "force_time_integrated_deep_spectra_unnormalized_*.pdf")
    surface = find_standard_plot(result_dir, "force_time_integrated_surface_spectra_unnormalized_*.pdf")
    trigger = find_standard_plot(result_dir, "trigger_rates_over_time_*.pdf")
    rms_force = find_standard_plot(result_dir, "rms_against_time_force_*.pdf")
    source_link = html.escape(link_from(report_path, result_dir), quote=True)
    run_range = (
        f"{metadata['first_run']}–{metadata['last_run']}"
        if metadata["first_run"] != metadata["last_run"] else str(metadata["first_run"])
    )
    overall_label = "Channel Health (FORCE)"
    overall = ""
    if overall_label in heatmap_paths:
        overall = image_panel("Overall channel health by run", heatmap_paths[overall_label], report_path)
    individual_heatmaps = "".join(
        image_panel(label, path, report_path)
        for label, path in heatmap_paths.items() if label != overall_label
    )
    combined_link = html.escape(link_from(report_path, combined_csv), quote=True)
    status_link = html.escape(link_from(report_path, status_csv), quote=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Station {metadata['station']} season report</title>
  <style>
    :root {{ color-scheme: light; --border:#d9dee5; --ink:#1d2733; --muted:#5d6b7a; --panel:#f5f7f9; }}
    body {{ margin:0; font-family:system-ui,-apple-system,sans-serif; color:var(--ink); background:#fff; }}
    main {{ max-width:1500px; margin:0 auto; padding:2rem; }}
    h1 {{ margin-bottom:.25rem; }} .source {{ color:var(--muted); margin-top:0; }}
    table {{ border-collapse:collapse; width:100%; margin:1.5rem 0 2rem; }}
    th,td {{ border:1px solid var(--border); padding:.75rem; text-align:left; }} th {{ background:var(--panel); }}
    .plot-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }}
    .plot-card {{ border:1px solid var(--border); border-radius:.5rem; padding:1rem; background:var(--panel); }}
    .plot-card h3 {{ margin-top:0; }} object {{ width:100%; height:620px; background:white; }}
    .trigger {{ margin-top:1rem; }} .trigger object {{ height:700px; }}
    .open-link {{ display:inline-block; margin-top:.6rem; }} details {{ margin:.6rem 0; }}
    .health-card {{ border:1px solid var(--border); border-radius:.5rem; padding:1rem; background:#fff; margin:1rem 0; }}
    .health-card img {{ display:block; width:100%; height:auto; }}
    .health-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }}
    summary {{ cursor:pointer; font-weight:600; }} li {{ margin:.3rem 0; overflow-wrap:anywhere; }}
    @media (max-width:900px) {{ .plot-grid,.health-grid {{ grid-template-columns:1fr; }} object {{ height:500px; }} }}
  </style>
</head>
<body><main>
  <h1>Station {metadata['station']} season report</h1>
  <p class="source">Source: <a href="{source_link}">{html.escape(result_dir.name)}</a></p>
  <table>
    <thead><tr><th>Analyzed time period (UTC)</th><th>Run range</th><th>Analyzed runs</th></tr></thead>
    <tbody><tr><td>{html.escape(str(metadata['start_time']))} to {html.escape(str(metadata['end_time']))}</td>
    <td>{html.escape(run_range)}</td><td>{metadata['analyzed_count']}</td></tr></tbody>
  </table>
  <h2>Channel health as a function of run</h2>
  <p><a href="{combined_link}">Download combined channel-health CSV</a> ·
     <a href="{status_link}">View per-run processing status</a></p>
  {overall}
  <div class="health-grid">{individual_heatmaps}</div>
  <h2>Full-period standard plots</h2>
  <div class="plot-grid">
    {pdf_panel('Time-integrated deep-channel spectrum (unnormalized)', deep, report_path)}
    {pdf_panel('Time-integrated surface-channel spectrum (unnormalized)', surface, report_path)}
  </div>
  <div class="trigger">{pdf_panel('Trigger rates over time', trigger, report_path)}</div>
  <div class="trigger">{pdf_panel('FORCE RMS against time', rms_force, report_path)}</div>
  <h2>All plot directories</h2>
  {plot_directory_links(result_dir, report_path)}
</main></body></html>
"""


if __name__ == "__main__":
    main()
