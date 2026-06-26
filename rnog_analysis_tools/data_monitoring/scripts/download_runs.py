#!/usr/bin/env python3
"""Download RNO-G run data from the Greenland server for a station and UTC time window.

The tool connects to the data server over SSH, reads the (tiny) ``aux/runinfo.txt``
files of every run of the requested station to learn each run's start/end time, selects
the runs whose [start, end] interval overlaps the requested UTC window, and downloads
the full run directories with ``rsync``.

Discovery uses a single remote ``ssh`` command (one round-trip, greps all runinfo files);
the bulk transfer uses ``rsync`` over SSH so large ``waveforms.root`` files transfer
efficiently and resumably.

Examples
--------
    # Everything for station 23 that overlaps an 8h window on 2023-04-24 (UTC)
    ./download_runs.py --station 23 \
        --start 2023-04-24T15:00:00 --end 2023-04-24T23:00:00 \
        --host rno-g@data.example.org --remote-root /data/handcarry/rootified \
        --dest ./downloads

    # Timestamps may also be given as UNIX seconds, and an SSH key can be passed
    ./download_runs.py --station 11 --start 1682353428 --end 1682362114 \
        --host data.example.org --user rno-g --ssh-key ~/.ssh/rnog \
        --remote-root /data/rootified --dest ./downloads --dry-run
"""

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


# When a run's RUN-END-TIME is missing/invalid we can't compute a real overlap, so we
# fall back to keeping it if its start is within this tolerance of the requested window.
UNKNOWN_END_TOLERANCE_S = 2 * 3600  # 2 hours


@dataclass
class RunInfo:
    run: int
    start: float            # UNIX seconds (UTC)
    end: Optional[float]    # UNIX seconds (UTC); None if RUN-END-TIME is missing/invalid
    path: str               # absolute remote path to the run directory

    def overlaps(self, win_start: float, win_end: float) -> bool:
        """Any-overlap test against the requested [win_start, win_end] window.

        With a known end time this is a normal interval overlap. When the end time is
        unknown (None) we instead keep the run if its start falls within
        ``UNKNOWN_END_TOLERANCE_S`` of the window.
        """
        if self.end is not None:
            return self.start <= win_end and self.end >= win_start
        return (win_start - UNKNOWN_END_TOLERANCE_S) <= self.start <= (win_end + UNKNOWN_END_TOLERANCE_S)


def parse_time(value: str) -> float:
    """Parse a UTC time given as ISO-8601 or as UNIX seconds into a float UNIX timestamp."""
    try:
        return float(value)
    except ValueError:
        pass

    s = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Could not parse time {value!r}; use ISO-8601 (2023-04-24T15:00:00) or UNIX seconds"
        )
    # Treat naive timestamps as UTC.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)

    return parsed.timestamp()


def fmt_utc(ts: Optional[float]) -> str:
    if ts is None:
        return "(unknown end)"
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_number_constraints(start_year: int, end_year: int):
    """Restrict candidate run directories by the year-based run-number naming scheme.

    Since 2024, every run number in a year is ``YYxxxx`` (last two year digits + 4
    digits): e.g. 2026 -> 260000..269999. Before 2024 run numbers are not year-encoded
    and are always < 5000.

    Returns ``(glob_suffixes, allowed_ranges)`` where the globs (relative to the station
    directory) precisely restrict the remote listing, and ``allowed_ranges`` is a list of
    inclusive ``(lo, hi)`` run-number bounds used as a safety filter after parsing.
    Run directories are not zero-padded (run5, run1144, run260694), hence one glob per digit length.
    """
    suffixes: list = []
    ranges: list = []
    old_added = False
    for year in range(start_year, end_year + 1):
        if year >= 2024:
            yy = year % 100
            base = yy * 10000
            suffixes.append(f"run{yy:02d}[0-9][0-9][0-9][0-9]")  # YY0000..YY9999
            ranges.append((base, base + 9999))
        elif not old_added:
            # pre-2024: run numbers < 5000, no year encoding -> cannot narrow by year
            suffixes += [
                "run[0-9]",                  # 0..9
                "run[0-9][0-9]",             # 10..99
                "run[0-9][0-9][0-9]",        # 100..999
                "run[0-4][0-9][0-9][0-9]",   # 1000..4999
            ]
            ranges.append((0, 4999))
            old_added = True
    return suffixes, ranges


def ssh_base_cmd(ssh_key: Optional[str]) -> list:
    """Common ssh options shared by discovery and the rsync -e string."""
    cmd = ["ssh"]
    if ssh_key:
        cmd += ["-i", os.path.expanduser(ssh_key)]
    # Fail fast instead of hanging on an unreachable host.
    cmd += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    return cmd


def remote_target(host: str, user: Optional[str]) -> str:
    return f"{user}@{host}" if user else host


def discover_runs(args, station: int) -> list:
    """Read start/end time of candidate runs for one station in a single remote command.

    Candidate run directories are restricted up-front by the year-based run-number naming
    scheme (see :func:`run_number_constraints`) so the server never stats runs from other
    years -- important when a station holds thousands of runs. ``--no-run-filter`` disables
    this and scans every run.
    """
    station_dir = f"{args.remote_root.rstrip('/')}/station{station}"

    if args.run:
        # Explicit run numbers -> exact run<N> globs (still filtered by the time window).
        runs = sorted(set(args.run))
        suffixes, ranges = [f"run{n}" for n in runs], None
        print("  run-number filter: explicit runs " + ", ".join(str(n) for n in runs))
    elif args.no_run_filter:
        suffixes, ranges = ["run*"], None
        print("  run-number filter: (none, scanning all runs)")
    else:
        start_year = dt.datetime.fromtimestamp(args.start, dt.timezone.utc).year
        end_year = dt.datetime.fromtimestamp(args.end, dt.timezone.utc).year
        suffixes, ranges = run_number_constraints(start_year, end_year)
        if not suffixes:  # window predates the scheme entirely; fall back to a full scan
            suffixes, ranges = ["run*"], None
        if ranges is None:
            print("  run-number filter: (none, scanning all runs)")
        else:
            print("  run-number filter: " + ", ".join(f"{lo}-{hi}" for lo, hi in ranges))

    glob_paths = " ".join(f"{station_dir}/{suf}/" for suf in suffixes)

    # POSIX-sh snippet executed on the server: for each run, emit "run|start|end".
    # awk strips the "KEY = value" formatting; a missing end-time yields an empty field.
    # Unmatched globs stay literal and are skipped by the [ -f "$info" ] guard.
    remote_script = (
        f'for d in {glob_paths}; do '
        'info="${d}aux/runinfo.txt"; '
        '[ -f "$info" ] || continue; '
        'run=$(basename "${d%/}"); '
        's=$(awk -F= "/RUN-START-TIME/ {gsub(/[ \\t]/,\\"\\",\\$2); print \\$2}" "$info"); '
        'e=$(awk -F= "/RUN-END-TIME/ {gsub(/[ \\t]/,\\"\\",\\$2); print \\$2}" "$info"); '
        'printf "%s|%s|%s\\n" "$run" "$s" "$e"; '
        'done'
    )

    cmd = ssh_base_cmd(args.ssh_key)
    cmd += [remote_target(args.host, args.user), remote_script]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"ERROR: remote discovery failed (exit {proc.returncode}):\n{proc.stderr.strip()}")

    runs = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = (line.split("|") + ["", ""])[:3]
        run_name, s, e = parts
        try:
            run_num = int(run_name.replace("run", ""))
            start = float(s)
        except ValueError:
            print(f"WARNING: skipping unparseable run entry: {line!r}", file=sys.stderr)
            continue

        # Safety net: enforce the run-number ranges even if a glob matched something extra.
        if ranges is not None and not any(lo <= run_num <= hi for lo, hi in ranges):
            continue

        # A missing/unparseable RUN-END-TIME (interrupted or in-progress runs) is kept with
        # end=None; such runs are matched by start-time proximity (see RunInfo.overlaps).
        try:
            end = float(e)
        except ValueError:
            end = None

        path = f"{station_dir}/{run_name}"
        runs.append(RunInfo(run=run_num, start=start, end=end, path=path))

    return sorted(runs, key=lambda r: r.run)


def rsync_run(run: RunInfo, args, station: int) -> int:
    """rsync one full run directory into <dest>/station<id>/run<N>/."""
    local_dir = os.path.join(args.dest, f"station{station}", f"run{run.run}")
    os.makedirs(local_dir, exist_ok=True)

    ssh_e = " ".join(shlex.quote(p) for p in ssh_base_cmd(args.ssh_key))
    src = f"{remote_target(args.host, args.user)}:{run.path}/"

    cmd = ["rsync", "-a", "--partial", "--progress", "--exclude=rootified-list.txt", "-e", ssh_e]
    if args.dry_run:
        cmd.append("--dry-run")
    cmd += [src, local_dir + "/"]

    print(f"  rsync run{run.run} -> {local_dir}")
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RNO-G run data covering a UTC time window for a station.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--station", type=int, required=True, nargs="+", metavar="ID",
                        help="One or more station ids, e.g. --station 23 or --station 11 12 13")
    parser.add_argument("--start", type=parse_time, required=True,
                        help="Window start, UTC ISO-8601 (e.g. 2026-04-24T15:00:00) or UNIX seconds")
    parser.add_argument("--end", type=parse_time, required=True,
                        help="Window end, UTC ISO-8601 (e.g. 2026-04-24T23:00:00) or UNIX seconds")

    parser.add_argument("--host", required=True, help="Server hostname (or user@host)")
    parser.add_argument("--user", default=None, help="SSH user (if not embedded in --host)")
    parser.add_argument("--ssh-key", default=None, help="Path to SSH private key (optional)")
    parser.add_argument("--remote-root", default="/data/rootified/",
                        help="Remote dir containing station<id>/run<N> folders, e.g. /data/rootified")

    parser.add_argument("--dest", default="./downloads", help="Local download directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="List matching runs and show rsync actions without transferring")
    parser.add_argument("--run", type=int, nargs="+", default=None, metavar="N",
                        help="Only consider these explicit run numbers (still filtered by the "
                             "time window); overrides the year-based run-number restriction")
    parser.add_argument("--no-run-filter", action="store_true",
                        help="Disable the year-based run-number restriction and scan every run "
                             "(use if the naming-scheme assumptions don't hold)")

    args = parser.parse_args()

    if args.end < args.start:
        sys.exit("ERROR: --end is before --start")

    print(f"Searching runs overlapping {fmt_utc(args.start)}  ->  {fmt_utc(args.end)}")
    print(f"  server: {remote_target(args.host, args.user)}:{args.remote_root}")
    print(f"  stations: {', '.join(str(s) for s in args.station)}")

    total_matches = 0
    failures = []  # (station, run) pairs
    for station in args.station:
        print(f"\n=== Station {station} ===")
        # Always create the per-station destination dir, even if nothing matches.
        station_dest = os.path.join(args.dest, f"station{station}")
        os.makedirs(station_dest, exist_ok=True)

        all_runs = discover_runs(args, station)
        if not all_runs:
            print("  No runs found for this station (check --remote-root / --station).")
            continue

        matches = [r for r in all_runs if r.overlaps(args.start, args.end)]
        print(f"  Found {len(all_runs)} candidate runs; {len(matches)} overlap the window:")
        for r in matches:
            print(f"    run{r.run:<8} {fmt_utc(r.start)}  ->  {fmt_utc(r.end)}")

        if not matches:
            continue
        total_matches += len(matches)

        if args.dry_run:
            print(f"  [dry-run] would download {len(matches)} run(s) into {station_dest}:")
        else:
            print(f"  Downloading {len(matches)} run(s) into {station_dest} ...")

        for r in matches:
            if rsync_run(r, args, station) != 0:
                failures.append((station, r.run))

    if failures:
        joined = ", ".join(f"station{s}/run{r}" for s, r in failures)
        sys.exit(f"\nERROR: rsync failed for: {joined}")

    if total_matches == 0:
        print("\nNothing to download.")
    else:
        print("\nDone." if not args.dry_run else "\n[dry-run] complete.")


if __name__ == "__main__":
    main()
