#!/usr/bin/env python3
"""Generate paper fragments and SVGs only from checked-in benchmark evidence."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "profiling" / "results"
GENERATED = ROOT / "docs" / "paper" / "generated"
OUTPUT = GENERATED / "profiling_results.tex"


def latex(value: Any) -> str:
    if value is None:
        return "--"
    escapes = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(character, character) for character in str(value))


def completed_manifests() -> list[tuple[Path, dict[str, Any]]]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = value.get("results") if isinstance(value, dict) else None
        if isinstance(results, dict) and any(v is not None for v in results.values()):
            manifests.append((path, value))
    return manifests


def quantile_csvs() -> list[Path]:
    usable: list[Path] = []
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                if csv.DictReader(stream).fieldnames:
                    usable.append(path)
        except OSError:
            continue
    return usable


def result_rows(manifests: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, manifest in manifests:
        results = manifest.get("results", {})
        latency = results.get("latency_microseconds", {}) if isinstance(results, dict) else {}
        model = manifest.get("model", {}) if isinstance(manifest.get("model"), dict) else {}
        load = manifest.get("load", {}) if isinstance(manifest.get("load"), dict) else {}
        rows.append({
            "artifact": path.name,
            "commit": manifest.get("commit", "unknown"),
            "backend": model.get("numeric_backend", "unknown"),
            "scope": manifest.get("measurement_scope", "unspecified"),
            "rate": load.get("requests_per_second"),
            "p50": latency.get("p50"),
            "p90": latency.get("p90"),
            "p99": latency.get("p99"),
            "p999": latency.get("p999"),
            "throughput": results.get("throughput_per_second"),
            "error_rate": results.get("error_rate"),
        })
    return rows


def short_label(backend: Any) -> str:
    text = str(backend)
    lowered = text.lower()
    if "graph" in lowered:
        return "Graph replay"
    if "direct" in lowered:
        return "Direct CUDA"
    if "shared" in lowered or "ipc" in lowered:
        return "HTTP + IPC"
    return text[:24]


def artifact_label(filename: str) -> str:
    if "run-" in filename and filename.endswith("-result.json"):
        return "Linux HTTP/IPC run"
    if filename.startswith("cuda_direct"):
        return "CUDA direct runs"
    if filename.startswith("cuda_graph_replay"):
        return "CUDA graph runs"
    return filename[:28]


def compact_number(value: Any, decimals: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "--"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.{decimals}f}"


def latency_svg(rows: list[dict[str, Any]]) -> None:
    # Keep the visualization comparison-scoped: the API observation is shown in
    # its own CDF artifact and must not dwarf or be conflated with CUDA timings.
    usable = [
        row for row in rows
        if "CPU wall time per warmed" in str(row.get("scope"))
        and isinstance(row.get("p50"), (int, float))
        and isinstance(row.get("p99"), (int, float))
    ]
    if not usable:
        return
    width, left, right = 760, 190, 36
    height = 85 + 66 * len(usable)
    maximum = max(float(row["p99"]) for row in usable) * 1.12
    scale = (width - left - right) / maximum
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.label{font-size:15px}.value{font-size:12px}.title{font-size:18px;font-weight:700}</style>',
        '<text class="title" x="18" y="26">Measured CUDA component latency (microseconds)</text>',
        '<rect x="410" y="14" width="13" height="13" fill="#1e4e79"/><text class="value" x="428" y="25">p50</text>',
        '<rect x="485" y="14" width="13" height="13" fill="#d9822b"/><text class="value" x="503" y="25">p99</text>',
    ]
    for index, row in enumerate(usable):
        y = 58 + index * 66
        p50, p99 = float(row["p50"]), float(row["p99"])
        parts += [
            f'<text class="label" text-anchor="end" x="{left - 10}" y="{y + 18}">{html.escape(short_label(row["backend"]))}</text>',
            f'<rect x="{left}" y="{y}" width="{p99 * scale:.2f}" height="18" rx="2" fill="#d9822b" opacity="0.78"/>',
            f'<rect x="{left}" y="{y + 22}" width="{p50 * scale:.2f}" height="18" rx="2" fill="#1e4e79"/>',
            f'<text class="value" x="{left + p99 * scale + 6:.2f}" y="{y + 14}">{p99:g}</text>',
            f'<text class="value" x="{left + p50 * scale + 6:.2f}" y="{y + 36}">{p50:g}</text>',
        ]
    parts.append('</svg>')
    (GENERATED / "latency_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def timeline_rows() -> list[dict[str, str]]:
    path = RESULTS / "nsys_timeline_2627824.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def timeline_svg(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    width, height, left = 820, 330, 160
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.title{font-size:18px;font-weight:700}.lane{font-size:14px}.small{font-size:11px}</style>',
        '<text class="title" x="18" y="26">Nsight Systems: measured host/device intervals</text>',
        '<text class="lane" x="18" y="57">CUDA Graph sample (0-85 us)</text>',
    ]
    graph = [row for row in rows if row["event"] in {"cudaGraphLaunch", "cudaGraphExec", "response_metadata_prep", "cudaStreamSynchronize"}]
    graph_scale = 6.8
    graph_y = {"CPU": 82, "GPU": 135}
    colors = {"cudaGraphLaunch": "#1e4e79", "cudaGraphExec": "#4f9d69", "response_metadata_prep": "#d9822b", "cudaStreamSynchronize": "#75808a"}
    for lane, y in graph_y.items():
        parts.append(f'<text class="lane" text-anchor="end" x="{left - 10}" y="{y + 17}">{lane}</text>')
    for row in graph:
        start, end = float(row["start_us"]), float(row["end_us"])
        y = graph_y[row["lane"]]
        x, bar_width = left + start * graph_scale, max(1.5, (end - start) * graph_scale)
        parts.append(f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="24" rx="2" fill="{colors[row["event"]]}"/>')
        if bar_width > 55:
            parts.append(f'<text class="small" fill="white" x="{x + 5:.2f}" y="{y + 16}">{html.escape(row["event"])}</text>')
    parts += [
        f'<line x1="{left + 28.739 * graph_scale:.2f}" y1="76" x2="{left + 28.739 * graph_scale:.2f}" y2="165" stroke="#d9822b" stroke-width="2"/>',
        f'<text class="small" x="{left + 28.739 * graph_scale + 5:.2f}" y="172">0.340 us host preparation overlaps graph interval</text>',
        '<text class="lane" x="18" y="218">Direct async-copy sample (0-11 us)</text>',
    ]
    direct = [row for row in rows if row["event"] in {"cudaMemcpyAsync_H2D_api", "H2D_128_bytes", "cudaLaunchKernel_api"}]
    direct_scale = 48
    direct_y = {"CPU": 238, "GPU": 283}
    for lane, y in direct_y.items():
        parts.append(f'<text class="lane" text-anchor="end" x="{left - 10}" y="{y + 17}">{lane}</text>')
    for row in direct:
        start, end = float(row["start_us"]), float(row["end_us"])
        y = direct_y[row["lane"]]
        x, bar_width = left + start * direct_scale, (end - start) * direct_scale
        color = "#1e4e79" if row["lane"] == "CPU" else "#4f9d69"
        parts.append(f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="23" rx="2" fill="{color}"/>')
        parts.append(f'<text class="small" x="{x + 3:.2f}" y="{y - 4}">{html.escape(row["event"])}</text>')
    parts.append('</svg>')
    (GENERATED / "nsys_timeline.svg").write_text("\n".join(parts), encoding="utf-8")


def result_figure_tex(rows: list[dict[str, Any]]) -> str:
    component = [row for row in rows if "CPU wall time per warmed" in str(row.get("scope")) and isinstance(row.get("p99"), (int, float))]
    if not component:
        return r"\fbox{\parbox{0.92\columnwidth}{\centering No comparable measured latency rows.}}"
    maximum = max(float(row["p99"]) for row in component) * 1.12
    xscale = 4.4 / maximum
    lines = [r"\begin{tikzpicture}[font=\scriptsize]", f"\\begin{{scope}}[x={xscale:.5f}cm,y=.72cm,xshift=1.25cm]", f"\\draw[->] (0,0)--({maximum:.3f},0) node[right]{{$\\mu$s}};"]
    for index, row in enumerate(component, start=1):
        y = len(component) - index + 1
        p50, p99 = float(row["p50"]), float(row["p99"])
        label = latex(short_label(row["backend"]))
        lines += [
            f"\\node[anchor=east] at (0,{y}) {{{label}}};",
            f"\\fill[orange!70] (0,{y - .20}) rectangle ({p99},{y + .20});",
            f"\\fill[accent] (0,{y - .12}) rectangle ({p50},{y + .12});",
            f"\\node[anchor=west] at ({p99},{y}) {{p99 {p99:g}}};",
        ]
    lines += [r"\end{scope}", r"\node[anchor=west] at (0,-.45) {\textcolor{accent}{p50 overlay} / \textcolor{orange!80!black}{p99}};", r"\end{tikzpicture}"]
    return "".join(lines)


def nsight_figure_tex(rows: list[dict[str, str]]) -> str:
    if not rows:
        return r"\fbox{\parbox{0.94\textwidth}{\centering Nsight timeline data unavailable.}}"
    return r"""\begin{tikzpicture}[font=\scriptsize,x=.072cm,y=.62cm]
\node[anchor=east] at (0,2) {CPU}; \node[anchor=east] at (0,1) {GPU};
\fill[accent] (0,1.8) rectangle (28.739,2.2);
\node[white] at (14.37,2) {graph launch API};
\fill[green!55!black] (8.987,.8) rectangle (69.467,1.2);
\node[white] at (39.2,1) {CUDA Graph interval};
\draw[orange,line width=1.2pt] (28.739,1.65)--(28.739,2.35);
\node[anchor=west,orange!80!black] at (29.5,2.45) {0.340 $\mu$s host prep overlap};
\fill[gray!70] (29.079,1.8) rectangle (84.988,2.2);
\node[white] at (57.0,2) {stream synchronize};
\draw[->] (0,.3)--(86,.3) node[right] {$\mu$s};
\foreach \x in {0,20,40,60,80}{\draw (\x,.2)--(\x,.4) node[below] {\x};}
\end{tikzpicture}"""


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    manifests = completed_manifests()
    rows = result_rows(manifests)
    csvs = quantile_csvs()
    timeline = timeline_rows()
    latency_svg(rows)
    timeline_svg(timeline)
    lines = ["% Generated by scripts/generate_research_figures.py; do not edit."]
    if not rows:
        lines += [
            r"\newcommand{\ResultEvidenceStatus}{\textbf{Not measured.} No completed JSON benchmark manifest was found under \texttt{docs/profiling/results/}.}",
            r"\newcommand{\ResultEvidenceFigure}{\fbox{\parbox{0.92\columnwidth}{\centering No machine-readable result is checked in.}}}",
        ]
    else:
        lines.append(f"\\newcommand{{\\ResultEvidenceStatus}}{{{len(rows)} completed measured configuration(s) were generated from checked-in manifests; scope labels below are binding.}}")
        lines.append("\\newcommand{\\ResultEvidenceFigure}{" + result_figure_tex(rows) + "}")
        lines += [
            r"\begin{table*}[t]", r"\centering",
            r"\caption{Measured summaries generated from checked-in JSON manifests.}",
            r"\label{tab:generated-results}", r"\scriptsize",
            r"\begin{tabularx}{\textwidth}{@{}lllXrrrr@{}}", r"\toprule",
            "Artifact & Commit & Backend & Scope & p50 ($\\mu$s) & p99 ($\\mu$s) & Rate/s & Errors \\\\",
            r"\midrule",
        ]
        for row in rows:
            scope = str(row["scope"])
            if len(scope) > 58:
                scope = scope[:55] + "..."
            error = row["error_rate"]
            error_text = "--" if not isinstance(error, (int, float)) else f"{100 * float(error):.2f}%"
            values = [artifact_label(row["artifact"]), str(row["commit"])[:8],
                      short_label(row["backend"]), scope,
                      compact_number(row["p50"]), compact_number(row["p99"]),
                      compact_number(row["throughput"], 2), error_text]
            lines.append(" & ".join(latex(value) for value in values) + " \\\\")
        lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    try:
        nsys = json.loads((RESULTS / "nsys_summary_2627824.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        nsys = None
    if nsys:
        overlap = nsys["overlap_probe_last_500"]
        lines.append(f"\\newcommand{{\\NsightEvidenceStatus}}{{Nsight Systems observed positive host/GPU overlap in {overlap['positive_overlap_count']} of 500 bounded probes; median overlap was {overlap['overlap_ns']['p50']} ns.}}")
    else:
        lines.append(r"\newcommand{\NsightEvidenceStatus}{Nsight Systems evidence is unavailable.}")
    lines.append("\\newcommand{\\NsightEvidenceFigure}{" + nsight_figure_tex(timeline) + "}")
    if csvs:
        lines.append("% CSV artifacts detected: " + ", ".join(path.name for path in csvs))
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
