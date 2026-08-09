#!/usr/bin/env python3
import csv
import datetime
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys


def json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    try:
        return json_value(dict(value))
    except Exception:
        return str(value)


def load_ncu_report():
    candidates = []
    root = os.environ.get("NSIGHT_COMPUTE_ROOT")
    if root:
        candidates.append(pathlib.Path(root) / "extras" / "python")
    ncu = shutil.which(os.environ.get("HYDRO_NCU", "ncu"))
    if ncu:
        candidates.append(pathlib.Path(ncu).resolve().parent / "extras" / "python")
    candidates.extend(pathlib.Path("/opt/nvidia/nsight-compute").glob("*/extras/python"))
    candidates.extend(pathlib.Path("/usr/local/cuda").glob("nsight-compute-*/extras/python"))
    for candidate in candidates:
        if (candidate / "ncu_report.py").is_file():
            sys.path.insert(0, str(candidate))
            break
    import ncu_report  # pylint: disable=import-outside-toplevel
    return ncu_report


def read_text(path, limit=8 * 1024 * 1024):
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[output truncated by Hydro]\n"


def parse_details(path):
    text = read_text(path)
    if not text.strip():
        return []
    rows = list(csv.DictReader(text.splitlines()))
    sections = {}
    for row in rows:
        action_id = row.get("ID", "")
        kernel_name = row.get("Kernel Name", "")
        section_name = row.get("Section Name", "") or "Other"
        key = (action_id, kernel_name, section_name)
        section = sections.setdefault(key, {
            "actionId": action_id,
            "kernelName": kernel_name,
            "sectionName": section_name,
            "items": [],
            "rules": [],
        })
        metric_label = row.get("Metric Name", "")
        if metric_label:
            match = re.match(r"^(.*) \(([^()]+)\)$", metric_label)
            section["items"].append({
                "bodyItem": row.get("Body Item Label", ""),
                "label": match.group(1) if match else metric_label,
                "name": match.group(2) if match else "",
                "unit": row.get("Metric Unit", ""),
                "value": row.get("Metric Value", ""),
            })
        if row.get("Rule Name"):
            section["rules"].append({
                "name": row.get("Rule Name", ""),
                "type": row.get("Rule Type", ""),
                "description": row.get("Rule Description", ""),
                "estimatedSpeedupType": row.get("Estimated Speedup Type", ""),
                "estimatedSpeedup": row.get("Estimated Speedup", ""),
            })
    return list(sections.values())


def metric_value(metric, index=None):
    try:
        value = metric.value() if index is None else metric.value(index)
        return json_value(value)
    except Exception as error:
        return {"error": str(error)}


def enum_value(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return json_value(value)


def named_metric(action, name):
    metric = action.metric_by_name(name)
    return metric_value(metric) if metric is not None else None


def action_summary(action, index, instance_budget, warnings):
    metrics = []
    used_instances = 0
    for name in action.metric_names():
        metric = action.metric_by_name(name)
        count = max(0, int(metric.num_instances()))
        remaining = max(0, instance_budget - used_instances)
        keep = min(count, remaining)
        instances = []
        correlations = metric.correlation_ids() if keep and metric.has_correlation_ids() else None
        for instance_id in range(keep):
            instance = {"index": instance_id, "value": metric_value(metric, instance_id)}
            if correlations is not None:
                instance["correlationId"] = metric_value(correlations, instance_id)
            instances.append(instance)
        used_instances += keep
        metrics.append({
            "name": name,
            "unit": metric.unit(),
            "description": metric.description(),
            "value": metric_value(metric),
            "kind": enum_value(metric.kind()),
            "metricType": enum_value(metric.metric_type()),
            "metricSubtype": enum_value(metric.metric_subtype()),
            "rollupOperation": enum_value(metric.rollup_operation()),
            "instanceCount": count,
            "instances": instances,
            "instancesTruncated": keep < count,
        })
    if used_instances >= instance_budget and any(item["instancesTruncated"] for item in metrics):
        warnings.append(
            f"Raw metric instances for action {index} were limited to {instance_budget} values; "
            "the .ncu-rep file retains the complete data."
        )
    source_files = []
    try:
        source_files = [
            {"path": name, "content": content}
            for name, content in dict(action.source_files()).items()
        ]
    except Exception as error:
        warnings.append(f"Unable to extract imported source files for action {index}: {error}")
    return {
        "index": index,
        "name": action.name(),
        "workloadType": enum_value(action.workload_type()),
        "device": named_metric(action, "device__attribute_display_name"),
        "computeCapability": [
            named_metric(action, "device__attribute_compute_capability_major"),
            named_metric(action, "device__attribute_compute_capability_minor"),
        ],
        "grid": [
            named_metric(action, "launch__grid_dim_x"),
            named_metric(action, "launch__grid_dim_y"),
            named_metric(action, "launch__grid_dim_z"),
        ],
        "block": [
            named_metric(action, "launch__block_dim_x"),
            named_metric(action, "launch__block_dim_y"),
            named_metric(action, "launch__block_dim_z"),
        ],
        "achievedOccupancyPct": named_metric(action, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "theoreticalOccupancyPct": named_metric(action, "sm__maximum_warps_per_active_cycle_pct"),
        "nvtxState": json_value(action.nvtx_state()),
        "rules": json_value(action.rule_results_as_dicts()),
        "sourceMarkers": json_value(action.source_markers()),
        "sourceFiles": source_files,
        "metricCount": len(metrics),
        "metrics": metrics,
    }, used_instances


def ncu_version():
    ncu = os.environ.get("HYDRO_NCU", "ncu")
    output = subprocess.run(
        [ncu, "--version"], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
    match = re.search(r"Version\s+([^\s]+)", output)
    return match.group(1) if match else output.strip().splitlines()[-1]


def main():
    if len(sys.argv) != 9:
        raise SystemExit(
            "usage: profile_summary.py REPORT DETAILS SESSION SOURCE OUTPUT SET MEASURE_RUN NVTX_RANGE"
        )
    report_path, details_path, session_path, source_path, output_path = sys.argv[1:6]
    section_set, measure_run, nvtx_range = sys.argv[6:9]
    max_instances = int(os.environ.get("HYDRO_PROFILE_MAX_INSTANCES", "100000"))
    ncu_report = load_ncu_report()
    report = ncu_report.load_report(report_path)
    warnings = []
    remaining_instances = max_instances
    ranges = []
    for range_index in range(report.num_ranges()):
        report_range = report.range_by_idx(range_index)
        actions = []
        for action_index in range(report_range.num_actions()):
            action, used = action_summary(
                report_range.action_by_idx(action_index), action_index,
                remaining_instances, warnings,
            )
            remaining_instances = max(0, remaining_instances - used)
            actions.append(action)
        ranges.append({"index": range_index, "actions": actions})

    summary = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ncuVersion": ncu_version(),
        "set": section_set,
        "measureRun": int(measure_run),
        "nvtxRange": nvtx_range,
        "rangeCount": len(ranges),
        "actionCount": sum(len(item["actions"]) for item in ranges),
        "ranges": ranges,
        "detailSections": parse_details(details_path),
        "sessionOutput": read_text(session_path),
        "sourceOutput": read_text(source_path),
        "warnings": warnings,
    }
    pathlib.Path(output_path).write_text(
        json.dumps(summary, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
