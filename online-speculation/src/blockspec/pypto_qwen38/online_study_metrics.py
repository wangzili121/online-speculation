"""已核对完整对照记录的双基线判定；数据隔离仍须另行审核。"""

from math import isfinite


def matched_request_timelines(count, window, *timelines):
    """三个方法必须覆盖相同请求和提交窗口，不能把排队窗口误当执行并发。"""
    if (type(count) is not int or count < 2 or count % 2
            or type(window) is not int or window not in (2, 4, 8, 16, 32) or len(timelines) != 3):
        return False
    expected = [list(range(i + 1, min(i + window, count) + 1)) for i in range(0, count, window)]
    for rows in timelines:
        if not isinstance(rows, list) or len(rows) != len(expected):
            return False
        for row, ids in zip(rows, expected):
            if (not isinstance(row, dict) or not isinstance(row.get("requests"), list)
                    or any(type(i) is not int for i in row["requests"]) or row["requests"] != ids):
                return False
            lengths = row.get("output_tokens")
            seconds = row.get("seconds")
            if (not isinstance(lengths, list) or len(lengths) != len(ids)
                    or any(type(n) is not int or n < 1 for n in lengths)
                    or type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0):
                return False
    return True


def confirmation_metrics(values):
    """三次与合并吞吐必须同时超过 AR、初始固定起草各 5%。"""
    result = dict(repeats=len(values), primary_metric="continuous_online_vs_AR_and_initial_frozen",
                  target_met=False)
    if not values:
        return result
    fields = tuple(f"{arm}_{field}" for arm in ("ar", "frozen", "online")
                   for field in ("tokens", "seconds"))
    for value in values:
        if any(type(value.get(key)) not in (int, float) or not isfinite(value[key]) or value[key] <= 0
               for key in fields):
            return {**result, "reason": "missing or invalid full-stream counts/timing"}
    rates = [{arm: value[f"{arm}_tokens"] / value[f"{arm}_seconds"]
              for arm in ("ar", "frozen", "online")} for value in values]
    pooled = {arm: sum(v[f"{arm}_tokens"] for v in values) /
                   sum(v[f"{arm}_seconds"] for v in values) for arm in ("ar", "frozen", "online")}
    gains = [row["online"] / row["ar"] - 1 for row in rates]
    fixed_gains = [row["online"] / row["frozen"] - 1 for row in rates]
    pooled_ar = pooled["online"] / pooled["ar"] - 1
    pooled_fixed = pooled["online"] / pooled["frozen"] - 1
    eligible = all(all(v.get(key) is True for key in
                       ("continuous", "full_cohort", "resource_policy_ok", "performance_run"))
                   for v in values)
    # 极小余量仅防止浮点舍入把恰好 5% 判成超过 5%，不是统计置信度。
    threshold = .05 + 1e-12
    return {**result, "eligible": eligible, "pooled_ar_tok_s": pooled["ar"],
            "pooled_frozen_tok_s": pooled["frozen"], "pooled_online_tok_s": pooled["online"],
            "pooled_gain": pooled_ar, "pooled_gain_vs_frozen": pooled_fixed,
            "individual_gains": gains, "individual_gains_vs_frozen": fixed_gains,
            "target_met": len(values) == 3 and eligible and pooled_ar > threshold
                          and pooled_fixed > threshold and all(g > threshold for g in gains + fixed_gains)}
