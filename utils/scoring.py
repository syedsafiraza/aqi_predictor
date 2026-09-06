def clean_air_score(aqi, momentum=0.0):
    if aqi is None:
        return None
    base = 100 - (aqi / 3)
    trend_adjustment = -(momentum or 0) / 5
    score = base + trend_adjustment
    return round(max(0, min(100, score)), 1)


def score_label(score):
    if score is None:
        return "Unknown"
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Fair"
    if score >= 20:
        return "Poor"
    return "Critical"
