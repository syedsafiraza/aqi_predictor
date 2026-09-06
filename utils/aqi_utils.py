import config

AQI_BANDS = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory."),
    (51, 100, "Moderate", "#ffff00", "Acceptable; sensitive individuals should take it easy outdoors."),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00",
     "Children, elderly, and people with heart/lung conditions should limit prolonged outdoor exertion."),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may feel effects; sensitive groups should avoid outdoor exertion."),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert — avoid outdoor activity."),
    (301, 500, "Hazardous", "#7e0023", "Health emergency — stay indoors, use a mask/purifier if you must go out."),
]


def aqi_category(aqi):
    if aqi is None:
        return "Unknown", "#9e9e9e", "No data available."
    aqi = max(0, aqi)
    for lo, hi, label, color, advice in AQI_BANDS:
        if lo <= aqi <= hi:
            return label, color, advice
    return "Hazardous", "#7e0023", AQI_BANDS[-1][4]


def is_hazardous(aqi, threshold=config.HAZARDOUS_AQI_THRESHOLD):
    return aqi is not None and aqi >= threshold


def is_smog_season(month, province):
    return month in config.SMOG_SEASON_MONTHS and province in config.SMOG_SEASON_PROVINCES
