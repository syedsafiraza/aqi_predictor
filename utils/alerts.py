import logging

import requests

import config

log = logging.getLogger("alerts")


def send_alert(message):
    log.warning(message)
    if not config.ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(config.ALERT_WEBHOOK_URL, json={"text": message, "content": message}, timeout=10)
    except Exception as e:
        log.error(f"Alert webhook failed: {e}")
