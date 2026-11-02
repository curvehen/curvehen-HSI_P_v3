"""Send the daily report by email: HTML body with inline charts (cid),
plain-text fallback, and the self-contained HTML file attached."""
import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import List

from src import config
from src.reporting import report

log = logging.getLogger(__name__)


def _recipients() -> List[str]:
    return [a.strip() for a in config.EMAIL_TO.replace(";", ",").split(",") if a.strip()]


def build_message(rep: dict) -> EmailMessage:
    to = _recipients()
    msg = EmailMessage()
    msg["Subject"] = rep["subject"]
    msg["From"] = config.SMTP_USER
    msg["To"] = ", ".join(to)
    msg.set_content(rep["text"])
    msg.add_alternative(rep["html_email"], subtype="html")

    html_part = msg.get_payload()[1]
    for cid, png in rep["images"].items():
        html_part.add_related(Path(png).read_bytes(), maintype="image",
                              subtype="png", cid=f"<{cid}>")

    path = Path(rep["path"])
    msg.add_attachment(path.read_bytes(), maintype="text", subtype="html",
                       filename=path.name)
    return msg


def send(rep: dict) -> bool:
    """Returns True if sent. Missing settings or SMTP errors are logged, not raised,
    so the rest of the daily job still completes."""
    missing = [k for k in ("SMTP_USER", "SMTP_PASS", "EMAIL_TO") if not getattr(config, k)]
    if missing:
        log.warning("Email skipped: missing %s", ", ".join(missing))
        return False
    try:
        msg = build_message(rep)
        if config.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=60) as s:
                s.login(config.SMTP_USER, config.SMTP_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=60) as s:
                s.starttls()
                s.login(config.SMTP_USER, config.SMTP_PASS)
                s.send_message(msg)
        log.info("Email sent to %s", msg["To"])
        return True
    except Exception as exc:
        log.warning("Email failed: %s: %s", type(exc).__name__, exc)
        return False


def main() -> None:
    rep = report.build_report()
    print("sent" if send(rep) else "not sent (report saved only)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
