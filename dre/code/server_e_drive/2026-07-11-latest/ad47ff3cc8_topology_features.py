from __future__ import annotations

import re


_CONTACT = re.compile(r"^([A-Za-z]+)[-_ ]?0*(\d+)(?:REF|CAR|AVG)?$", re.IGNORECASE)


def parse_channel_topology(channel_name: str) -> tuple[str | None, int | None, bool]:
    match = _CONTACT.fullmatch(str(channel_name).strip())
    if match is None:
        return None, None, False
    return match.group(1).upper(), int(match.group(2)), True
