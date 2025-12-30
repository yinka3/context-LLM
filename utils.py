from datetime import datetime

def format_relative_time(now: datetime, ts: datetime) -> str:
    delta = now - ts
    seconds = delta.total_seconds()
    
    if seconds < 3600:
        mins = int(seconds // 60)
        return f"{mins}m ago" if mins > 1 else "just now"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours}h ago"
    elif seconds < 604800:
        days = int(seconds // 86400)
        return f"{days}d ago"
    else:
        weeks = int(seconds // 604800)
        return f"{weeks}w ago"