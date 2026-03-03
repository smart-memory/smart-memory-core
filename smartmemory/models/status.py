import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional


@dataclass
class Status:
    status: str
    notes: Optional[str] = None
    updated: datetime = field(default_factory=datetime.now)
    # Optionally: who, cause, etc.


class StatusLoggerMixin:
    def update_status(self, status: str, notes: str = None):
        from datetime import datetime
        # Always operate directly on self.metadata['status_history'] as JSON
        history_json = self.metadata.get('status_history', '[]')
        try:
            history = json.loads(history_json)
        except Exception:
            history = []
        now = datetime.now(UTC).isoformat()
        entry = {
            'status': status,
            'timestamp': now,
            'notes': notes or f'Status updated to {status}'
        }
        history.insert(0, entry)
        # Ensure metadata exists
        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}
        self.metadata['status'] = status
        self.metadata['updated_at'] = now
        self.metadata['status_history'] = json.dumps(history)

    def get_status_history(self):
        history_json = self.metadata.get('status_history', '[]')
        try:
            return json.loads(history_json)
        except Exception:
            return []

    def status_criteria(self, status: str):
        return {"status_history.0.status": str(status)}
