import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path


JSON_DATABASE_FILE = Path('test_drives.json')


@dataclass(frozen=True, kw_only=True)
class TestDrive:
    id: uuid.UUID
    lot_id: uuid.UUID
    space_number: int
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> dict:
        return {
            'id': str(self.id),
            'lot_id': self.lot_id,
            'space_number': self.space_number,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TestDrive':
        return cls(
            id=uuid.UUID(data['id']),
            lot_id=uuid.UUID(data['lot_id']),
            space_number=data['space_id'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
        )


class TestDriveRegistry:
    def __init__(self) -> None:
        self._test_drives: dict[uuid.UUID, TestDrive] = {}
        td_id = uuid.UUID('fe1da530-f829-4ddb-8712-6e76f36595cd')
        self._test_drives[td_id] = TestDrive(
            id=td_id,
            lot_id=uuid.UUID('802b546a-2653-4c1f-b912-821dfdbf7f3d'),
            space_number=2,
            start_time=datetime.now() - timedelta(seconds=214),
            end_time=datetime.now() + timedelta(days=1),
        )
        # self._try_load_from_file()  # Fixme: Uncomment this line when you're ready to load data from a file

    def _try_load_from_file(self) -> None:
        if JSON_DATABASE_FILE.exists():
            with JSON_DATABASE_FILE.open(mode='r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    test_drive = TestDrive.from_dict(item)
                    self._test_drives[test_drive.id] = test_drive

    def _save_to_file(self) -> None:
        with JSON_DATABASE_FILE.open(mode='w', encoding='utf-8') as file:
            file.write(json.dumps([test_drive.to_dict() for test_drive in self._test_drives.values()]))

    def commit(self) -> None:
        self._save_to_file()

    def add_test_drive(self, test_drive: TestDrive) -> None:
        self._test_drives[test_drive.id] = test_drive

    def list_all_test_drives(self) -> list[TestDrive]:
        return list(self._test_drives.values())

    def get_test_drive(self, test_drive_id: uuid.UUID) -> TestDrive | None:
        return self._test_drives.get(test_drive_id)
