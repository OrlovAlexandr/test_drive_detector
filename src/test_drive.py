import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


JSON_DATABASE_FILE = Path('./test_drives.json')


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
            'lot_id': str(self.lot_id),
            'space_number': self.space_number,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TestDrive':
        return cls(
            id=uuid.UUID(data['id']),
            lot_id=uuid.UUID(data['lot_id']),
            space_number=data['space_number'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
        )


class TestDriveRegistry:
    def __init__(self) -> None:
        self._test_drives: dict[uuid.UUID, TestDrive] = {}
        self._try_load_from_file()

    def _try_load_from_file(self) -> None:
        if JSON_DATABASE_FILE.exists():
            with JSON_DATABASE_FILE.open(mode='rt', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    test_drive = TestDrive.from_dict(item)
                    self._test_drives[test_drive.id] = test_drive

    def _save_to_file(self) -> None:
        with JSON_DATABASE_FILE.open(mode='wt', encoding='utf-8') as file:
            json.dump(
                [test_drive.to_dict() for test_drive in self._test_drives.values()],
                file,
                ensure_ascii=False,
                indent=4,
            )

    def commit(self) -> None:
        self._save_to_file()

    def add_test_drive(self, test_drive: TestDrive) -> None:
        self._test_drives[test_drive.id] = test_drive

    def list_all_test_drives(self) -> list[TestDrive]:
        return list(self._test_drives.values())

    def get_test_drive(self, test_drive_id: uuid.UUID) -> TestDrive | None:
        return self._test_drives.get(test_drive_id)


def test_test_drives_save():
    registry = TestDriveRegistry()
    test_drive = TestDrive(
        id=uuid.uuid4(),
        lot_id=uuid.uuid4(),
        space_number=1,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    registry.add_test_drive(test_drive)
    registry.commit()
    newly_created_test_drive_registry = TestDriveRegistry()
    loaded_test_drive = newly_created_test_drive_registry.get_test_drive(test_drive.id)
    assert loaded_test_drive == test_drive  # noqa: S101
