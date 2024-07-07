import enum
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

JSON_DATABASE_FILE = Path('./parking_lots.json')


class ParkingLotState(enum.StrEnum):
    PENDING_SETUP_ACTIVE_ZONE = 'PENDING_SETUP_ACTIVE_ZONE'
    PENDING_SETUP_PARKING_SPACES = 'PENDING_SETUP_PARKING_SPACES'
    READY = 'READY'


class InvalidStateError(Exception):
    ...


@dataclass(frozen=True, kw_only=True)
class ParkingSpace:
    number: int
    position: tuple[float, float]
    radius: float

    def to_dict(self) -> dict:
        return {
            'number': self.number,
            'position': self.position,
            'radius': self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ParkingSpace':
        return cls(
            number=data['number'],
            position=cast(tuple[int, int], tuple(data['position'])),
            radius=data['radius'],
        )


class ParkingLot:
    def __init__(
            self,
            id: uuid.UUID,
            name: str,
            state: ParkingLotState,
            active_zone_video: str | None,
            active_zone: list[tuple[int, int]],
            spaces: list[ParkingSpace],
    ) -> None:
        self._id = id
        self._name = name
        self._state = state
        self._active_zone_video = active_zone_video
        self._active_zone = active_zone
        self._spaces = spaces

    @property
    def id(self) -> uuid.UUID:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> ParkingLotState:
        return self._state

    def set_active_zone_video(self, video_path: str | os.PathLike) -> None:
        if self._state != ParkingLotState.PENDING_SETUP_ACTIVE_ZONE:
            raise InvalidStateError
        self._active_zone_video = video_path

    @property
    def active_zone_video(self) -> str | None:
        return self._active_zone_video

    def set_active_zone(self, points: list[tuple[float, float]]) -> None:
        if self._state != ParkingLotState.PENDING_SETUP_ACTIVE_ZONE:
            raise InvalidStateError
        self._active_zone = points
        self._state = ParkingLotState.PENDING_SETUP_PARKING_SPACES

    @property
    def active_zone(self) -> list[tuple[int, int]]:
        return self._active_zone

    def set_parking_spaces(self, spaces: list[ParkingSpace]) -> None:
        if self._state != ParkingLotState.PENDING_SETUP_PARKING_SPACES:
            raise InvalidStateError
        self._spaces = spaces
        self._state = ParkingLotState.READY

    @property
    def parking_spaces(self) -> list[ParkingSpace]:
        return self._spaces

    def to_dict(self) -> dict:
        return {
            'id': str(self._id),
            'name': self._name,
            'state': self._state.value,
            'active_zone_video': self._active_zone_video,
            'active_zone': self._active_zone,
            'spaces': [space.to_dict() for space in self._spaces],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ParkingLot':
        return cls(
            id=uuid.UUID(data['id']),
            name=data['name'],
            state=ParkingLotState(data['state']),
            active_zone_video=data['active_zone_video'],
            active_zone=data['active_zone'],
            spaces=[ParkingSpace.from_dict(space_data) for space_data in data['spaces']],
        )


class ParkingLotRepository:
    def __init__(self) -> None:
        self._lots: dict[uuid.UUID, ParkingLot] = {
            uuid.UUID('802b546a-2653-4c1f-b912-821dfdbf7f3d'): ParkingLot(
                id=uuid.UUID('802b546a-2653-4c1f-b912-821dfdbf7f3d'),
                name='Северный пролёт гусей',
                active_zone_video='',
                state=ParkingLotState.PENDING_SETUP_ACTIVE_ZONE,
                active_zone=[],
                spaces=[],
            ),
        }
        self._try_load_from_file()

    def _save_to_file(self) -> None:
        with JSON_DATABASE_FILE.open(mode='w', encoding='utf-8') as file:
            json.dump(
                [lot.to_dict() for lot in self._lots.values()],
                file,
                ensure_ascii=False,
                indent=4,
            )

    def commit(self) -> None:
        self._save_to_file()

    def _try_load_from_file(self) -> None:
        if JSON_DATABASE_FILE.exists():
            with JSON_DATABASE_FILE.open(encoding='utf-8') as file:
                data = json.load(file)
                self._lots = {
                    uuid.UUID(lot['id']): ParkingLot.from_dict(lot) for lot in data
                }

    def add_parking_lot(self, id: uuid.UUID, name: str) -> None:
        self._lots[id] = ParkingLot(
            id=id,
            name=name,
            state=ParkingLotState.PENDING_SETUP_ACTIVE_ZONE,
            active_zone_video=None,
            active_zone=[],
            spaces=[],
        )

    def list_parking_lots(self) -> list[ParkingLot]:
        return list(self._lots.values())

    def get_parking_lot(self, id: uuid.UUID) -> ParkingLot | None:
        return self._lots.get(id, None)
