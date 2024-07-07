import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from nicegui import app
from nicegui import events as nicegui_events
from nicegui import ui
from starlette.responses import RedirectResponse

from src import detect_test_drives
from src import image_ops
from src import video_ops
from src.parking_lot import ParkingLot
from src.parking_lot import ParkingLotRepository
from src.parking_lot import ParkingLotState
from src.parking_lot import ParkingSpace
from src.parking_spaces import detect_parking_spaces
from src.test_drive import TestDriveRegistry


INBOX_DIRECTORY = Path('./inbox')
OUTPUT_DIRECTORY = Path('./output')

HUMAN_READABLE_DATE_TIME_FORMAT = '%d.%m.%Y %H:%M:%S'
HUMAN_READABLE_DATE_FORMAT = '%d.%m.%Y'

MINIMUM_POINT_COUNT = 3

FIRST_FRAME_WIDTH = 640
FIRST_FRAME_HEIGHT = 360


@ui.refreshable
def parking_lot_list(parking_lot_repository: ParkingLotRepository) -> None:
    def show_parking_lot_setup(parking_lot: ParkingLot) -> None:
        parking_lot_setup_area.refresh(parking_lot=parking_lot, parking_lot_repository=parking_lot_repository)

    for lot in parking_lot_repository.list_parking_lots():
        with ui.item(on_click=lambda _lot=lot: show_parking_lot_setup(_lot)):
            with ui.item_section().props('avatar'):
                ui.icon(name='camera')
            with ui.item_section():
                ui.item_label(text=lot.name)


@ui.refreshable
def lot_spaces_list(spaces: list[ParkingSpace]) -> None:
    with ui.list():
        ui.label(text='Места')
        ui.separator()
        if not spaces:
            ui.label(text='Список пуст')
        for space in spaces:
            with ui.item():
                with ui.item_section().props('avatar'):
                    ui.icon('spot')
                with ui.item_section():
                    ui.label(text=f'Место #{space.number}')


@ui.refreshable
def parking_zone_preview(
        points: list[tuple[float, float]],
        on_click: Callable,
        image: bytes | None,
) -> None:
    if image is None:
        return
    image_with_points = image_ops.draw_parking_lot_map(
        image_data=image,
        points=points,
    )
    ui.interactive_image(
        source=image_ops.bytes_to_b64_embedded_image(image_with_points),
        on_mouse=on_click,
        events=['mousedown'],
        cross=True,
    )


@ui.refreshable
def parking_lot_setup_area(
        parking_lot: ParkingLot | None,
        parking_lot_repository: ParkingLotRepository | None,
) -> None:
    if parking_lot is not None:
        ui.label(text=parking_lot.name).classes('text-2xl')
    if parking_lot is not None and parking_lot.state != ParkingLotState.READY:
        with ui.stepper().classes('w-full') as stepper:
            with (ui.step('Разметка парковочной зоны').classes('w-full'),
                  ui.stepper().props('vertical').classes('w-full no-shadow border-[0px] p-0') as inner_stepper):
                with ui.step('Загрузка видео'):
                    ui.label('Загрузите короткое (2-3 секунды) видео (mp4), когда все машины на своих местах.')

                    def on_video_uploaded(event: nicegui_events.UploadEventArguments) -> None:
                        video_file_path = INBOX_DIRECTORY / f'{uuid.uuid4()}.mp4'
                        with video_file_path.open(mode='wb') as file:
                            file.write(event.content.read())
                        parking_lot.set_active_zone_video(video_file_path.name)
                        parking_lot_repository.commit()
                        ui.notify('Видео успешно загружено', position='top-right', type='positive')
                        parking_lot_setup_area.refresh(parking_lot=parking_lot,
                                                       parking_lot_repository=parking_lot_repository)

                    ui.upload(
                        on_upload=on_video_uploaded,
                        auto_upload=True,
                    )
                with ui.step('Разметка'):
                    if parking_lot.active_zone_video:
                        video = video_ops.VideoFile(INBOX_DIRECTORY / parking_lot.active_zone_video,
                                                    FIRST_FRAME_WIDTH, FIRST_FRAME_HEIGHT)
                        first_frame = video.get_first_frame()
                    else:
                        first_frame = None
                    points: list[tuple[float, float]] = []

                    def add_point(event: nicegui_events.MouseEventArguments):
                        point_x = int(event.image_x) / FIRST_FRAME_WIDTH
                        point_y = int(event.image_y) / FIRST_FRAME_HEIGHT
                        points.append((point_x, point_y))
                        parking_zone_preview.refresh(points=points, on_click=add_point, image=first_frame)

                    def delete_all_points() -> None:
                        if not points:
                            return
                        points.clear()
                        parking_zone_preview.refresh(points=points, on_click=add_point, image=first_frame)

                    def delete_last_point() -> None:
                        if not points:
                            return
                        points.pop()
                        parking_zone_preview.refresh(points=points, on_click=add_point, image=first_frame)

                    ui.label(
                        text='Обведите парковочную зону, последовательно устанавливая точки. '
                             'Постарайтесь обвести так, чтобы границы не пересекали очертания автомобилей.')
                    with ui.row():
                        ui.button(text='Очистить', on_click=delete_all_points)
                        ui.button(text='Удалить последнюю точку', on_click=delete_last_point)

                    parking_zone_preview(points=points, on_click=add_point, image=first_frame)
                    with ui.stepper_navigation():
                        def on_confirm() -> None:
                            if len(points) < MINIMUM_POINT_COUNT:
                                ui.notify(
                                    f'Минимальное количество точек: {MINIMUM_POINT_COUNT}',
                                    position='top-right',
                                    type='negative',
                                )
                                return
                            parking_lot.set_active_zone(points)
                            parking_lot_repository.commit()
                            parking_lot_setup_area.refresh(parking_lot=parking_lot,
                                                           parking_lot_repository=parking_lot_repository)

                        ui.button(text='Перейти к разметке мест', on_click=on_confirm)

            with ui.step('Разметка мест'):
                spaces = []
                if parking_lot.active_zone_video:
                    video = video_ops.VideoFile(INBOX_DIRECTORY / parking_lot.active_zone_video,
                                                FIRST_FRAME_WIDTH, FIRST_FRAME_HEIGHT)
                    first_frame_2 = image_ops.bytes_to_b64_embedded_image(
                        image_ops.draw_parking_lot_map(
                            video.get_first_frame(),
                            parking_lot.active_zone,
                        ),
                    )
                else:
                    first_frame_2 = None
                ui.label(text='Выполните анализ для получения списка мест.')
                if first_frame_2:
                    ui.image(
                        source=first_frame_2,
                    )

                analyze_progress = ui.linear_progress(value=0, show_value=False)

                async def analyze() -> None:
                    analyze_button.disable()
                    analyze_progress.set_value(0.5)
                    found_spaces = detect_parking_spaces(
                        video_path=str(INBOX_DIRECTORY / parking_lot.active_zone_video),
                        parking_polygon=parking_lot.active_zone,
                    )
                    analyze_progress.set_value(1)
                    nonlocal spaces
                    spaces = found_spaces
                    lot_spaces_list.refresh(found_spaces)

                analyze_button = ui.button(text='Проанализировать', on_click=analyze)

                lot_spaces_list([])

                with ui.stepper_navigation():
                    def on_setup_spaces_confirm() -> None:
                        if not spaces:
                            ui.notify('Места не размечены', position='top-right', type='negative')
                            return
                        parking_lot.set_parking_spaces(spaces)
                        parking_lot_repository.commit()
                        ui.notify('Места успешно размечены', position='top-right', type='positive')
                        parking_lot_setup_area.refresh(parking_lot=parking_lot,
                                                       parking_lot_repository=parking_lot_repository)

                    ui.button('Завершить настройку', on_click=on_setup_spaces_confirm)

            # Deep linking
            if (parking_lot is not None
                    and parking_lot.state == ParkingLotState.PENDING_SETUP_ACTIVE_ZONE
                    and parking_lot.active_zone_video):
                inner_stepper.next()
            if parking_lot is not None and parking_lot.state == ParkingLotState.PENDING_SETUP_PARKING_SPACES:
                stepper.next()
    if parking_lot is None:
        ui.label('Выберите парковочную зону')
    if parking_lot is not None and parking_lot.state == ParkingLotState.READY:
        ui.label(text='Разметка выполнена')
        ui.label(text=f'Мест: {len(parking_lot.parking_spaces)}')
        video = video_ops.VideoFile(INBOX_DIRECTORY / parking_lot.active_zone_video,
                                    FIRST_FRAME_WIDTH, FIRST_FRAME_HEIGHT)
        if parking_lot.active_zone_video:
            first_frame_3 = image_ops.bytes_to_b64_embedded_image(
                image_ops.draw_parking_lot_map(
                    video.get_first_frame(),
                    parking_lot.active_zone,
                    parking_lot.parking_spaces,
                ),
            )
        else:
            first_frame_3 = None
        if first_frame_3:
            ui.image(
                source=first_frame_3,
            )


@ui.page(path='/setup', title='Разметка')
async def setup_page():
    parking_lot_repository = ParkingLotRepository()

    async def create_parking_lot() -> None:
        with (ui.dialog(value=True) as dialog, ui.card()):
            ui.label(text='Создание парковочной зоны').classes('text-xl font-bold')
            with ui.row().classes('w-full flex items-stretch'):
                name_input = ui.input(
                    label='Название',
                    validation={
                        'Длина названия не может превышать 32 символа.': lambda value: len(value) <= 32,
                    },
                )

            async def create() -> None:
                parking_lot_repository.add_parking_lot(
                    id=uuid.uuid4(),
                    name=name_input.value,
                )
                parking_lot_repository.commit()
                ui.notify(
                    f'Парковочная зона {name_input.value} успешно создана',
                    position='top-right',
                    type='positive',
                )
                parking_lot_list.refresh(parking_lot_repository)
                dialog.close()

            with ui.row().classes('w-full'):
                ui.button('Создать', on_click=create)
                ui.button('Отмена', on_click=dialog.close)

    ui.label(text='Разметка парковочных зон').classes('text-xl font-bold')

    with ui.row():
        with ui.card():
            ui.button(text='Новая зона', on_click=create_parking_lot)
            with ui.list():
                ui.item_label('Парковочные зоны').props('header').classes('text-bold')
                ui.separator()
                parking_lot_list(parking_lot_repository)

        with ui.card().classes('w-[700px]'):
            parking_lot_setup_area(parking_lot=None, parking_lot_repository=parking_lot_repository)


class TestDriveDetector:
    def __init__(
            self,
            test_drive_registry: TestDriveRegistry,
    ) -> None:
        self._test_drive_registry = test_drive_registry
        self._lot: ParkingLot | None = None
        self._video_file_paths: dict[str, str] = {}

    def set_parking_lot(self, lot: ParkingLot) -> None:
        self._lot = lot

    def add_video_file_paths(self, source_name: str, file_path: str | os.PathLike) -> None:
        self._video_file_paths[source_name] = str(file_path)

    def detect(self, on_start: Callable) -> None:
        if self._lot is None:
            ui.notify('Не выбрана парковочная зона', position='top-right', type='negative')
            return
        if not self._video_file_paths:
            ui.notify('Не выбраны видеофайлы', position='top-right', type='negative')
            return
        on_start()
        ui.notify('Запущено обнаружение тест-драйвов', position='top-right', type='positive')
        detected_test_drives = detect_test_drives.detect_test_drives(
            video_paths=[
                path
                for name, path in sorted(self._video_file_paths.items(), key=lambda item: item[0])
            ],
            parking_polygon=self._lot.active_zone,
            parking_spaces={
                space.number: {'cx': space.position[0], 'cy': space.position[1], 'radius': space.radius}
                for space in self._lot.parking_spaces
            },
            lot_id=self._lot.id,
            output_video_dir=OUTPUT_DIRECTORY,
        )
        for test_drive in detected_test_drives:
            self._test_drive_registry.add_test_drive(test_drive)
        # self._test_drive_registry.commit()
        ui.notify('Завершено обнаружение тест-драйвов', position='top-right', type='positive')


class TestDriveData(TypedDict):
    parking_space: int
    state: str
    test_drive_number: int
    event_timestamp: float
    clip_path: str


class DetectionResult(TypedDict):
    test_drives: list[TestDriveData]


async def load_video(
        parking_lot_repository: ParkingLotRepository,
        test_drive_registry: TestDriveRegistry,
) -> None:
    detector = TestDriveDetector(
        test_drive_registry=test_drive_registry,
    )
    with (ui.dialog(value=True) as dialog, ui.card()):
        ui.label(text='Load video').classes('text-xl font-bold')

        def on_file_upload(event: nicegui_events.UploadEventArguments) -> None:
            video_file_path = INBOX_DIRECTORY / f'{uuid.uuid4()}.mp4'
            with video_file_path.open(mode='wb') as file:
                file.write(event.content.read())
            detector.add_video_file_paths(source_name=event.name, file_path=video_file_path)

        with ui.column().classes('w-full flex items-stretch'):
            ui.select(
                {
                    lot: lot.name
                    for lot in parking_lot_repository.list_parking_lots()
                    if lot.state == ParkingLotState.READY
                },
                label='Парковочная зона',
                on_change=lambda event: detector.set_parking_lot(event.value),
            )
            upload_form = ui.upload(
                label='MP4 видео',
                multiple=True,
                on_upload=on_file_upload,
                on_multi_upload=lambda: detector.detect(on_start=dialog.close),
            )

        def on_load() -> None:
            upload_form.run_method('upload')

        with ui.row().classes('w-full'):
            ui.button('Загрузить', on_click=on_load)
            ui.button('Отмена', on_click=dialog.close)


@ui.page(path='/detection', title='Обнаружение')
async def detection_page():
    parking_lot_repository = ParkingLotRepository()
    test_drive_registry = TestDriveRegistry()

    ui.label(text='Обнаружение тест-драйвов').classes('text-xl font-bold')

    with ui.row(), ui.card():
        ui.button(text='Обработать видео', on_click=lambda: load_video(
            parking_lot_repository=parking_lot_repository,
            test_drive_registry=test_drive_registry,
        ))
        table = ui.table(
            columns=[
                {'name': 'id', 'label': 'ID', 'field': 'id', 'align': 'left'},
                {'name': 'lot', 'label': 'Зона', 'field': 'lot', 'align': 'left'},
                {'name': 'space', 'label': 'Место', 'field': 'space', 'align': 'left'},
                {'name': 'start', 'label': 'Начало', 'field': 'start', 'align': 'left'},
                {'name': 'end', 'label': 'Завершение', 'field': 'end', 'align': 'left'},
                {'name': 'duration', 'label': 'Длительность', 'field': 'duration', 'align': 'left'},
                {'name': 'replay_link', 'label': 'Подробности', 'field': 'replay_link', 'align': 'left'},
            ],
            rows=[
                {
                    'id': test_drive.id,
                    'lot': parking_lot_repository.get_parking_lot(test_drive.lot_id).name,
                    'space': test_drive.space_number,
                    'start': test_drive.start_time.strftime(HUMAN_READABLE_DATE_TIME_FORMAT),
                    'end': test_drive.end_time.strftime(HUMAN_READABLE_DATE_TIME_FORMAT),
                    'duration': str(test_drive.end_time - test_drive.start_time),
                    'replay_link': f'/test-drive/{test_drive.id}',
                }
                for test_drive in test_drive_registry.list_all_test_drives()
            ],
        ).classes('overflow-y-none h-[500px]')
        table.add_slot('body-cell-replay_link', '''
                <q-td :props="props">
                    <a :href="props.value" target="_blank" rel="noopener noreferrer">Подробности</a>
                </q-td>
            ''')


@ui.page(path='/test-drive/{test_drive_id}', title='Тест-драйв')
async def replay(test_drive_id: str):
    test_drive_registry = TestDriveRegistry()
    parking_lot_repository = ParkingLotRepository()
    try:
        test_drive_id = uuid.UUID(test_drive_id)
    except ValueError:
        with ui.card():
            ui.label(text='Недействительный идентификатор тест-драйва').classes('text-xl font-bold')
            return
    test_drive = test_drive_registry.get_test_drive(test_drive_id)
    if test_drive is None:
        with ui.card():
            ui.label(text='Тест-драйв не найден').classes('text-xl font-bold')
            return
    parking_lot = parking_lot_repository.get_parking_lot(test_drive.lot_id)
    if parking_lot is None:
        with ui.card():
            ui.label(text='Парковочная зона не найдена').classes('text-xl font-bold')
            return
    with ui.card():
        title = f'Тест-драйв: {parking_lot.name} / {test_drive.space_number}'
        ui.page_title(title)
        ui.label(text=title).classes('text-xl font-bold')

        ui.label(text=f'Зона: {parking_lot.name} [{parking_lot.id}]')
        ui.label(text=f'Место: {test_drive.space_number}')
        ui.label(text=f'Длительность: {test_drive.end_time - test_drive.start_time}')

        ui.label(
            text=f'Начало {test_drive.start_time.strftime(HUMAN_READABLE_DATE_TIME_FORMAT)}',
        ).classes('text-lg font-bold')
        ui.video(src=f'/output/{test_drive.id}_departure.mp4').classes('w-[540px]')
        ui.label(
            text=f'Завершение {test_drive.end_time.strftime(HUMAN_READABLE_DATE_TIME_FORMAT)}',
        ).classes('text-lg font-bold')
        ui.video(src=f'/output/{test_drive.id}_arrival.mp4').classes('w-[540px]')


@ui.page(path='/')
async def root_page():
    # return RedirectResponse('/test-drive/fe1da530-f829-4ddb-8712-6e76f36595cd')
    return RedirectResponse('/setup')


app.add_media_files(url_path='../output', local_directory='output')
ui.run(
    title='Test drive detector',
    host='0.0.0.0',  # noqa: S104
    port=80,
    reload=False,
    dark=False,
    uvicorn_logging_level='critical',
    endpoint_documentation='none',
)
