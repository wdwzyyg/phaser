import asyncio
import logging
import multiprocessing
from multiprocessing.connection import Connection
import struct
import io
import sys
import socket
import random
import typing as t
import threading

from quart import Quart, render_template, request, abort, websocket, url_for

from phaser.plan import ReconsPlan
from .util import pipe_deserialize, pipe_serialize

ID: t.TypeAlias = str
ReconstructionState: t.TypeAlias = t.Literal["queued", "starting", "running", "stopping", "stopped", "done"]
PLAN: ReconsPlan


class ReconstructionStateChange(t.TypedDict):
    msg: t.Literal['state_change']
    id: ID
    state: ReconstructionState


class ReconstructionUpdate(t.TypedDict):
    msg: t.Literal['update']
    id: ID
    state: ReconstructionState
    data: t.Any


ReconstructionMessage = ReconstructionUpdate | ReconstructionStateChange


class ReconstructionStatus(t.TypedDict):
    id: ID
    state: ReconstructionState
    links: t.Dict[str, str]


def run_and_observe(plan: ReconsPlan, connection: Connection):
    from phaser.execute import execute_plan
    from phaser.utils.num import to_numpy
    from phaser.web.util import ConnectionWrapper
    from phaser.state import ReconsState

    conn = ConnectionWrapper(connection)
    conn.message("running")

    def observe_state(state: ReconsState):
        conn.update({
            'probe': to_numpy(state.probe.data),
            'obj': to_numpy(state.object.data),
            'progress': (
                state.errors.iters,
                state.errors.detector_errors
            ),
        })

    execute_plan(plan, [observe_state])


class Reconstruction:
    def __init__(self, id: ID, plan: ReconsPlan):
        print(f"main thread id: {threading.get_ident()}")

        self.id: ID = id
        (read, write) = multiprocessing.Pipe(duplex=True)
        self._conn: Connection = read
        self.process: multiprocessing.Process = multiprocessing.Process(target=run_and_observe, args=[plan, write])
        self.subscribers: set[asyncio.Queue[ReconstructionMessage]] = set()
        self.task = None
        self.state: ReconstructionState = "queued"

    async def _state_change(self, new_state: ReconstructionState):
        self.state = new_state
        await self.message_subscribers({
            'msg': 'state_change',
            'id': self.id,
            'state': new_state
        })

    async def watch_conn(self):
        print(f"watch_conn, thread id: {threading.get_ident()}")
        sock = socket.fromfd(self._conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM)

        loop = asyncio.get_event_loop()
        stream_reader = asyncio.StreamReader()
        transport, protocol = await loop.connect_accepted_socket(lambda: asyncio.StreamReaderProtocol(stream_reader), sock)

        while True:
            size, = struct.unpack("!i", await stream_reader.readexactly(4))
            if size == -1:
                # long size
                size, = struct.unpack("!Q", await stream_reader.readexactly(8))

            buf = io.BytesIO()
            remaining = size
            while remaining > 0:
                chunk = await stream_reader.read(remaining)
                if len(chunk) == 0:
                    if remaining == size:
                        raise EOFError()
                    else:
                        raise OSError("got end of file during message")
                buf.write(chunk)
                remaining -= len(chunk)

            try:
                msg = pipe_deserialize(buf.getvalue())
            except Exception:
                logging.error("Error deserializing message", exc_info=sys.exc_info())
                continue

            if msg['msg'] == 'update':
                await self.message_subscribers({
                    'msg': 'update',
                    'id': self.id,
                    'state': self.state,
                    'data': msg['data']
                })
            elif msg['msg'] == 'running':
                await self._state_change('running')
            elif msg['msg'] == 'done':
                await self._state_change('done')
            else:
                logging.error(f"Unknown message type {msg['msg']}")
                continue

    async def subscribe(self) -> t.AsyncGenerator[ReconstructionMessage, None]:
        connection: asyncio.Queue[ReconstructionMessage] = asyncio.Queue()
        self.subscribers.add(connection)
        try:
            while True:
                yield await connection.get()
        finally:  # called when connection closes
            self.subscribers.remove(connection)

    async def message_subscribers(self, msg: ReconstructionMessage):
        for subscriber in self.subscribers:
            await subscriber.put(msg)

    async def start(self):
        self.process.start()
        app.add_background_task(self.watch_conn)
        self.task = asyncio.create_task(self._process_future())
        await self._state_change('starting')

    async def stop(self):
        # TODO request stop
        await self._state_change('stopping')
        await asyncio.to_thread(lambda: self.process.join())

    async def _process_future(self):
        await asyncio.to_thread(lambda: self.process.join())
        if self.state != "done":
            await self._state_change('stopped')
        await reconstructions.remove(self.id)

    def join(self, timeout: t.Optional[float] = None):
        if self.process.is_alive():
            self.process.join(timeout)


class Reconstructions:
    def __init__(self):
        self.inner: t.Dict[ID, Reconstruction] = {}
        self.subscribers: set[asyncio.Queue] = set()

    async def add(self, recons: Reconstruction):
        self.inner[recons.id] = recons
        await recons.start()
        await self.message_subscribers({
            'msg': 'state_change',
            'id': recons.id,
            'state': 'running',
        })

    async def remove(self, id: ID):
        try:
            recons = self.inner.pop(id)
            #TODO send stop request
            await asyncio.to_thread(lambda: recons.join())
        except KeyError:
            pass

    async def subscribe(self) -> t.AsyncGenerator[ReconstructionMessage, None]:
        connection = asyncio.Queue()
        self.subscribers.add(connection)
        try:
            while True:
                yield await connection.get()
        finally:
            self.subscribers.remove(connection)

    async def message_subscribers(self, msg: ReconstructionMessage):
        for subscriber in self.subscribers:
            await subscriber.put({'event': msg, 'state': self.state()})

    def state(self) -> t.List[ReconstructionStatus]:
        return [
            {
                'id': recons.id,
                'state': "running",
                'links': {
                    'dashboard': url_for('dashboard', id=recons.id),
                    'cancel': url_for('cancel', id=recons.id),
                }
            }
            for recons in self.values()
        ]

    def __contains__(self, item: t.Any) -> bool:
        return self.inner.__contains__(item)

    def __getitem__(self, item: ID) -> Reconstruction:
        return self.inner[item]

    def items(self) -> t.ItemsView[ID, Reconstruction]:
        return self.inner.items()

    def keys(self) -> t.KeysView[ID]:
        return self.inner.keys()

    def values(self) -> t.ValuesView[Reconstruction]:
        return self.inner.values()


app = Quart(
    __name__,
    static_url_path="/static",
    static_folder="dist",
)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 5

reconstructions: Reconstructions = Reconstructions()

_ID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def make_id() -> ID:
    while True:
        id = "".join(_ID_CHARS[random.randrange(0, len(_ID_CHARS))] for _ in range(10))
        if id not in reconstructions:
            return id

def run(plan) -> None:
    global PLAN
    PLAN = plan

    logging.basicConfig(level=logging.INFO)
    app.run(port=5005, debug=True)

@app.after_serving
async def shutdown():
    for recons in reconstructions.values():
        recons.join()

@app.get("/")
async def index():
    return await render_template("manager.html")

@app.post("/start")
async def start_recons():
    _ = await request.get_data()
    id = make_id()
    await reconstructions.add(Reconstruction(id, PLAN))
    return {
        'id': id,
        'links': {'dashboard': url_for('dashboard', id=id)}
    }

@app.get("/<string:id>")
async def dashboard(id: ID):
    if id == "fake":
        return await render_template("dashboard.html")
    if id not in reconstructions:
        abort(404)
    return await render_template("dashboard.html")

@app.post("/<string:id>/cancel")
async def cancel(id: ID):
    print(f"cancel ID {id}")
    await reconstructions.remove(id)

    return {'result': 'success'}

@app.websocket("/listen")
async def manager_websocket():
    await websocket.accept()

    await websocket.send_json({
        'msg': 'connected',
        'state': reconstructions.state()
    })

    async for msg in reconstructions.subscribe():
        await websocket.send_json(msg)

@app.websocket("/<string:id>/listen")
async def dashboard_websocket(id: ID):
    print(f"attempting websocket connection, id: {id}")
    try:
        recons = reconstructions[id]
    except KeyError:
        abort(404)

    await websocket.accept()

    await websocket.send_json({
        'msg': 'connected',
        'state': recons.state
    })

    print("subscribing")
    async for msg in recons.subscribe():
        await websocket.send(pipe_serialize(msg))