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

from quart import Quart, render_template, request, abort, websocket, url_for

from .util import pipe_deserialize

# reconstruction:
# - make queue, schedule on scheduler
# websocket connect:
# - add websocket to reconstruction queue
# run:
# - start process

ID: t.TypeAlias = str

class Reconstruction:
    def __init__(self, id: ID):
        from phaser.main import main

        self.id: ID = id
        (read, write) = multiprocessing.Pipe(duplex=True)
        self._conn: Connection = read
        self.process: multiprocessing.Process = multiprocessing.Process(target=main, args=[write])
        self.subscribers: set[asyncio.Queue] = set()
        self.task = None

    async def watch_conn(self):
        sock = socket.fromfd(self._conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM)

        loop = asyncio.get_event_loop()
        stream_reader = asyncio.StreamReader()
        transport, protocol = await loop.connect_accepted_socket(lambda: asyncio.StreamReaderProtocol(stream_reader), sock)

        while self.running():
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
            except Exception as e:
                logging.error("Error deserializing message", exc_info=sys.exc_info())
                continue
            print(msg)

    async def subscribe(self) -> t.AsyncGenerator[t.Any, None]:
        connection = asyncio.Queue()
        self.subscribers.add(connection)
        try:
            while True:
                yield await connection.get()
        finally:  # called when connection closes
            self.subscribers.remove(connection)

    async def message_subscribers(self, msg: t.Any):
        for subscriber in self.subscribers:
            await subscriber.put(msg)

    def running(self) -> bool:
        return self.process.is_alive()

    async def start(self):
        self.process.start()
        app.add_background_task(self.watch_conn)

        self.task = asyncio.create_task(self._process_future())

    async def _process_future(self):
        await asyncio.to_thread(lambda: self.process.join())
        await self.message_subscribers({'status': 'closed'})

    def join(self, timeout: t.Optional[float] = None):
        if self.process.is_alive():
            self.process.join(timeout)


class Reconstructions:
    def __init__(self):
        self.inner: t.Dict[ID, Reconstruction] = {}
        self.subscribers: set[asyncio.Queue] = set()

    async def add(self, recons: Reconstruction):
        await recons.start()
        self.inner[recons.id] = recons
        await self.message_subscribers(('started', recons.id))

    async def remove(self, recons: Reconstruction):
        try:
            self.inner.pop(recons.id)
            await self.message_subscribers(('stopped', recons.id))
        except KeyError:
            pass

    async def subscribe(self) -> t.AsyncGenerator[t.Any, None]:
        connection = asyncio.Queue()
        self.subscribers.add(connection)
        try:
            while True:
                yield await connection.get()
        finally:
            self.subscribers.remove(connection)

    async def message_subscribers(self, msg: t.Any):
        for subscriber in self.subscribers:
            await subscriber.put({'event': msg, 'state': self.state()})

    def state(self) -> t.Any:
        return [
            {
                'id': recons.id,
                'status': "running",
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

def run() -> None:
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
    await reconstructions.add(Reconstruction(id))
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

    return {
        'result': 'failed'
    }

@app.websocket("/listen")
async def manager_websocket():
    await websocket.accept()

    await websocket.send_json({
        'event': 'connected',
        'state': reconstructions.state()
    })

    async for msg in reconstructions.subscribe():
        await websocket.send_json(msg)

@app.websocket("/<string:id>/listen")
async def dashboard_websocket(id: ID):
    try:
        recons = reconstructions[id]
    except KeyError:
        abort(404)

    async for msg in recons.subscribe():
        await websocket.send_json(msg)