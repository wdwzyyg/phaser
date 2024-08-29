import asyncio
import logging
import multiprocessing
from multiprocessing.connection import Connection
import struct
import io
import os
import socket
import random
import typing as t

from quart import Quart, render_template, request, abort, websocket, url_for

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
        self.fut = None

    async def watch_conn(self):
        print(f"watch_conn()")
        sock = socket.fromfd(self._conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM)

        loop = asyncio.get_event_loop()
        stream_reader = asyncio.StreamReader()
        transport, protocol = await loop.connect_accepted_socket(lambda: asyncio.StreamReaderProtocol(stream_reader), sock)

        print(f"connected")

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

            print(buf.getvalue())

    async def subscribe(self) -> t.AsyncGenerator[t.Any, None]:
        connection = asyncio.Queue()
        self.subscribers.add(connection)
        try:
            while True:
                yield await connection.get()
        finally:  # called when connection closes
            self.subscribers.remove(connection)

    def running(self) -> bool:
        return self.process.is_alive()

    async def start(self):
        self.process.start()
        app.add_background_task(self.watch_conn)
        print(f"started future")

    async def join(self, timeout: t.Optional[float] = None):
        if self.process.is_alive():
            self.process.join(timeout)

app = Quart(
    __name__,
    static_url_path="/static",
    static_folder="dist",
)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 5

reconstructions: t.Dict[ID, Reconstruction] = {}

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
        await recons.join()

@app.get("/")
async def index():
    return await render_template("manager.html")

@app.post("/start")
async def start_recons():
    _ = await request.get_data()
    id = make_id()
    recons = Reconstruction(id)
    await recons.start()
    reconstructions[id] = Reconstruction(id)
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

@app.websocket("/<string:id>/listen")
async def listen_websocket(id: ID):
    try:
        recons = reconstructions[id]
    except KeyError:
        abort(404)

    async for msg in recons.subscribe():
        await websocket.send_json(msg)