import base64
import json
from multiprocessing.connection import Connection
import typing as t

import numpy


class ReconstructionMessage(t.TypedDict):
    msg: t.Literal['update', 'running', 'done']
    data: t.Dict[str, t.Any]


class PipeEncoder(json.JSONEncoder):
    def default(self, obj: t.Any) -> t.Any:
        if isinstance(obj, numpy.ndarray):
            d = obj.__array_interface__
            d['data'] = base64.urlsafe_b64encode(obj.tobytes()).decode('ascii')
            d['_ty'] = 'numpy'
            return d

        if isinstance(obj, bytes):
            return {
                '_ty': 'bytes',
                'data': base64.urlsafe_b64decode(obj).decode('ascii')
            }

        return super().default(obj)


class _dummy():
    def __init__(self, array_interface: t.Any):
        self.__array_interface__ = array_interface


def _pipe_decode_obj(obj: t.Dict[t.Any, t.Any]) -> t.Any:
    if '_ty' not in obj:
        return obj
    ty = obj.pop('_ty')

    if ty == 'numpy':
        obj['data'] = base64.urlsafe_b64encode(obj['data'].encode('utf-8'))
        obj['shape'] = tuple(obj['shape'])
        return numpy.array(_dummy(obj))

    if ty == 'bytes':
        return base64.urlsafe_b64decode(obj['data'].encode('utf-8'))

    raise ValueError(f"Unknown custom type '{ty}', while parsing JSON object {obj}")


def pipe_serialize(obj: t.Any) -> bytes:
    return json.dumps(obj, cls=PipeEncoder).encode('utf-8')


def pipe_deserialize(buf: bytes) -> t.Any:
    return json.loads(buf, object_hook=_pipe_decode_obj)


class ConnectionWrapper():
    def __init__(self, conn: Connection):
        self.inner = conn

    def send(self, msg: ReconstructionMessage):
        self.inner.send_bytes(pipe_serialize(msg))

    def update(self, data: t.Any):
        self.send({'msg': 'update', 'data': data})

    def message(self, msg: t.Literal['running', 'done']):
        self.send({'msg': msg, 'data': {}})

    def recv(self) -> t.Any:
        return pipe_deserialize(self.inner.recv_bytes())