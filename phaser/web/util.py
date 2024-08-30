import base64
import json
from multiprocessing.connection import Connection
import typing as t

import numpy


class PipeEncoder(json.JSONEncoder):
    def default(self, obj: t.Any) -> t.Any:
        if isinstance(obj, numpy.ndarray):
            d = obj.__array_interface__
            d['data'] = base64.encodebytes(obj.tobytes()).decode('ascii')
            d['_ty'] = 'numpy'
            return d

        if isinstance(obj, bytes):
            return {
                '_ty': 'bytes',
                'data': base64.encodebytes(obj).decode('ascii')
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
        obj['data'] = base64.decodebytes(obj['data'].encode('utf-8'))
        obj['shape'] = tuple(obj['shape'])
        return numpy.array(_dummy(obj))

    if ty == 'bytes':
        return base64.decodebytes(obj['data'].encode('utf-8'))

    raise ValueError(f"Unknown custom type '{ty}', while parsing JSON object {obj}")


def pipe_serialize(obj: t.Any) -> bytes:
    return json.dumps(obj, cls=PipeEncoder).encode('utf-8')


def pipe_deserialize(buf: bytes) -> t.Any:
    return json.loads(buf, object_hook=_pipe_decode_obj)


class ConnectionWrapper():
    def __init__(self, conn: Connection):
        self.inner = conn

    def send(self, obj: t.Any):
        self.inner.send_bytes(pipe_serialize(obj))

    def recv(self) -> t.Any:
        return pipe_deserialize(self.inner.recv_bytes())