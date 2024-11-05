import json
import typing as t

import aiostream.stream
from quart import Quart, render_template, request, abort, websocket

import pane

from .types import JobID, WorkerID, WorkerMessage
from .types import ManagerConnected, DashboardConnected, OkResponse
from .server import server, Job, LocalWorker, ManualWorker


def serialize(obj: t.Any, ty: t.Any = None) -> bytes:
    return json.dumps(pane.into_data(obj, ty)).encode('utf-8')


app: Quart = server.app

@app.get("/")
async def index():
    return await render_template("manager.html")

@app.post("/worker/start")
async def start_worker():
    _ = await request.get_data()
    worker = LocalWorker(server.make_workerid())
    #worker = ManualWorker(server.make_workerid())
    await server.workers.add(worker)
    return serialize(worker.state())

@app.post("/job/start")
async def start_job():
    _ = await request.get_data()
    job = Job(server.make_jobid(), server.plan)
    await server.jobs.add(job)
    return serialize(job.state())

@app.get("/job/<string:job_id>")
async def job_dashboard(job_id: JobID):
    if job_id == "fake":
        return await render_template("dashboard.html")
    if job_id not in server.jobs:
        abort(404)
    return await render_template("dashboard.html")

@app.post("/job/<string:job_id>/cancel")
async def cancel_job(job_id: JobID):
    print(f"Shutdown job ID {id}")
    try:
        job = server.jobs[job_id]
        await job.cancel()
    except KeyError:
        pass

    return serialize(OkResponse())

@app.post("/worker/<string:worker_id>/shutdown")
async def shutdown_worker(worker_id: WorkerID):
    print(f"Shutdown worker ID {id}")
    try:
        worker = server.workers[worker_id]
        await worker.cancel()
    except KeyError:
        pass

    return serialize(OkResponse())

@app.websocket("/listen")
async def manager_websocket():
    await websocket.accept()

    await websocket.send(serialize(ManagerConnected(
        server.workers.state(), server.jobs.state()
    )))

    async with aiostream.stream.merge(
        server.workers.subscribe(),
        server.jobs.subscribe(),
    ).stream() as stream:
        async for msg in stream:
            #print(f"manager msg: {msg}")
            await websocket.send(serialize(msg))

@app.websocket("/job/<string:job_id>/listen")
async def dashboard_websocket(job_id: JobID):
    try:
        job = server.jobs[job_id]
    except KeyError:
        abort(404)

    await websocket.accept()

    await websocket.send(serialize(DashboardConnected(
        job.state()
    )))

    async for msg in job.subscribe():
        #print(f"job msg: {msg}")
        await websocket.send(serialize(msg))

@app.post("/worker/<string:worker_id>/update")
async def worker_update(worker_id: WorkerID):
    try:
        worker = server.workers[worker_id]
    except KeyError:
        abort(400)

    msg: WorkerMessage = pane.convert(await request.json, WorkerMessage)  # type: ignore
    #print(f"got worker message: {msg}")
    return serialize(await worker.handle_message(msg))