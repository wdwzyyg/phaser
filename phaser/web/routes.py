import json
import typing as t

import aiostream.stream
from quart import Quart, render_template, request, Response, abort, websocket

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

@app.post("/worker/<string:worker_type>/start")
async def start_worker(worker_type: str):
    _ = await request.get_data()

    worker_id = server.make_workerid()

    if worker_type not in ('manual', 'local', 'slurm'):
        abort(404)

    if worker_type == 'manual':
        worker = ManualWorker(worker_id)
    elif worker_type == 'local':
        worker = LocalWorker(worker_id, server.get_worker_url(worker_id))
    elif worker_type == 'slurm':
        try:
            await server.slurm_manager.check_slurm_exists()
        except RuntimeError as e:
            abort(Response(f"Slurm not available: {e}", 400))
        # TODO: this is hardcoded
        url = server.get_worker_url(worker_id).replace('localhost', '172.22.254.14')
        worker = await server.slurm_manager.make_worker(worker_id, url)

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

@app.get("/job/<string:job_id>/logs")
async def job_logs(job_id: JobID):
    try:
        job = server.jobs[job_id]
    except KeyError:
        abort(404)

    limit = min(request.args.get('limit', 100, type=int), 100)
    before = request.args.get('before', len(job.logs), type=int)

    first = max(before-limit, 0)
    last = before - 1
    logs = job.logs[first:before]

    return {
        'first': first,
        'last': last,
        'length': len(logs),
        'total_length': len(job.logs),
        'logs': logs,
    }


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
    #print("dashboard_websocket")
    try:
        job = server.jobs[job_id]
    except KeyError:
        abort(404)

    await websocket.accept()

    await websocket.send(serialize(DashboardConnected(
        job.state(full=True)
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