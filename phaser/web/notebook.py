import logging.config
import weakref
import threading
import logging
import time
import io
import os
import typing as t
import yaml
try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper

import requests
from IPython.display import display
import ipywidgets

if t.TYPE_CHECKING:
    from phaser.plan import ReconsPlan


class ServerLogHandler(logging.Handler):
    def __init__(self, manager: 'Manager'):
        self.manager = weakref.ref(manager)
        super().__init__(logging.INFO)

    def emit(self, record: logging.LogRecord):
        #print(self.format(record))
        try:
            if (mgr := self.manager()) and mgr.out:
                mgr.out.append_stdout(self.format(record) + '\n')
        except Exception:
            self.handleError(record)


def iframe(src: str, width: t.Any, height: t.Any) -> ipywidgets.HTML:
    return ipywidgets.HTML(f'\
<iframe width="{width}" height="{height}" src="{src}" frameborder="0" allowfullscreen></iframe>\
')


class Manager:
    def __init__(self, port: int = 5050):
        self.port: int = 5050

        # path from browser
        service_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
        self.root_path = f"{service_prefix}proxy/absolute/{self.port}"
        # path from server
        self.base_url = f"http://localhost:{self.port}{self.root_path}"

        self._server_thread: t.Optional[threading.Thread] = None
        self.out: t.Optional[ipywidgets.Output] = None
        self._log_handler = ServerLogHandler(self)

    def _run_server(self):
        from phaser.web.server import server

        logging.basicConfig(level=logging.INFO, handlers=[self._log_handler])

        server.run(port=self.port, root_path=self.root_path)

    def start(self):
        if self.is_running():
            raise RuntimeError("Manager already running")

        self.out = ipywidgets.Output()

        # TODO this should really be in a separate process
        self._server_thread = threading.Thread(
            target=self._run_server, name='server', daemon=True
        )
        self._server_thread.start()
        time.sleep(2)

        display(ipywidgets.Accordion(children=[
            self.out, self.manager_view()
        ], titles=['Server logs', 'Job manager'], selected_index=1))

    def __del__(self):
        self.shutdown()

    def is_running(self) -> bool:
        return (
            self._server_thread is not None and self._server_thread.is_alive()
        )

    def shutdown(self):
        if not self.is_running():
            return

        resp = requests.post(
            f"{self.base_url}/shutdown",
        )
        resp.raise_for_status()
        self._server_thread.join()  # type: ignore (we already checked)
        self._server_thread = None

    def start_job(self, plan: 'ReconsPlan'):
        from phaser.plan import ReconsPlan

        if not isinstance(plan, ReconsPlan):
            plan = ReconsPlan.from_yaml(plan)

        buf = io.StringIO()
        yaml.dump(plan.into_data(), buf, Dumper)

        resp = requests.post(
            f"{self.base_url}/job/start",
            json={
                'source': 'yaml',
                'data': buf.getvalue(),
            }
        )
        resp.raise_for_status()
        result = resp.json()
        if result['result'] == 'error':
            raise ValueError(result['msg'])

        jobs = result['jobs']
        assert len(jobs) == 1
        display(self.job_view(jobs[0]['job_id']))

    def manager_view(self) -> ipywidgets.HTML:
        return iframe(
            f'{self.root_path}/', width=800, height=500
        )

    def job_view(self, job_id: str) -> ipywidgets.HTML:
        return iframe(
            f'{self.root_path}/job/{job_id}',
            width=1200, height=2000
        )



"""
class Manager:
    def __init__(self, port: int = 5050):
        self.port: int = 5050

        self.process: t.Optional[subprocess.Popen] = None
        self.out: t.Optional[ipywidgets.Output] = None

    def start(self):
        if self.process is not None:
            raise RuntimeError("Process already started")

        self.process = subprocess.Popen(
            [sys.executable, "-m", "phaser", "serve", "--port", str(self.port)],
            env=dict(**os.environ, SCRIPT_NAME=f"/proxy/absolute/{self.port}"),
        )
        self.out = ipywidgets.Output()

        time.sleep(2)

        display(self.manager())
        display(self.out)

    def stop(self):
        if self.process is None:
            return None
        self.process.send_signal(signal.SIGINT)
        time.sleep(0.5)
        self.process.send_signal(signal.SIGINT)

        try:
            self.process.wait(10.0)
        except TimeoutError:
            self.process.kill()
            self.process.wait()

    def __del__(self):
        if self.process is not None:
            self.process.kill()

    def manager(self) -> IFrame:
        return IFrame(src=f"/proxy/absolute/{self.port}/", width=800, height=800)
"""