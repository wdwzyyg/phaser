
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtomValue, createStore, Provider } from 'jotai';

import { JobState, WorkerState, ManagerMessage } from './types';
import { Section } from './components';

let socket: WebSocket | null = null;
const jobs: PrimitiveAtom<Array<JobState>> = atom([] as Array<JobState>);
const workers: PrimitiveAtom<Array<WorkerState>> = atom([] as Array<WorkerState>);
const store = createStore();

export function Workers(props: {}) {
    const cols = [20, 60, 20].map((w, i) => <col style={{width: `${w}%`}} key={i}></col>);
    const headers = ["ID", "Status", "Shutdown"].map((name, i) => <th scope="col" key={i}>{name}</th>);

    const workers_val = useAtomValue(workers);

    const rows = workers_val.map((worker) => {
        return <tr key={worker.worker_id}>
            <td>{worker.worker_id}</td>
            <td>{worker.status}</td>
            <td><button className="simple-button" onClick={(e) => shutdown_worker(worker, e)}/></td>
        </tr>;
    });


    return <table>
        <colgroup>{cols}</colgroup>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows}</tbody>
    </table>;
}

export function Jobs(props: {}) {
    const cols = [20, 60, 10, 10].map((w, i) => <col style={{width: `${w}%`}} key={i}></col>);
    const headers = ["ID", "Status", "Watch", "Cancel"].map((name, i) => <th scope="col" key={i}>{name}</th>);

    const jobs_val = useAtomValue(jobs);

    const rows = jobs_val.map((job) => {
        return <tr key={job.job_id}>
            <td>{job.job_id}</td>
            <td>{job.status}</td>
            <td><a className="simple-button" href={job.links.dashboard}></a></td>
            <td><button className="simple-button" onClick={(e) => cancel_job(job, e)}/></td>
        </tr>;
    });

    return <table>
        <colgroup>{cols}</colgroup>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows}</tbody>
    </table>;
}

function start_worker(worker_type: string): (e: React.MouseEvent) => void {
    return (e: React.MouseEvent) => {
        fetch(`/worker/${worker_type}/start`, {
            method: "POST",
            body: "",
        })
        .then((response) => response.ok ? response.json() : Promise.reject(response))
        .then((json) => {
            console.log(`Got response: ${JSON.stringify(json)}`);
        })
        .catch((response: Response) => {
            console.error(`Error: HTTP ${response.status} ${response.statusText}`)
        });
    };
}

function shutdown_worker(worker: WorkerState, e: React.MouseEvent) {
    fetch(worker.links.shutdown, {
        method: "POST",
        body: "",
    })
    .then((response) => response.ok ? response.json() : Promise.reject(response))
    .then((json) => {
        console.log(`Got response: ${JSON.stringify(json)}`);
    })
    .catch((response: Response) => {
        console.error(`Error: HTTP ${response.status} ${response.statusText}`)
    });
};

function start_job(e: React.MouseEvent) {
    fetch("/job/start", {
        method: "POST",
        body: "",

    })
    .then((response) => response.ok ? response.json() : Promise.reject(response))
    .then((json) => {
        console.log(`Got response: ${JSON.stringify(json)}`);
    })
    .catch((response: Response) => {
        console.error(`Error: HTTP ${response.status} ${response.statusText}`)
    });
};

function cancel_job(job: JobState, e: React.MouseEvent) {
    console.log(`cancelRecons: recons: ${JSON.stringify(job)}`);
    fetch(job.links.cancel, {
        method: "POST",
        body: "",
    })
    .then((response) => response.ok ? response.json() : Promise.reject(response))
    .then((json) => {
        console.log(`Got response: ${JSON.stringify(json)}`);
    })
    .catch((response: Response) => {
        console.error(`Error: HTTP ${response.status} ${response.statusText}`)
    });
};

const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <Provider store={store}>
            <Section name="Start reconstruction">
                <button onClick={start_worker("local")}>Start local worker</button>
                <button onClick={start_worker("slurm")}>Start slurm worker</button>
                <button onClick={start_job}>Start reconstruction</button>
            </Section>
            <Section name="Workers">
                <Workers/>
            </Section>
            <Section name="Jobs">
                <Jobs/>
            </Section>
        </Provider>
    </StrictMode>
);

addEventListener("DOMContentLoaded", (event) => {
    socket = new WebSocket(`ws://${window.location.host}/listen`);
    socket.binaryType = "arraybuffer";

    socket.addEventListener("open", (event) => {
        console.log("Socket connected");
    });

    socket.addEventListener("error", (event) => {
        console.log("Socket error: ", event);
    });

    socket.addEventListener("message", (event) => {
        let text: string;
        if (event.data instanceof ArrayBuffer) {
            let utf8decoder = new TextDecoder();
            text = utf8decoder.decode(event.data);
        } else {
            text = event.data;
        }

        console.log(`Socket event: ${text}`)
        let data: ManagerMessage = JSON.parse(text);

        if (data.msg === "jobs_update") {
            store.set(jobs, (prev) => data.state);
        } else if (data.msg === "workers_update") {
            store.set(workers, (prev) => data.state);
        } else if (data.msg === "connected") {
            store.set(workers, (prev) => data.workers);
            store.set(jobs, (prev) => data.jobs);
        }
    });

    socket.addEventListener("close", (event) => {
        console.log("Socket disconnected");
    });

    let sections = document.getElementsByClassName("section-header");
    for (let i = 0; i < sections.length; i++) {
        const sibling = sections[i].nextElementSibling;
        if (sibling === null || !sibling?.classList.contains("section")) {
            continue
        }
        sections[i].addEventListener("click", function() {
            sibling.classList.toggle("collapsed");
        })
    }
});