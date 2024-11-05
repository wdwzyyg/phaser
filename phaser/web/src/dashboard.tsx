
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtomValue, createStore, Provider } from 'jotai';


import { np } from './wasm-array';
import { JobStatus, JobUpdate, DashboardMessage, ProbeData, ObjectData, ProgressData, PartialReconsData } from './types';
import { Section } from './components';
import { ProbePlot, ObjectPlot } from './plots';
import { IArrayInterchange } from 'wasm-array';

let socket: WebSocket | null = null;
const statusState: PrimitiveAtom<JobStatus | null> = atom(null as JobStatus | null);
const probeState: PrimitiveAtom<ProbeData | null> = atom(null as ProbeData | null);
const objectState: PrimitiveAtom<ObjectData | null> = atom(null as ObjectData | null);
const progressState: PrimitiveAtom<ProgressData | null> = atom(null as ProgressData | null);
const store = createStore();


function StatusBar(props: {}) {
    let status = useAtomValue(statusState) ?? "Disconnected";

    const title_case = (s: string) => s[0].toUpperCase() + s.substring(1).toLowerCase();

    return <h2>
        Status: {title_case(status)}
    </h2>;
}

const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <Provider store={store}>
            <StatusBar/>
            <Section name="Progress"></Section>
            <Section name="Probe"><ProbePlot state={probeState}/></Section>
            <Section name="Object"><ObjectPlot state={objectState}/></Section>
        </Provider>
    </StrictMode>
);

addEventListener("DOMContentLoaded", (event) => {
    socket = new WebSocket(`ws://${window.location.host}${window.location.pathname}/listen`);
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

        //console.log(`Socket event: ${text}`)
        let data: DashboardMessage = JSON.parse(text);

        if (data.msg === 'job_update') {
            updateState(data.state);
        } else if (data.msg === 'status_change') {
            store.set(statusState, (_: any) => data.status);
        } else if (data.msg === 'job_stopped') {
            store.set(statusState, (_: any) => 'stopped');
        } else if (data.msg === 'connected') {
            store.set(statusState, (_: any) => data.state.status);
            updateState(data.state.state);
        } else {
            console.warn(`Unknown message type: ${data}`);
        }
    });

    socket.addEventListener("close", (event) => {
        console.log("Socket disconnected");
    });
});

function decodeState(state: Record<any, any>): any {
    if (state._ty !== undefined) {
        if (state._ty === 'numpy') {
            if (!np) return null;
            return np.from_interchange(state as IArrayInterchange);
        }
        throw new Error(`Unknown custom type '${state._ty}'`);
    }

    let out = {};
    for (const [k, v] of Object.entries(state)) {
        out[k] = (typeof v === 'object') ? decodeState(v) : v;
    }
    return out;
}

async function updateState(raw_state: Record<string, any>) {
    const state = decodeState(raw_state) as PartialReconsData;

    if (state.probe) {
        const probe = state.probe;
        store.set(probeState, (_: any) => probe);
    }
    if (state.object) {
        const object = state.object;
        store.set(objectState, (_: any) => object);
    }
    if (state.progress) {
        const progress = state.progress;
        store.set(progressState, (_: any) => progress);
    }
}