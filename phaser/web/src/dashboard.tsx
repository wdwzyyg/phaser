
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtomValue, createStore, Provider } from 'jotai';


import { np } from './wasm-array';
import { ReconstructionStatus, ReconstructionUpdate, ProbeData, ObjectData, ProgressData } from './types';
import { Section } from './components';
import { ProbePlot, ObjectPlot } from './plots';

let socket: WebSocket | null = null;
const statusState: PrimitiveAtom<ReconstructionStatus | null> = atom(null as ReconstructionStatus | null);
const probeState: PrimitiveAtom<ProbeData> = atom(null as ProbeData);
const objectState: PrimitiveAtom<ObjectData> = atom(null as ObjectData);
const progressState: PrimitiveAtom<ProgressData> = atom(null as ProgressData);
const store = createStore();


function StatusBar(props: {}) {
    let status = useAtomValue(statusState) ?? "Unknown";

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

        console.log(`Socket event: ${text}`)
        let data = JSON.parse(text);

        if (data.msg === 'update') {
            handleUpdate(data as ReconstructionUpdate);
        } else if (data.msg === 'state_change' || data.msg === 'connected') {
            store.set(statusState, (_: any) => data.state);
        } else {
            console.warn(`Unknown message type: ${data.msg}`);
        }
    });

    socket.addEventListener("close", (event) => {
        console.log("Socket disconnected");
    });
});

async function handleUpdate(update: ReconstructionUpdate) {
    if (!np) return;

    if (update.state != store.get(statusState)) {
        store.set(statusState, (_: any) => update.state);
    }
    if ('probe' in update.data || 'init_probe' in update.data) {
        const probe = np.from_interchange(update.data.probe ?? update.data.init_probe);
        store.set(probeState, (_: any) => probe);
    }
    if ('obj' in update.data || 'init_obj' in update.data) {
        const object = np.from_interchange(update.data.obj ?? update.data.init_obj);
        store.set(objectState, (_: any) => object);
    }
    if ('progress' in update.data) {
        const [iters, errors] = [np.from_interchange(update.data.progress[0]), np.from_interchange(update.data.progress[0])];
        store.set(progressState, (_: any) => { return {iters: iters, errors: errors}; });
    }
}