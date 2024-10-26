
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtomValue, createStore, Provider } from 'jotai';


import { np } from './wasm-array';
import { ReconstructionStatus, ReconstructionUpdate, ProbeData, ObjectData, ProgressData } from './types';
import { Section } from './components';
import { ProbePlot } from './plots';

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
            <Section name="Object"></Section>
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
    if ('probe' in update.data) {
        const probe = np.from_interchange(update.data.probe);
        store.set(probeState, (_: any) => probe);
    }
    if ('object' in update.data) {
        const object = np.from_interchange(update.data.object);
        store.set(objectState, (_: any) => object);
    }
    if ('progress' in update.data) {
        const [iters, errors] = [np.from_interchange(update.data.progress[0]), np.from_interchange(update.data.progress[0])];
        store.set(progressState, (_: any) => { return {iters: iters, errors: errors}; });
    }
}

/*
import * as d3_scale from 'd3-scale';
import * as d3_scales from 'd3-scale-chromatic';

import { Section, HBox } from './components';

import { PlotScale } from './plotting/scale';
import { Figure, Plot, PlotGrid, AxisSpec, ColorScale } from './plotting/plot';
import { Colorbar } from './plotting/colorbar';

const axes: Map<string, AxisSpec> = new Map([
    ["x1", {
        scale: new PlotScale([-2.0, 2.0], [0.0, 200.0]),
        label: "X1",
        show: 'one',
    }],
    ["x2", {
        scale: new PlotScale([-4.0, 4.0], [0.0, 400.0]),
        label: "X2",
        show: 'one',
    }],
    ["y1", {
        scale: new PlotScale([-2.0, 2.0], [0.0, 200.0]),
        label: "Y1",
        show: 'one',
    }],
    ["y2", {
        scale: new PlotScale([-2.0, 2.0], [0.0, 200.0]),
        label: "Y2",
        show: 'one',
    }],
]);

const scales: Map<string, ColorScale> = new Map([
    ["v", {
        scale: d3_scale.scaleSequential(d3_scales.interpolateMagma),
        label: "Values",
    }]
]);

const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <Section name="Section 1">
            <Figure axes={axes}>
                <Plot xaxis="x1" yaxis="y1">
                    <rect x="50" y="50" width="100" height="100" />
                </Plot>
            </Figure>
        </Section>
        <Section name="Section 2">
            <Figure axes={axes} scales={scales}>
                <HBox>
                    <PlotGrid ncols={2} nrows={2} xaxes={"x1"} yaxes={"y1"}>
                        <Plot fixedAspect={true}><rect x="50" y="50" width="100" height="100" /></Plot>
                        <Plot fixedAspect={true}><rect x="50" y="50" width="100" height="100" /></Plot>
                        <Plot fixedAspect={true}><rect x="50" y="50" width="100" height="100" /></Plot>
                        <Plot fixedAspect={true}><rect x="50" y="50" width="100" height="100" /></Plot>
                    </PlotGrid>
                    <Colorbar scale="v"></Colorbar>
                </HBox>
            </Figure>
        </Section>
    </StrictMode>
);
*/


/*

import * as d3 from 'd3';
import * as np from 'wasm-array';

import { make_recip_grid, make_focused_probe } from './optics';
import { canvasPlot } from './plot';

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

let n = 512;
let wavelength = 0.0251;
let aperture = 10;
let defocus = 100;

const plot = canvasPlot([n, n], [-5, 5], [-5, 5], "X [nm]", "Y [nm]");

document.getElementById("probe-plot")!.appendChild(plot.svg.node()!);
const ctx = plot.canvas.node()?.getContext("2d")!;

let [ky, kx] = make_recip_grid([100., 100.], n);

function sleep(ms: number): Promise<undefined> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function run() {
    const imageData = ctx.createImageData(n, n);

    const n_defocuses = 10;

    // warmup
    //make_focused_probe(ky, kx, wavelength, aperture, defocus);

    const startTime = Date.now();

    for (let i = 0; i < n_defocuses; i++) {
        console.log(`running, i = ${i}`);
        let probe = make_focused_probe(ky, kx, wavelength, aperture, defocus + 50*(i**1.7));
        let probe_mag = np.abs(probe);
        imageData.data.set(np.expr`${probe_mag} / ${np.max(probe_mag)}`.apply_cmap('magma'));
        ctx.putImageData(imageData, 0, 0);

        await sleep(200.);
    }

    const elapsed = Date.now() - startTime;
    console.log(`Elapsed time: ${elapsed} ms (${elapsed / n_defocuses} ms per iteration)`);
}

run();

*/