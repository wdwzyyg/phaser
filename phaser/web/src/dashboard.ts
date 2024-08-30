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