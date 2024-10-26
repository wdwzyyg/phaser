import React from 'react';
import { useAtomValue, PrimitiveAtom } from 'jotai';

import * as d3_scale from 'd3-scale';
import * as d3_scales from 'd3-scale-chromatic';


import { np } from './wasm-array';
import { ProbeData, ObjectData } from './types';
import { PlotScale } from './plotting/scale';
import { Figure, PlotGrid, Plot, AxisSpec, ColorScale } from './plotting/plot';
import { Colorbar } from './plotting/colorbar';


interface ObjectPlotProps {
    state: PrimitiveAtom<ObjectData>
}

export function ObjectPlot(props: ObjectPlotProps) {
    const state = useAtomValue(props.state);
    if (!state) return <div></div>;

    return <div></div>;
}

interface ProbePlotProps {
    state: PrimitiveAtom<ProbeData>
}

export function ProbePlot(props: ProbePlotProps) {
    const probes = useAtomValue(props.state);
    if (!probes || !np) return <div></div>;

    console.log(`Probe: ${probes.toString()}`);

    const [nprobes, ny, nx] = probes.shape.values();

    const probes_intensity = np.expr`${np.abs(probes)}**2`;

    const axes: Map<string, AxisSpec> = new Map([
        ["x", {
            scale: new PlotScale([0, nx], [0.0, 200.0]),
            label: "X",
            show: 'one',
        }],
        ["y", {
            scale: new PlotScale([0, ny], [0.0, 200.0]),
            label: "Y",
            show: 'one',
        }],
    ]);

    const scales: Map<string, ColorScale> = new Map([
        ["v", {
            scale: d3_scale.scaleSequential(d3_scales.interpolateMagma),
            label: "Intensity",
        }]
    ]);

    return <Figure axes={axes} scales={scales}>
        <PlotGrid ncols={nprobes} nrows={1} xaxes="x" yaxes="y">
            <Colorbar scale="v"/>
        </PlotGrid>
    </Figure>;
}