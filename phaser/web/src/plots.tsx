import React from 'react';
import { atom, useAtomValue, PrimitiveAtom } from 'jotai';

import * as d3_scale from 'd3-scale';
import * as d3_scales from 'd3-scale-chromatic';


import { np } from './wasm-array';
import { ProbeData, ObjectData } from './types';
import { PlotScale } from './plotting/scale';
import { Figure, PlotGrid, Plot, AxisSpec, ColorScale, PlotImage } from './plotting/plot';
import { Colorbar } from './plotting/colorbar';
import { HBox } from './components';


interface ObjectPlotProps {
    state: PrimitiveAtom<ObjectData>
}

export function ObjectPlot(props: ObjectPlotProps) {
    let object = useAtomValue(props.state);
    if (!object || !np) return <div></div>;

    while (object.shape.length > 2) {
        object = np.nanmean(object, [0]);
    }
    const [ny, nx] = object.shape.values();

    const phase = np.angle(object);
    const [vmin, vmax]: [number, number] = [
        np.nanmin(phase).toNestedArray() as number,
        np.nanmax(phase).toNestedArray() as number
    ];

    const axes: Map<string, AxisSpec> = new Map([
        ["x", {
            scale: new PlotScale([0, nx], [0.0, 400.0]),
            label: "X",
            show: 'one',
        }],
        ["y", {
            scale: new PlotScale([0, ny], [0.0, 400.0]),
            label: "Y",
            show: 'one',
        }],
    ]);

    const scales: Map<string, ColorScale> = new Map([
        ["phase", {
            cmap: 'magma',
            range: [vmin, vmax],
            label: "Object Phase",
        }]
    ]);

    return <Figure axes={axes} scales={scales}>
        <HBox>
            <Plot xaxis="x" yaxis="y"><PlotImage data={phase} scale="phase"/></Plot>
            <Colorbar scale="phase"/>
        </HBox>
    </Figure>;
}

interface ProbePlotProps {
    state: PrimitiveAtom<ProbeData>
}

export function ProbePlot(props: ProbePlotProps) {
    const probes = useAtomValue(props.state);
    if (!probes || !np) return <div></div>;

    const [nprobes, ny, nx] = probes.shape.values();

    const intensities = np.abs2(probes);
    const [vmin, vmax]: [number, number] = [
        np.nanmin(intensities).toNestedArray() as number,
        np.nanmax(intensities).toNestedArray() as number
    ];

    const axes: Map<string, AxisSpec> = new Map([
        ["x", {
            scale: new PlotScale([0, nx], [0.0, 150.0]),
            label: "X",
            show: 'one',
        }],
        ["y", {
            scale: new PlotScale([0, ny], [0.0, 150.0]),
            label: "Y",
            show: 'one',
        }],
    ]);

    const scales: Map<string, ColorScale> = new Map([
        ["intensity", {
            cmap: 'magma',
            range: [vmin, vmax],
            label: "Probe Intensity",
        }]
    ]);

    const plots = np.split(intensities).map((intensity, i) =>
        <Plot key={i}><PlotImage data={intensity} scale="intensity"/></Plot>
    );

    return <Figure axes={axes} scales={scales}>
        <HBox>
            <PlotGrid ncols={nprobes} nrows={1} xaxes="x" yaxes="y">
                {plots}
            </PlotGrid>
            <Colorbar scale="intensity"/>
        </HBox>
    </Figure>;
}