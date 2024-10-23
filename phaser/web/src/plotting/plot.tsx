import React from 'react';
import { atom, useAtomValue, PrimitiveAtom } from 'jotai';

import * as format from 'd3-format';
import * as array from 'd3-array';

import { Transform1D } from './transform';
import { PlotScale, Pair } from './scale';
import { Zoomer } from "./zoom";


export interface FigureContextData<K> {
    scales: Map<K, PlotScale>
    transforms: Map<K, PrimitiveAtom<Transform1D>>
    translateExtents: Map<K, Pair>

    zoomExtent: Pair
}

export const FigureContext = React.createContext<FigureContextData<string> | undefined>(undefined);

interface FigureProps {
    scales: Map<string, PlotScale>
    translateExtents?: Map<string, Pair>

    zoomExtent?: Pair

    children?: React.ReactNode
}

export function Figure(props: FigureProps) {
    console.log("Redrawing Figure");

    let scales = new Map();
    let transforms = new Map();

    for (const [k, scale] of props.scales) {
        scales.set(k, scale);
        transforms.set(k, atom(new Transform1D()));
    }

    let translateExtents = new Map();

    if (props.translateExtents) {
        for (const k of props.scales.keys()) {
            const translateExtent = props.translateExtents.get(k);
            if (!translateExtent) throw new Error(`translateExtents missing entry for scale ${k}`);
            translateExtents.set(k, translateExtent)
        }
    } else {
        for (const k of props.scales.keys()) {
            translateExtents.set(k, [-Infinity, Infinity]);
        }
    }

    const ctx = {
        scales: scales,
        transforms: transforms,
        translateExtents: translateExtents,
        zoomExtent: props.zoomExtent || [1, Infinity],
    };

    return <FigureContext.Provider value={ctx}>
        {props.children}
    </FigureContext.Provider>;
}

export interface PlotContextData<K> {
    xscale: K
    yscale: K
}

export const PlotContext = React.createContext<PlotContextData<string> | undefined>(undefined);

function makeId(prefix: string): string {
    return prefix + `-${format.format("06g")(Math.floor(Math.random() * 1000000))}`;
}

interface AxisProps {
    label?: string | undefined

    ticks?: number
    tickFormat?: string
    tickLength?: number
}

export function XAxis(props: AxisProps) {
    const fig = React.useContext(FigureContext);
    const plot = React.useContext(PlotContext);
    if (fig === undefined || plot === undefined) {
        throw new Error("Component 'XAxis' must be used inside a 'Plot'");
    }
    let fullScale = fig.scales.get(plot.xscale)!;
    let scale = new PlotScale(
        fullScale.untransform(useAtomValue(fig.transforms.get(plot.xscale)!).unapply(fullScale.range)),
        fullScale.range
    );

    let label: React.ReactElement | undefined = undefined;
    if (props.label) {
        label = <text className="axis-label" transform={`translate(${scale.rangeFromUnit(0.5)}, 50)`}>
            {props.label}
        </text>;
    }

    // TODO factor some stuff out
    // TODO replace with path

    const fmt = format.format(props.tickFormat ?? "~g");
    const tickLength = props.tickLength ?? 8;

    let ticks = array.ticks(...scale.domain, props.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = scale.transform(val);
        return <g className="tick" key={val}>
            <line x1={pos} x2={pos} y1={0} y2={tickLength} stroke="black"/>
            <text x={pos} y={tickLength} dy="0.9em">{text}</text>
        </g>;
    });

    let ax_ypos = fig.scales.get(plot.yscale)!.rangeFromUnit(1.0);
    let [ax_start, ax_stop] = scale.range;
    return <g className='bot-axis' transform={`translate(0, ${ax_ypos})`}>
        <line x1={ax_start} x2={ax_stop} y1="0" y2="0" stroke="black"/>
        { ticks }
        { label }
    </g>;
}

export function YAxis(props: AxisProps) {
    const fig = React.useContext(FigureContext);
    const plot = React.useContext(PlotContext);
    if (fig === undefined || plot === undefined) {
        throw new Error("Component 'YAxis' must be used inside a 'Plot'");
    }
    let fullScale = fig.scales.get(plot.yscale)!;
    let scale = new PlotScale(
        fullScale.untransform(useAtomValue(fig.transforms.get(plot.yscale)!).unapply(fullScale.range)),
        fullScale.range
    );

    let label: React.ReactElement | undefined = undefined;
    if (props.label) {
        label = <text className="axis-label" transform={`translate(-70, ${scale.rangeFromUnit(0.5)}) rotate(90)`}>
            {props.label}
        </text>;
    }

    const fmt = format.format(props.tickFormat ?? "~g");
    const tickLength = props.tickLength ?? 8;

    let ticks = array.ticks(...scale.domain, props.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = scale.transform(val);
        return <g className="tick" key={val}>
            <line x1={-tickLength} x2={0} y1={pos} y2={pos} stroke="black"/>
            <text x={-tickLength} y={pos} dx="-0.3em" dy="0.4em">{text}</text>
        </g>;
    });

    let ax_xpos = fig.scales.get(plot.xscale)!.rangeFromUnit(0.0);
    let [ax_start, ax_stop] = scale.range;
    return <g className='left-axis' transform={`translate(${ax_xpos}, 0)`}>
        <line x1="0" x2="0" y1={ax_start} y2={ax_stop} stroke="black"/>
        { ticks }
        { label }
    </g>;
}

interface PlotProps {
    xscale: string
    yscale: string
    /*width: number
    height: number
    xDomain?: [number, number]
    yDomain?: [number, number]*/
    margins?: [number, number, number, number]

    children?: React.ReactNode
}

export function Plot(props: PlotProps) {
    console.log("Redrawing Plot");

    const fig = React.useContext(FigureContext);
    if (fig === undefined) {
        throw new Error("Component 'Plot' must be used inside a 'Figure'");
    }

    let xAxis: boolean = false;
    let yAxis: boolean = false;

    let clippedChildren: React.ReactNode[] = [];
    let children: React.ReactNode[] = [];

    React.Children.forEach(props.children, child => {
        if (React.isValidElement(child) && typeof child.type == "function") {
            if (child.type.name == "XAxis") {
                xAxis = true;
                children.push(child);
                //children.push(<XAxis {...child.props} axisLimits={xLim}/>);
                return;
            } else if (child.type.name == "YAxis") {
                yAxis = true;
                children.push(child);
                //children.push(<YAxis {...child.props} axisLimits={yLim}/>);
                return;
            }
        }
        clippedChildren.push(child);
    });

    const [xscale, yscale] = [fig.scales.get(props.xscale)!, fig.scales.get(props.yscale)!];
    const [width, height] = [xscale.rangeSize(), yscale.rangeSize()];
    const [marginTop, marginRight, marginBottom, marginLeft] = props.margins ?? [10, 10, xAxis ? 80 : 10, yAxis ? 90 : 10];

    const totalWidth = width + marginLeft + marginRight;
    const totalHeight = height + marginBottom + marginTop;
    const viewBox = [-marginLeft, -marginTop, totalWidth, totalHeight];

    const clipId = React.useMemo(() => makeId("ax-clip"), []);

    const ctx = {
        xscale: props.xscale, yscale: props.yscale
    };

    return <PlotContext.Provider value={ctx}> <Zoomer>
        <svg className="plot" viewBox={viewBox.join(" ")} width={totalWidth} height={totalHeight}>
            <clipPath id={clipId}><rect x={0} y={0} width={width} height={height}/></clipPath>
            <g className="ax-cont">
                <rect className="ax-box" width={width} height={height}/>
                { children }
                <g className="ax-clip" clipPath={`url(#${clipId})`}>
                    <g className="zoom">
                        { clippedChildren }
                    </g>
                </g>
            </g>
        </svg>
    </Zoomer> </PlotContext.Provider>;
}



/*
interface PlotGridProps {
    width: number
    height: number

    children?: React.ReactNode
}

export function PlotGrid(props: PlotGridProps) {
    const [ncols, nrows] = [3, 3];
    const [width, height] = [props.width, props.height];

    const zoomExtent: [number, number] = [1.0, 40.0];
    const translateExtent: [[number, number], [number, number]] = [[0.0, 10.0], [0.0, 10.0]];

    const gridStyle = {
        display: "grid",
        gridTemplateColumns: `repeat(${ncols}, ${width}px)`,
        gridAutoRows: `${height}px`,
        columnGap: "50px",
        rowGap: "50px",
    };

    return <Zoomer zoomExtent={zoomExtent} translateExtent={translateExtent}>
        <div className="plotGrid" style={gridStyle}>

        </div>
    </Zoomer>
}

export function PlotList() {
    const listStyle = {
        display: "flex",
        flexDirection: "row",
    };

    <div className="plotList" style={listStyle}>
    </div>
}
*/

/*

PlotGrid, PlotList

Zoomer contains many plots, it attaches to each one
Each plot has an associated xscale and a yscale

on zoom, an event is tracked to a given x and y axis, which are updated


*/