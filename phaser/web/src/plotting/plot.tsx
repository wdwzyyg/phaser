import React from 'react';
import { atom, useAtomValue, PrimitiveAtom } from 'jotai';

import * as d3_format from 'd3-format';
import * as d3_array from 'd3-array';
import * as d3_scale from 'd3-scale';

import { Transform1D } from './transform';
import { PlotScale, Pair } from './scale';
import { Zoomer } from "./zoom";

export interface AxisSpec {
    scale: PlotScale

    translateExtent?: Pair | boolean
    label?: string
    show?: boolean | 'one'
}

export interface Axis {
    scale: PlotScale

    translateExtent: Pair
    label?: string
    show: boolean | 'one'
}

function normalize_axis(axis: AxisSpec | PlotScale): Axis {
    if (axis instanceof PlotScale) {
        axis = {
            scale: axis
        };
    }

    axis.show = ("show" in axis) ? axis.show : true;

    if (axis.translateExtent === true || !("translateExtent" in axis)) {
        axis.translateExtent = axis.scale.domain;
    } else if (!axis.translateExtent) {
        axis.translateExtent = [-Infinity, Infinity];
    }

    return axis as Axis;
}

export interface ColorScale {
    scale?: d3_scale.ScaleSequential<string>
    label?: string
}

export interface FigureContextData<K> {
    axes: Map<K, Axis>
    transforms: Map<K, PrimitiveAtom<Transform1D>>

    scales: Map<K, PrimitiveAtom<ColorScale>>

    zoomExtent: Pair
}

export const FigureContext = React.createContext<FigureContextData<string> | undefined>(undefined);

interface FigureProps {
    axes: Map<string, AxisSpec | PlotScale>
    zoomExtent?: Pair

    scales?: Map<string, ColorScale>

    children?: React.ReactNode
}

export function Figure(props: FigureProps) {
    console.log("Redrawing Figure");

    let axes: Map<string, Axis> = new Map();
    let transforms: Map<string, PrimitiveAtom<Transform1D>> = new Map();

    for (let [k, axis] of props.axes) {
        axes.set(k, normalize_axis(axis));
        transforms.set(k, atom(new Transform1D()));
    }

    const scales: Map<string, PrimitiveAtom<ColorScale>> = new Map();

    if (props.scales) {
        for (const [k, v] of props.scales) {
            scales.set(k, atom(v));
        }
    }

    const ctx = {
        axes: axes,
        transforms: transforms,
        zoomExtent: props.zoomExtent || [1, Infinity],
        scales: scales,
    };

    return <FigureContext.Provider value={ctx}>
        {props.children}
    </FigureContext.Provider>;
}

export interface PlotContextData<K> {
    xaxis: K | Axis
    yaxis: K | Axis

    fixedAspect: boolean
}

export const PlotContext = React.createContext<PlotContextData<string> | undefined>(undefined);

function makeId(prefix: string): string {
    return prefix + `-${d3_format.format("06g")(Math.floor(Math.random() * 1000000))}`;
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
    if (fig === undefined || plot == undefined) {
        throw new Error("Component 'XAxis' must be used inside a 'Plot'");
    }

    let xtransform = (typeof plot.xaxis === "string") ? useAtomValue(fig.transforms.get(plot.xaxis)!) : new Transform1D();
    let xaxis = (typeof plot.xaxis === "string") ? fig.axes.get(plot.xaxis)! : plot.xaxis;
    let yaxis = (typeof plot.yaxis === "string") ? fig.axes.get(plot.yaxis)! : plot.yaxis;

    let fullScale = xaxis.scale;
    let scale = new PlotScale(
        fullScale.untransform(xtransform.unapply(fullScale.range)),
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

    const fmt = d3_format.format(props.tickFormat ?? "~g");
    const tickLength = props.tickLength ?? 8;

    let ticks = d3_array.ticks(...scale.domain, props.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = scale.transform(val);
        return <g className="tick" key={val}>
            <line x1={pos} x2={pos} y1={0} y2={tickLength} stroke="black"/>
            <text x={pos} y={tickLength} dy="0.9em">{text}</text>
        </g>;
    });

    let ax_ypos = yaxis.scale.rangeFromUnit(1.0);
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

    let ytransform = (typeof plot.yaxis === "string") ? useAtomValue(fig.transforms.get(plot.yaxis)!) : new Transform1D();
    let xaxis = (typeof plot.xaxis === "string") ? fig.axes.get(plot.xaxis)! : plot.xaxis;
    let yaxis = (typeof plot.yaxis === "string") ? fig.axes.get(plot.yaxis)! : plot.yaxis;

    let fullScale = yaxis.scale;
    let scale = new PlotScale(
        fullScale.untransform(ytransform.unapply(fullScale.range)),
        fullScale.range
    );

    let label: React.ReactElement | undefined = undefined;
    if (props.label) {
        label = <text className="axis-label" transform={`translate(-70, ${scale.rangeFromUnit(0.5)}) rotate(90)`}>
            {props.label}
        </text>;
    }

    const fmt = d3_format.format(props.tickFormat ?? "~g");
    const tickLength = props.tickLength ?? 8;

    let ticks = d3_array.ticks(...scale.domain, props.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = scale.transform(val);
        return <g className="tick" key={val}>
            <line x1={-tickLength} x2={0} y1={pos} y2={pos} stroke="black"/>
            <text x={-tickLength} y={pos} dx="-0.3em" dy="0.4em">{text}</text>
        </g>;
    });

    let ax_xpos = xaxis.scale.rangeFromUnit(0.0);
    let [ax_start, ax_stop] = scale.range;
    return <g className='left-axis' transform={`translate(${ax_xpos}, 0)`}>
        <line x1="0" x2="0" y1={ax_start} y2={ax_stop} stroke="black"/>
        { ticks }
        { label }
    </g>;
}

interface PlotProps {
    xaxis?: string | AxisSpec
    yaxis?: string | AxisSpec

    fixedAspect?: boolean /* = false*/
    /*width: number
    height: number
    xDomain?: [number, number]
    yDomain?: [number, number]*/
    margins?: [number, number, number, number]

    show_xaxis?: boolean
    show_yaxis?: boolean

    children?: React.ReactNode
}

export function Plot(props: PlotProps) {
    console.log("Redrawing Plot");

    const fig = React.useContext(FigureContext);
    if (fig === undefined) {
        throw new Error("Component 'Plot' must be used inside a 'Figure'");
    }

    if (!props.xaxis || !props.yaxis) {
        throw new Error("Component 'Plot' must have xaxis and yaxis props defined.");
    }

    let xaxis = (typeof props.xaxis === "string") ? fig.axes.get(props.xaxis)! : normalize_axis(props.xaxis);
    let yaxis = (typeof props.yaxis === "string") ? fig.axes.get(props.yaxis)! : normalize_axis(props.yaxis);
    if (!xaxis) throw new Error("Invalid xaxis passed to component 'Plot'");
    if (!yaxis) throw new Error("Invalid yaxis passed to component 'Plot'");

    let ctx: PlotContextData<string> = {
        xaxis: (typeof props.xaxis === "string") ? props.xaxis : xaxis,
        yaxis: (typeof props.yaxis === "string") ? props.yaxis : yaxis,
        fixedAspect: props.fixedAspect ?? false,
    };

    let show_xaxis = props.show_xaxis ?? !!xaxis.show;
    let show_yaxis = props.show_yaxis ?? !!yaxis.show;

    let clippedChildren: React.ReactNode[] = [];
    let children: React.ReactNode[] = [];

    React.Children.forEach(props.children, child => {
        clippedChildren.push(child);
    });

    if (show_xaxis) children.push(<XAxis label={xaxis.label} key="xaxis"/>)
    if (show_yaxis) children.push(<YAxis label={yaxis.label} key="yaxis"/>)

    const dims = calc_plot_dims(fig, xaxis, yaxis, show_xaxis, show_yaxis, props.margins);

    const clipId = React.useMemo(() => makeId("ax-clip"), []);

    return <PlotContext.Provider value={ctx}> <Zoomer>
        <svg className="plot" viewBox={dims.viewBox.join(" ")} width={dims.totalWidth} height={dims.totalHeight}>
            <clipPath id={clipId}><rect x={0} y={0} width={dims.width} height={dims.height}/></clipPath>
            <g className="ax-cont">
                <rect className="ax-box" width={dims.width} height={dims.height}/>
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

interface PlotDims {
    width: number
    height: number
    totalWidth: number
    totalHeight: number
    viewBox: [number, number, number, number]
}

function calc_plot_dims(
    fig: FigureContextData<string>,
    xaxis: Axis, yaxis: Axis, show_xaxis: boolean, show_yaxis: boolean,
    margins?: [number, number, number, number]
): PlotDims {
    let [xscale, yscale] = [xaxis.scale, yaxis.scale] ;

    const [width, height] = [xscale.rangeSize(), yscale.rangeSize()];
    const [marginTop, marginRight, marginBottom, marginLeft] = margins ?? [10, 10, show_xaxis ? 80 : 10, show_yaxis ? 90 : 10];

    const totalWidth = width + marginLeft + marginRight;
    const totalHeight = height + marginBottom + marginTop; 
    const viewBox: [number, number, number, number] = [-marginLeft, -marginTop, totalWidth, totalHeight];

    return {
        width: width, height: height,
        totalWidth: totalWidth, totalHeight: totalHeight,
        viewBox: viewBox,
    }
}

interface PlotGridProps {
    ncols: number;
    nrows: number;

    xaxes: string | ReadonlyArray<string>;
    yaxes: string | ReadonlyArray<string>;

    pad?: string | number; /* = 0px */

    zoomExtent?: [number, number];

    children?: React.ReactNode
}

export function PlotGrid(props: PlotGridProps) {
    const [ncols, nrows] = [props.ncols, props.nrows];

    const fig = React.useContext(FigureContext);
    if (fig === undefined) {
        throw new Error("Component 'Plot' must be used inside a 'Figure'");
    }

    let xaxes: Array<string>;
    if (typeof(props.xaxes) === "string") {
        // share x axis
        xaxes = Array(props.ncols).fill(props.xaxes);
    } else {
        if (props.xaxes.length != ncols) {
            throw new Error("PlotGrid: `xaxes` must an axis key or an array of `ncols` axis keys");
        }
        xaxes = [...props.xaxes];
    }

    let yaxes: Array<string>;
    if (typeof(props.yaxes) === "string") {
        // share y axis
        yaxes = Array(props.nrows).fill(props.yaxes);
    } else {
        if (props.yaxes.length != nrows) {
            throw new Error("PlotGrid: `yaxes` must an axis key or an array of `nrows` axis keys");
        }
        yaxes = [...props.yaxes];
    }

    if (React.Children.count(props.children) > nrows * ncols) {
        throw new Error(`PlotGrid: Too many children, maximum is nrows*ncols = ${nrows * ncols}`);
    }

    let widths: Array<number> = Array(props.ncols).fill(0);
    let heights: Array<number> = Array(props.nrows).fill(0);

    const children = React.Children.map(props.children, (child, i) => {
        const [row, col] = [Math.floor(i / ncols), i % ncols];

        if (React.isValidElement(child) && typeof child.type == "function") {
            if (child.type.name == "Plot") {
                const props_xaxis = child.props.xaxis ?? xaxes[col];
                const props_yaxis = child.props.yaxis ?? yaxes[row];
                let xaxis = (typeof props_xaxis === "string") ? fig.axes.get(props_xaxis)! : normalize_axis(props_xaxis);
                let yaxis = (typeof props_yaxis === "string") ? fig.axes.get(props_yaxis)! : normalize_axis(props_yaxis);

                const show_xaxis: boolean = child.props.show_xaxis ?? (
                    xaxis.show == "one" ? row == nrows - 1 : xaxis.show
                );
                const show_yaxis: boolean = child.props.show_yaxis ?? (
                    yaxis.show == "one" ? col == 0 : yaxis.show
                );

                const dims = calc_plot_dims(fig, xaxis, yaxis, show_xaxis, show_yaxis, child.props.margins);
                widths[col] = Math.max(widths[col], dims.totalWidth);
                heights[row] = Math.max(heights[row], dims.totalHeight);

                const plotProps = {xaxis: props_xaxis, yaxis: props_yaxis, show_xaxis: show_xaxis, show_yaxis: show_yaxis};
                child = React.cloneElement(child, plotProps);
            }
        }

        const style = {
            gridColumn: col + 1,
            gridRow: row + 1,
        };

        return <div className="plotGridItem" style={style}> { child } </div>
    });

    const gridStyle = {
        display: "grid",
        gridTemplateColumns: widths.map((v) => `${v}px`).join(' '),
        gridTemplateRows: heights.map((v) => `${v}px`).join(' '),
        gap: props.pad ?? "0px",
    };

    return <div className="plotGrid" style={gridStyle}>
        { children }
    </div>;
}

/*

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