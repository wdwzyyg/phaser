import React from 'react';
import * as format from 'd3-format';
import * as array from 'd3-array';

import { PlotScale } from './scale';
import Zoomer, { ZoomEvent } from './zoom';

interface PlotContextData {
    xscale: PlotScale
    yscale: PlotScale

    //setXScale: (scale: Scale) => void
    //setYScale: (scale: Scale) => void
}

const PlotContext = React.createContext<PlotContextData | undefined>(undefined);

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
    const plot = React.useContext(PlotContext);
    if (plot === undefined) {
        throw new Error("Component 'XAxis' must be used inside a 'Plot'");
    }

    let label: React.ReactElement | undefined = undefined;
    if (props.label) {
        label = <text className="axis-label" transform={`translate(${plot.xscale.rangeFromUnit(0.5)}, 50)`}>
            {props.label}
        </text>;
    }

    // TODO factor some stuff out
    // TODO replace with path

    const fmt = format.format(props.tickFormat ?? "~g");
    const tickLength = props.tickLength ?? 8;

    let ticks = array.ticks(...plot.xscale.domain, props.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = plot.xscale.transform(val);
        return <g className="tick" key={val}>
            <line x1={pos} x2={pos} y1={0} y2={tickLength} stroke="black"/>
            <text x={pos} y={tickLength} dy="0.9em">{text}</text>
        </g>;
    });

    let ax_ypos = plot.yscale.rangeFromUnit(1.0);
    let [ax_start, ax_stop] = plot.xscale.range;
    return <g className='bot-axis' transform={`translate(0, ${ax_ypos})`}>
        <line x1={ax_start} x2={ax_stop} y1="0" y2="0" stroke="black"/>
        { ticks }
        { label }
    </g>;
}

export function YAxis(props: AxisProps) {
    const plot = React.useContext(PlotContext);
    if (plot === undefined) {
        throw new Error("Component 'XAxis' must be used inside a 'Plot'");
    }

    let label: React.ReactElement | undefined = undefined;
    if (props.label) {
        label = <text className="axis-label" transform={`translate(-70, ${plot.yscale.rangeFromUnit(0.5)}) rotate(90)`}>
            {props.label}
        </text>;
    }

    const fmt = format.format(props.tickFormat ?? "~g");
    const tickLength = props.tickLength ?? 8;

    let ticks = array.ticks(...plot.yscale.domain, props.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = plot.yscale.transform(val);
        return <g className="tick" key={val}>
            <line x1={-tickLength} x2={0} y1={pos} y2={pos} stroke="black"/>
            <text x={-tickLength} y={pos} dx="-0.3em" dy="0.4em">{text}</text>
        </g>;
    });

    let ax_xpos = plot.xscale.rangeFromUnit(0.0);
    let [ax_start, ax_stop] = plot.yscale.range;
    return <g className='left-axis' transform={`translate(${ax_xpos}, 0)`}>
        <line x1="0" x2="0" y1={ax_start} y2={ax_stop} stroke="black"/>
        { ticks }
        { label }
    </g>;
}

interface PlotProps {
    width: number
    height: number
    xDomain?: [number, number]
    yDomain?: [number, number]
    margins?: [number, number, number, number]

    children?: React.ReactNode
}

export function Plot(props: PlotProps) {
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

    const [width, height] = [props.width, props.height];
    const [marginTop, marginRight, marginBottom, marginLeft] = props.margins ?? [10, 10, xAxis ? 80 : 10, yAxis ? 90 : 10];

    let initXScale = new PlotScale([-2.0, 2.0], [0, width]);
    let initYScale = new PlotScale([-2.0, 2.0], [0, height]);
    let [xscale, setXScale] = React.useState(initXScale);
    let [yscale, setYScale] = React.useState(initYScale);

    let [zoomExtent, setZoomExtent] = React.useState<[number, number]>([1.0, 40.0]);
    let [translateExtent, setTranslateExtent] = React.useState<[[number, number], [number, number]]>([[0.0, width], [0.0, height]]);

    const totalWidth = width + marginLeft + marginRight;
    const totalHeight = height + marginBottom + marginTop;
    const viewBox = [-marginLeft, -marginTop, totalWidth, totalHeight];

    const clipId = React.useMemo(() => makeId("ax-clip"), []);

    //const zoomHandler = React.useMemo(() => new ZoomHandler(xscale, yscale), [props.width, props.height, props.margins]);
    //zoomHandler.zoomExtent = [1, 20];
    //zoomHandler.translateExtent = [[0, width], [0, height]];

    function onzoom(event: ZoomEvent) {
        setXScale(event.xscale);
        setYScale(event.yscale);
    }

    const ctx = {
        width: width, height: height, xscale: xscale, yscale: yscale, //zoom: zoomHandler,
    };

    return <Zoomer xscale={initXScale} yscale={initYScale} onzoom={onzoom} zoomExtent={zoomExtent} translateExtent={translateExtent}>
        <svg className="plot" viewBox={viewBox.join(" ")} width={totalWidth} height={totalHeight}>
            <PlotContext.Provider value={ctx}>
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
            </PlotContext.Provider>
        </svg>
    </Zoomer>;
}