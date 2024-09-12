import React from 'react';
import * as format from 'd3-format';
import * as array from 'd3-array';

interface PlotContextData {
    width: number
    height: number

    xLim: [number, number]
    yLim: [number, number]

    setXLim: (xLim: [number, number]) => void
    setYLim: (xLim: [number, number]) => void
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
        label = <text className="axis-label" transform={`translate(${plot.width / 2}, 50)`}>
            {props.label}
        </text>;
    }

    // TODO factor some stuff out
    // TODO replace with path

    const fmt = format.format(props.tickFormat ?? "~g");
    const tickLength = props.tickLength ?? 8;

    let ticks = array.ticks(plot.xLim[0], plot.xLim[1], props.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = (val - plot.xLim[0]) * plot.width / (plot.xLim[1] - plot.xLim[0]);
        console.log(`tick pos ${pos} text ${text}`);
        return <g className="tick" key={val}>
            <line x1={pos} x2={pos} y1={0} y2={tickLength} stroke="black"/>
            <text x={pos} y={tickLength} dy="0.9em">{text}</text>
        </g>;
    });

    return <g className='x-axis' transform={`translate(0, ${plot.height})`}>
        <line x1="0" y1="0" x2={plot.width} y2="0" stroke="black"/>
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
        label = <text className="axis-label" transform={`translate(-70, ${plot.height / 2}) rotate(90)`}>
            {props.label}
        </text>;
    }

    const fmt = format.format(props.tickFormat ?? "~g");
    const tickLength = props.tickLength ?? 8;

    let ticks = array.ticks(plot.xLim[0], plot.xLim[1], props.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = (val - plot.yLim[0]) * plot.height / (plot.yLim[1] - plot.yLim[0]);
        console.log(`tick pos ${pos} text ${text}`);
        return <g className="tick" key={val}>
            <line x1={-tickLength} x2={0} y1={pos} y2={pos} stroke="black"/>
            <text x={-tickLength} y={pos} dx="-0.3em" dy="0.4em">{text}</text>
        </g>;
    });

    return <g className='y-axis'>
        <line x1="0" y1="0" x2="0" y2={plot.height} stroke="black"/>
        { ticks }
        { label }
    </g>;
}

interface PlotProps {
    width: number
    height: number
    margins?: [number, number, number, number]

    children?: React.ReactNode
}

export function Plot(props: PlotProps) {
    let xAxis: boolean = false;
    let yAxis: boolean = false;

    let [fullXLim, setFullXLim] = React.useState<[number, number]>([-2.0, 2.0]);
    let [fullYLim, setFullYLim] = React.useState<[number, number]>([-2.0, 2.0]);
    let [xLim, setXLim] = React.useState<[number, number]>(fullXLim);
    let [yLim, setYLim] = React.useState<[number, number]>(fullYLim);

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

    const [marginTop, marginRight, marginBottom, marginLeft] = props.margins ?? [10, 10, xAxis ? 80 : 10, yAxis ? 90 : 10];

    const viewBox = [0, 0, props.width + marginLeft + marginRight, props.height + marginBottom + marginTop];
    const [width, height] = [props.width, props.height];
    const totalWidth = width + marginLeft + marginRight;
    const totalHeight = height + marginBottom + marginTop;

    const clipId = React.useMemo(() => makeId("ax-clip"), []);

    /*const zoomHandler = React.useMemo(() => new ZoomHandler(width, height), [props.width, props.height, props.margins]);

    zoomHandler.bind((event) => {
        console.log("zoom event")
    });*/

    const ctx = {
        width: width, height: height, xLim: xLim, yLim: yLim, setXLim: setXLim, setYLim: setYLim, //zoom: zoomHandler,
    };

    return <svg className="plot" viewBox={viewBox.join(" ")} width={totalWidth} height={totalHeight}>
        <PlotContext.Provider value={ctx}>
            <clipPath id={clipId}><rect x={0} y={0} width={width} height={height}/></clipPath>
            <g className="ax-cont" transform={`translate(${marginLeft}, ${marginTop})`}>
                <rect className="ax-box" width={width} height={height}/>
                { children }
                <g className="ax-clip" clipPath={`url(#${clipId})`}>
                    { clippedChildren }
                </g>
            </g>
        </PlotContext.Provider>
    </svg>;
}