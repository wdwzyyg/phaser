import * as d3 from 'd3';

import { Scalebar } from './scalebar';

function makeId(prefix: string): string {
    return prefix + `-${d3.format("06g")(Math.floor(Math.random() * 1000000))}`;
}

export interface Plot {
    svg: d3.Selection<SVGSVGElement, undefined, null, undefined>;
    ax: d3.Selection<SVGGElement, undefined, null, undefined>;
    canvas: d3.Selection<HTMLCanvasElement, undefined, null, undefined>;
    zoom: d3.ZoomBehavior<SVGGElement, undefined>,
}

export function canvasPlot(
    canvasSize: [number, number],
    xDomain: [number, number],
    yDomain: [number, number],
    xLabel?: string, yLabel?: string
): Plot {
    const [width, height] = canvasSize;
    const marginTop = 10;
    const marginRight = 10;
    const marginBottom = 60;
    const marginLeft = 80;

    // root SVG element
    const svg = d3.create("svg")
        .attr("class", "plot")
        .attr("viewBox", [0, 0, width + marginLeft + marginRight, height + marginBottom + marginTop])
        .attr("width", width + marginLeft + marginRight)
        .attr("height", height + marginBottom + marginTop);

    // axes clip
    const clipId = makeId("ax-clip");
    svg.append("clipPath")
        .attr("id", clipId)
    .append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height);

    // axes container
    const axCont = svg.append("g").attr("class", "ax-cont")
        .attr("transform", `translate(${marginLeft}, ${marginTop})`);
    // axes clip box
    const axClip = axCont.append("g").attr("class", "ax-clip")
        .attr("clip-path", `url(#${clipId})`);
    // axes
    const ax = axClip.append("g").attr("class", "ax");

    // x and y scales
    const xScale = d3.scaleLinear()
        .domain(xDomain)
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain(yDomain)
        .range([0, height]);

    const xAxis = (g, x) => g
        .call(d3.axisBottom(x).ticks(width / 80).tickSize(10).tickSizeOuter(0));

    const yAxis = (g, x) => g
        .call(d3.axisLeft(x).ticks(height / 80).tickSize(10).tickSizeOuter(0));

    // x and y axis
    const gx = axCont.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0, ${height})`)
        .call(xAxis, xScale);

    const gy = axCont.append("g")
        .attr("class", "y-axis")
        .attr("transform", `translate(0, 0)`)
        .call(yAxis, yScale);

    gy.selectAll(".tick text")
        .attr("x", "-10px");

    if (xLabel !== undefined) {
        gx.append("text")
            .attr("class", "x-axis-label")
            .attr("transform", `translate(${width / 2}, 50)`)
            .attr("fill", "currentColor")
            .html(xLabel);
    }
    if (yLabel !== undefined) {
        gy.append("text")
            .attr("class", "y-axis-label")
            .attr("transform", `translate(-50, ${height / 2}) rotate(90)`)
            .attr("fill", "currentColor")
            .html(yLabel);
    }

    const scalebar = new Scalebar(axCont, width, height, "m", 1e-9);
    scalebar.scale(xScale);

    // canvas container
    const canvasCont = ax.append("foreignObject")
        .attr("x", "0").attr("y", "0")
        .attr("width", `${width}`).attr("height", `${height}`);

    const canvas = (canvasCont.append("xhtml:canvas") as d3.Selection<HTMLCanvasElement, undefined, null, undefined>)
        .attr("xmlns", "http://www.w3.org/1999/xhtml")
        .attr("width", `${width}`).attr("height", `${height}`)
        .attr("id", "ax-canvas");

    // zooming logic
    const zoom = d3.zoom<SVGGElement, undefined>()
        .scaleExtent([1, 40])
        .extent([[0, 0], [width, height]])
        .translateExtent([[0, 0], [width, height]])
        .on("zoom", zoomed);

    axCont.call(zoom);

    function zoomed({transform}) {
        //console.log(`zoomed: ${transform}`);
        ax.attr("transform", `${transform}`);
        gx.call(xAxis, transform.rescaleX(xScale));
        gy.call(yAxis, transform.rescaleY(yScale));
        scalebar.scale(transform.rescaleX(xScale))
    } 

    return {
        "svg": svg, "ax": ax, "canvas": canvas, "zoom": zoom,
    }
}