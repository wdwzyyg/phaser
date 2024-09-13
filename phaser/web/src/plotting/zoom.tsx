import React from 'react';

import { clamp, Pair, PlotScale } from "./scale";
import Transform from "./transform";


export class ZoomEvent {
    readonly type: "start" | "zoom" | "end"

    readonly xscale: PlotScale
    readonly yscale: PlotScale
    readonly transform: Transform

    constructor(type: "start" | "zoom" | "end", xscale: PlotScale, yscale: PlotScale, transform: Transform) {
        this.type = type;
        this.xscale = xscale;
        this.yscale = yscale;
        this.transform = transform;
    }
}

function viewCoords(node: SVGElement, client: Pair): Pair {
    let svg = node.ownerSVGElement || node as SVGSVGElement;
    let pt = svg.createSVGPoint();
    pt.x = client[0]; pt.y = client[1];
    pt = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [pt.x, pt.y];
}


interface ZoomerProps {
    zoomExtent?: Pair// = [1, Infinity]
    translateExtent?: [Pair, Pair]// = [[-Infinity, Infinity], [-Infinity, Infinity]]

    xscale: PlotScale
    yscale: PlotScale

    onzoom?: (event: ZoomEvent) => void

    children?: React.ReactNode
}


export default function Zoomer(props: ZoomerProps) {
    const reactChild = React.Children.only(props.children) as React.ReactElement;
    const childRef: React.MutableRefObject<SVGElement | null> = React.useRef(null);
    const managerRef: React.MutableRefObject<ZoomManager | null> = React.useRef(null);

    React.useEffect(() => {
        console.log("constructing zoomer");
        let manager = new ZoomManager(props.xscale, props.yscale);

        if (props.onzoom) manager.bind(props.onzoom);
        if (props.zoomExtent) manager.zoomExtent = props.zoomExtent;
        if (props.translateExtent) manager.translateExtent = props.translateExtent;

        manager.register(childRef.current!);
        managerRef.current = manager;

        return () => {
            manager.cleanup();
        }
    }, []);

    if (managerRef.current) {
        if (props.zoomExtent) managerRef.current.zoomExtent = props.zoomExtent;
        if (props.translateExtent) managerRef.current.translateExtent = props.translateExtent;
    }

    return React.cloneElement(reactChild, {ref: childRef});
}


class ZoomManager {
    readonly fullXScale: PlotScale;
    readonly fullYScale: PlotScale;
    xscale: PlotScale;
    yscale: PlotScale;
    // transform, which applies in range coordinate system
    transform: Transform = new Transform();

    private listeners: Array<(event: ZoomEvent) => void> = [];
    bind(listener: (event: ZoomEvent) => void) { this.listeners.push(listener); }

    zoomExtent: Pair = [1, Infinity];
    translateExtent: [Pair, Pair] = [[-Infinity, Infinity], [-Infinity, Infinity]];

    state: "idle" | "drag" = "idle";
    dragStart: Pair = [0, 0];

    private eventListeners: Array<[Element, string, EventListener, any]> = [];

    constructor(xscale: PlotScale, yscale: PlotScale) {
        this.fullXScale = xscale;
        this.fullYScale = yscale;
        this.xscale = xscale;
        this.yscale = yscale;
    }

    private update(type: "start" | "zoom" | "end", node?: SVGElement) {
        // get new domain by applying transform to range
        let xdomain = this.fullXScale.untransform(this.transform.invert().xlim(this.fullXScale.range));
        let ydomain = this.fullYScale.untransform(this.transform.invert().ylim(this.fullYScale.range));
        // apply new domain to scales
        this.xscale = new PlotScale(xdomain, this.fullXScale.range);
        this.yscale = new PlotScale(ydomain, this.fullYScale.range);

        // update transforms
        if (node) {
            const elems = node.getElementsByClassName('zoom');
            for (let i = 0; i < elems.length; i++) {
                (elems[i] as SVGElement).setAttribute("transform", this.transform.toString());
            }
        }

        let event = new ZoomEvent(type, this.xscale, this.yscale, this.transform);
        this.listeners.forEach((listener) => listener(event));
    }

    constrain(transform: Transform): Transform {
        // taken from d3-zoom
        let currentExtent = [transform.invert().xlim(this.xscale.range), transform.invert().ylim(this.yscale.range)];
        // desired shift to bring extent to translateExtent
        let x0 = currentExtent[0][0] - this.translateExtent[0][0];
        let x1 = currentExtent[0][1] - this.translateExtent[0][1];
        let y0 = currentExtent[1][0] - this.translateExtent[1][0];
        let y1 = currentExtent[1][1] - this.translateExtent[1][1];

        return transform.pretranslate(
            // if x1 > x0, overconstrained, return the average.
            // otherwise, return 
            x1 > x0 ? (x0 + x1) / 2.0 : Math.min(0, x0) + Math.max(0, x1),
            y1 > y0 ? (y0 + y1) / 2.0 : Math.min(0, y0) + Math.max(0, y1),
        );
    }

    // events to handle:
    // mousedown (no prevent default)
    // mousemove
    // mouseup
    // wheel
    // touchstart
    // touchmove
    // touchend (no prevent default)
    // touchcancel (no prevent default)
    // on window, during gesture:
    // dragstart
    // selectstart

    mousedown(event: MouseEvent) {
        if (event.button != 0) { return; } // LMB only
        const [x, y] = viewCoords(event.currentTarget as SVGElement, [event.clientX, event.clientY]);

        this.state = "drag";
        this.dragStart = [this.xscale.untransform(x), this.yscale.untransform(y)];

        event.stopPropagation(); event.preventDefault();
    }

    mousemove(event: MouseEvent) {
        if (this.state != "drag") { return; }

        let [x, y] = viewCoords(event.currentTarget as SVGElement, [event.clientX, event.clientY]);
        [x, y] = [this.xscale.untransform(x), this.yscale.untransform(y)];
        //console.log(`translating [${x}, ${y}] => [${this.dragStart[0]}, ${this.dragStart[1]}]`);
        let [deltaX, deltaY] = [this.xscale.scale(x - this.dragStart[0]), this.yscale.scale(y - this.dragStart[1])]

        this.transform = this.constrain(this.transform.translate(deltaX, deltaY));
        this.update("zoom", event.currentTarget as SVGElement);
        event.stopPropagation(); event.preventDefault();
    }

    mouseup(event: MouseEvent) {
        this.state = "idle";
        event.stopPropagation(); event.preventDefault();
    }

    wheel(event: WheelEvent) {
        const [x, y] = viewCoords(event.currentTarget as SVGElement, [event.clientX, event.clientY]);
        const k = Math.exp(-event.deltaY / 500.0);
        const totalK = this.transform.k.map((oldK) => clamp(k * oldK, this.zoomExtent)) as Pair;

        const [origx, origy] = this.transform.unapply([x, y]);
        this.transform = this.constrain(new Transform(totalK, [-origx * totalK[0] + x, -origy * totalK[1] + y]));

        this.update("zoom", event.currentTarget as SVGElement);

        event.stopPropagation(); event.preventDefault();
    }

    register(elem: SVGElement) {
        this.addListener(elem, "mousedown", this.mousedown.bind(this));
        this.addListener(elem, "mousemove", this.mousemove.bind(this));
        this.addListener(elem, "mouseup", this.mouseup.bind(this));
        this.addListener(elem, "wheel", this.wheel.bind(this));
    }

    private addListener<K extends keyof SVGElementEventMap>(elem: SVGElement, type: K, listener: (this: SVGElement, ev: SVGElementEventMap[K]) => any, options?: boolean | AddEventListenerOptions) {
        elem.addEventListener(type, listener, options);
        this.eventListeners.push([elem, type, listener as EventListener, options]);
    }

    cleanup() {
        for (let [elem, type, listener, options] of this.eventListeners) {
            elem.removeEventListener(type, listener, options);
        }
    }
}