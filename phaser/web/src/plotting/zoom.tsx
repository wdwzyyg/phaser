import React from 'react';
import { useAtom } from 'jotai';

import { clamp, PlotScale, Pair } from "./scale";
import { Transform1D, Transform2D } from "./transform";

import { PlotContext, FigureContext } from "./plot";


function viewCoords(node: SVGElement, client: Pair): Pair {
    let svg = node.ownerSVGElement || node as SVGSVGElement;
    let pt = svg.createSVGPoint();
    pt.x = client[0]; pt.y = client[1];
    pt = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [pt.x, pt.y];
}

export function Zoomer({children: children}: {children?: React.ReactNode}) {
    const fig = React.useContext(FigureContext)!;
    const plot = React.useContext(PlotContext)!;

    const child = React.Children.only(children) as React.ReactElement;
    const childRef: React.MutableRefObject<(HTMLElement & SVGSVGElement) | null> = React.useRef(null);
    const managerRef: React.MutableRefObject<ZoomManager | null> = React.useRef(null);

    let [xtrans, setXTrans] = useAtom(fig.transforms.get(plot.xscale)!);
    let [ytrans, setYTrans] = useAtom(fig.transforms.get(plot.yscale)!);

    let xscale = fig.scales.get(plot.xscale)!;
    let yscale = fig.scales.get(plot.yscale)!;

    React.useEffect(() => {
        if (!managerRef.current) {
            let zoomExtent = fig.zoomExtent;
            let xTranslateExtent = fig.translateExtents.get(plot.xscale)!;
            let yTranslateExtent = fig.translateExtents.get(plot.xscale)!;
            let transform = Transform2D.from_1d(xtrans, ytrans);

            managerRef.current = new ZoomManager(
                xscale, yscale, transform, setXTrans, setYTrans,
                xTranslateExtent, yTranslateExtent,
                zoomExtent
            );
        }
        const manager = managerRef.current;

        manager.register(childRef.current!);

        () => {
            manager.unregister(childRef.current!);
        }
    }, [fig, plot.xscale, plot.yscale, xscale, yscale]);

    React.useEffect(() => {
        if (!managerRef.current) return;
        const manager = managerRef.current;

        manager.transform = Transform2D.from_1d(xtrans, ytrans);
        manager.updateTransform();
    }, [xtrans, ytrans])

    return React.cloneElement(child, {ref: childRef});
}

class ZoomManager {
    xscale: PlotScale;
    yscale: PlotScale;

    transform: Transform2D;
    setXTrans: (val: Transform1D) => void;
    setYTrans: (val: Transform1D) => void;

    xTranslateExtent: Pair;
    yTranslateExtent: Pair;
    zoomExtent: Pair;

    state: "drag" | "idle" = "idle";
    dragStart: Pair = [0, 0];
    //transform: Transform2D = new Transform2D();

    elem: (SVGElement & HTMLElement) | null = null;

    listeners: EventListenerManager = new EventListenerManager();

    constructor(
        xscale: PlotScale, yscale: PlotScale, transform: Transform2D,
        setXTrans: (val: Transform1D) => void, setYTrans: (val: Transform1D) => void,
        xTranslateExtent: Pair, yTranslateExtent: Pair, zoomExtent: Pair,
    ) {
        this.xscale = xscale;
        this.yscale = yscale;

        this.transform = transform;
        this.setXTrans = setXTrans;
        this.setYTrans = setYTrans;

        this.xTranslateExtent = xTranslateExtent;
        this.yTranslateExtent = yTranslateExtent;
        this.zoomExtent = zoomExtent;
    }

    register(elem: HTMLElement & SVGElement) {
        this.listeners.addEventListener(elem, "mousedown", (ev) => this.mousedown(elem, ev));
        this.listeners.addEventListener(elem, "mousemove", (ev) => this.mousemove(elem, ev));
        this.listeners.addEventListener(elem, "mouseup", (ev) => this.mouseup(elem, ev));
        this.listeners.addEventListener(elem, "wheel", (ev) => this.wheel(elem, ev));
        this.elem = elem;
    }

    unregister(elem: HTMLElement & SVGElement) {
        this.elem = null;
        this.listeners.removeElementListeners(elem);
    }

    setTransform(transform: Transform2D): void {
        //console.log("Setting transform");
        this.transform = transform;
        const [xtrans, ytrans] = transform.to_1d();
        this.setXTrans(xtrans); this.setYTrans(ytrans);
    }

    updateTransform() {
        if (this.elem) {
            const elems = this.elem.getElementsByClassName('zoom');
            for (let i = 0; i < elems.length; i++) {
                (elems[i] as SVGElement).setAttribute("transform", this.transform.toString());
            }
        }
    }

    constrain(transform: Transform2D): Transform2D { 
        // taken from d3-zoom
        let currentExtent = [transform.invert().xlim(this.xscale.range), transform.invert().ylim(this.yscale.range)];
        // transform translateExtent to range coordinates
        let xExtent = this.xscale.transform(this.xTranslateExtent);
        let yExtent = this.yscale.transform(this.yTranslateExtent);

        // desired shift to bring extent to translateExtent
        let x0 = currentExtent[0][0] - xExtent[0];
        let x1 = currentExtent[0][1] - xExtent[1];
        let y0 = currentExtent[1][0] - yExtent[0];
        let y1 = currentExtent[1][1] - yExtent[1];

        return transform.pretranslate(
            // if x1 > x0, overconstrained, return the average.
            // otherwise, return 
            x1 > x0 ? (x0 + x1) / 2.0 : Math.min(0, x0) + Math.max(0, x1),
            y1 > y0 ? (y0 + y1) / 2.0 : Math.min(0, y0) + Math.max(0, y1),
        );
    }

    mousedown(elem: (HTMLElement & SVGElement), event: MouseEvent) {
        if (event.button != 0) { return; } // LMB only
        const [x, y] = this.transform.unapply(viewCoords(elem, [event.clientX, event.clientY]));

        this.state = "drag";
        this.dragStart = [x, y];

        this.listeners.addDocumentListener("mousemove", (ev) => this.mousemove(elem, ev));
        this.listeners.addDocumentListener("mouseup", (ev) => this.mouseup(elem, ev));

        event.stopPropagation(); event.preventDefault();
    }

    mousemove(elem: (HTMLElement & SVGElement), event: MouseEvent) {
        if (this.state != "drag") { return; }

        let [x, y] = this.transform.unapply(viewCoords(elem, [event.clientX, event.clientY]));
        let [deltaX, deltaY] = [x - this.dragStart[0], y - this.dragStart[1]];

        this.setTransform(this.constrain(this.transform.translate(deltaX, deltaY)));
        event.stopPropagation(); event.preventDefault();
    }

    mouseup(elem: (HTMLElement & SVGElement), event: MouseEvent) {
        this.state = "idle";
        this.listeners.removeDocumntListeners();
        event.stopPropagation(); event.preventDefault();
    }

    wheel(elem: (HTMLElement & SVGElement), event: WheelEvent) {
        const [x, y] = viewCoords(elem, [event.clientX, event.clientY]);
        const k = Math.exp(-event.deltaY / 500.0);
        const totalK = this.transform.k.map((oldK) => clamp(k * oldK, this.zoomExtent)) as Pair;

        const [origx, origy] = this.transform.unapply([x, y]);
        const transform = new Transform2D(totalK, [-origx * totalK[0] + x, -origy * totalK[1] + y]);

        this.setTransform(this.constrain(transform));
        event.stopPropagation(); event.preventDefault();
    } 
}

class EventListenerManager {
    private elemListeners: Map<HTMLElement, Array<[keyof HTMLElementEventMap, (this: HTMLElement, ev: any) => void, boolean | undefined | AddEventListenerOptions]>> = new Map();
    private docListeners: Array<[keyof DocumentEventMap, (this: Document, ev: any) => void, boolean | undefined | AddEventListenerOptions]> = [];
    private winListeners: Array<[keyof WindowEventMap, (this: Window, ev: any) => void, boolean | undefined | AddEventListenerOptions]> = [];

    addEventListener<K extends keyof HTMLElementEventMap>(
        elem: HTMLElement, type: K,
        listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => void,
        options?: boolean | AddEventListenerOptions,
    ) {
        elem.addEventListener(type, listener, options);

        let arr = this.elemListeners.get(elem);
        if (!arr) {
            this.elemListeners.set(elem, [[type, listener, options]]);
        } else {
            arr.push([type, listener, options]);
        }
    }

    removeElementListeners(elem: HTMLElement) {
        let arr = this.elemListeners.get(elem);
        if (arr) {
            for (const [type, listener, options] of arr) {
                elem.removeEventListener(type, listener, options);
            }
            this.elemListeners.set(elem, []);
        }
    }

    addDocumentListener<K extends keyof DocumentEventMap>(
        type: K, listener: (this: Document, ev: DocumentEventMap[K]) => void,
        options?: boolean | AddEventListenerOptions,
    ) {
        document.addEventListener(type, listener, options);
        this.docListeners.push([type, listener, options]);
    }

    removeDocumntListeners() {
        for (const [type, listener, options] of this.docListeners) {
            document.removeEventListener(type, listener, options);
        }
        this.docListeners = [];
    }

    addWindowListener<K extends keyof WindowEventMap>(
        type: K, listener: (this: Window, ev: WindowEventMap[K]) => void,
        options?: boolean | AddEventListenerOptions,
    ) {
        window.addEventListener(type, listener, options);
        this.winListeners.push([type, listener, options]);
    }

    removeWindowListeners() {
        for (const [type, listener, options] of this.docListeners) {
            window.removeEventListener(type, listener, options);
        }
        this.winListeners = [];
    }
}