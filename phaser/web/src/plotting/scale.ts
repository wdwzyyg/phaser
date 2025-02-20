
import * as d3_array from "d3-array";
import { Transform1D } from "./transform";

export type Pair = [number, number];
export type ArrayOrNum = number | ReadonlyArray<number>;
export type Writable<T extends ArrayOrNum> = T & (T extends number ? number : T extends readonly number[] ? number[] : never);
type TransformFn = (val: number) => number;


export function clamp<T extends ArrayOrNum>(val: T, extent: Pair): Writable<T> {
    if (typeof val == "number") {
        return Math.max(extent[0], Math.min(extent[1], val)) as Writable<T>;
    }
    return val.map((val) => clamp(val, extent)) as Writable<T>;
}


function transform<T extends ArrayOrNum>(val: T, transform: TransformFn): Writable<T> {
    if (typeof val == "number") {
        return transform(val) as Writable<T>;
    }
    return val.map(transform) as Writable<T>;
}


export function isClose<T extends ArrayOrNum>(left: T, right: T, rtol: number = 1e-6, atol: number = 1e-6): boolean {
    if (typeof left == "number") {
        return typeof right == "number" && (
            Math.abs(left - right) < Math.max(rtol * Math.max(Math.abs(left), Math.abs(right)), atol)
        );
    }
    if (typeof right == "number" || left.length != right.length) return false;
    for (let i = 0; i < left.length; i++) {
        if (!isClose(left[i], right[i], rtol, atol)) return false;
    }
    return true;
}

const id: TransformFn = (v) => v;


export class PlotScale {
    readonly domain: Pair
    readonly linDomain: Pair
    readonly range: Pair

    readonly fwd_transform: TransformFn
    readonly rev_transform: TransformFn

    constructor(domain: Pair, range: Pair, fwd_transform?: TransformFn, rev_transform?: TransformFn) {
        this.domain = domain;
        this.range = range;
        this.fwd_transform = fwd_transform ?? id;
        this.rev_transform = rev_transform ?? id;

        this.linDomain = this.domain.map(this.fwd_transform) as Pair;
    }

    toString = () => `PlotScale { domain: [${this.domain[0]}, ${this.domain[1]}] range: [${this.range[0]}, ${this.range[1]}] }`;

    isClose = (other: PlotScale) => (
        isClose(this.domain, other.domain) && isClose(this.range, other.range) &&
        this.fwd_transform == other.fwd_transform && this.rev_transform == other.rev_transform
    );

    linDomainSize = () => Math.abs(this.linDomain[1] - this.linDomain[0]);
    rangeSize = () => Math.abs(this.range[1] - this.range[0]);

    domainToUnit<T extends ArrayOrNum>(val: T, clip?: boolean): Writable<T> {
        val = transform(val, this.fwd_transform);
        if (clip) {
            val = clamp(val, this.domain);
        }
        if (typeof val == "number") {
            return (val - this.linDomain[0]) / (this.linDomain[1] - this.linDomain[0]) as Writable<T>;
        }
        return val.map((val) => (val - this.linDomain[0]) / (this.linDomain[1] - this.linDomain[0])) as Writable<T>;
    }

    rangeToUnit<T extends ArrayOrNum>(val: T, clip?: boolean): Writable<T> {
        if (clip) {
            val = clamp(val, this.range);
        }
        if (typeof val == "number") {
            return (val - this.range[0]) / (this.range[1] - this.range[0]) as Writable<T>;
        }
        return val.map((val) => (val - this.range[0]) / (this.range[1] - this.range[0])) as Writable<T>;
    }

    domainFromUnit<T extends ArrayOrNum>(val: T, clip?: boolean): Writable<T> {
        if (clip) {
            val = clamp(val, [0.0, 1.0]);
        }
        if (typeof val == "number") {
            return this.rev_transform(
                val * (this.linDomain[1] - this.linDomain[0]) + this.linDomain[0],
            ) as Writable<T>;
        }
        return val.map((val) => val * (this.linDomain[1] - this.linDomain[0]) + this.linDomain[0])
            .map(this.rev_transform) as Writable<T>;
    }

    rangeFromUnit<T extends ArrayOrNum>(val: T, clip?: boolean): Writable<T> {
        if (clip) {
            val = clamp(val, [0.0, 1.0]);
        }
        if (typeof val == "number") {
            return val * (this.range[1] - this.range[0]) + this.range[0] as Writable<T>;
        }
        return val.map((val) => val * (this.range[1] - this.range[0]) + this.range[0]) as Writable<T>;
    }

    transform<T extends ArrayOrNum>(val: T, clip?: boolean): Writable<T> {
        return this.rangeFromUnit(this.domainToUnit(val, clip), false);
    }

    untransform<T extends ArrayOrNum>(val: T, clip?: boolean): Writable<T> {
        return this.domainFromUnit(this.rangeToUnit(val, clip), false);
    }

    linScale(val: number): number {
        return val * (this.range[1] - this.range[0]) / (this.domain[1] - this.domain[0]);
    }

    linUnscale(val: number): number {
        return val * (this.domain[1] - this.domain[0]) / (this.range[1] - this.range[0]);
    }

    applyTransform(transform: Transform1D): PlotScale {
        return new PlotScale(
            this.untransform(transform.unapply(this.range)),
            this.range,
            this.fwd_transform,
            this.rev_transform
        );
    }

    ticks(n: number): Array<number> {
        return d3_array.ticks(...this.domain, n);
    }

    pad_frac(frac: number): PlotScale {
        return new PlotScale(
            this.domainFromUnit([-frac, 1.0 + frac], false) as Pair,
            this.range, this.fwd_transform, this.rev_transform
        );
    }
}

export class LogPlotScale extends PlotScale {
    readonly base: number

    constructor(domain: Pair, range: Pair, base: number = 10) {
        const log_base = Math.log(base)
        super(domain, range, (val) => Math.log(val) / log_base, (val) => Math.exp(val * log_base));
        this.base = base;
    }

    ticks(n: number): Array<number> {
        const ret = d3_array.ticks(...this.linDomain, n).map(this.rev_transform);
        return ret;
    }

    applyTransform(transform: Transform1D): LogPlotScale {
        return new LogPlotScale(
            this.untransform(transform.unapply(this.range)),
            this.range,
            this.base,
        );
    }

    pad_frac(frac: number): LogPlotScale {
        return new LogPlotScale(
            this.domainFromUnit([-frac, 1.0 + frac], false) as Pair,
            this.range, this.base
        );
    }
}