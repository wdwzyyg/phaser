
export type Pair = [number, number];
type ArrayOrNum = number | ReadonlyArray<number>;
type Writable<T extends ArrayOrNum> = T & (T extends number ? number : T extends readonly number[] ? number[] : never);


export function clamp<T extends ArrayOrNum>(val: T, extent: Pair): Writable<T> {
    if (typeof val == "number") {
        return Math.max(extent[0], Math.min(extent[1], val)) as Writable<T>;
    }
    return val.map((val) => clamp(val, extent)) as Writable<T>;
}


export class PlotScale {
    readonly domain: Pair
    readonly range: Pair

    constructor(domain: Pair, range: Pair) {
        this.domain = domain;
        this.range = range;
    }

    domainToUnit<T extends ArrayOrNum>(val: T, clip?: boolean): Writable<T> {
        if (clip) {
            val = clamp(val, this.domain);
        }
        if (typeof val == "number") {
            return (val - this.domain[0]) / (this.domain[1] - this.domain[0]) as Writable<T>;
        }
        return val.map((val) => (val - this.domain[0]) / (this.domain[1] - this.domain[0])) as Writable<T>;
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
            return val * (this.domain[1] - this.domain[0]) + this.domain[0] as Writable<T>;
        }
        return val.map((val) => val * (this.domain[1] - this.domain[0]) + this.domain[0]) as Writable<T>;
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

    scale(val: number): number {
        return val * (this.range[1] - this.range[0]) / (this.domain[1] - this.domain[0]);
    }

    unscale(val: number): number {
        return val * (this.domain[1] - this.domain[0]) / (this.range[1] - this.range[0]);
    }
}