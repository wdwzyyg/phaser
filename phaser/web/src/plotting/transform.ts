
import { Pair, clamp } from "./scale";

export default class Transform {
    readonly k: Pair = [1., 1.]
    readonly p: Pair = [0., 0.]

    constructor(scale: Pair = [1., 1.], offset: Pair = [0., 0.]) {
        this.k = scale;
        this.p = offset;
    }

    /**
     * Create a transform from the unit box to the given bounds
     */
    static toBounds(xlim: [number, number], ylim: [number, number]): Transform {
        const min: [number, number] = [xlim[0], ylim[0]];
        const range: [number, number] = [xlim[1] - xlim[0], ylim[1] - ylim[0]];
        return new Transform(range, min);
    }

    /**
     * Create a transform from the given bounds to the unit box
     */
    static fromBounds(xlim: [number, number], ylim: [number, number]): Transform {
        return Transform.fromBounds(xlim, ylim).invert();
    }

    pretranslate(x: number, y: number): Transform {
        return new Transform(
            this.k,
            [x*this.k[0] + this.p[0], y*this.k[1] + this.p[1]]
        );
    }

    translate(x: number, y: number): Transform {
        return new Transform(
            this.k,
            [x + this.p[0], y + this.p[1]]
        );
    }

    scale(kx: number, ky: number | undefined): Transform {
        if (ky === undefined) {
            ky = kx
        }
        return new Transform(
            [this.k[0] * kx, this.k[1] * ky], this.p
        );
    }

    apply(point: [number, number]): [number, number] {
        return [point[0] * this.k[0] + this.p[0], point[1] * this.k[1] + this.p[1]];
    }

    unapply(point: [number, number]): [number, number] {
        return [(point[0] - this.p[0])/this.k[0], (point[1] - this.p[1])/this.k[1]];
    }

    xlim(extent: [number, number] = [0.0, 1.0]): [number, number] {
        return [extent[0] * this.k[0] + this.p[0], extent[1] * this.k[0] + this.p[0]]
    }

    ylim(extent: [number, number] = [0.0, 1.0]): [number, number] {
        return [extent[0] * this.k[1] + this.p[1], extent[1] * this.k[1] + this.p[1]]
    }

    invert(): Transform {
        return new Transform(
            [1/this.k[0], 1/this.k[1]],
            [-this.p[0]/this.k[0], -this.p[1]/this.k[1]]
        );
    }

    compose(other: Transform) {
        return new Transform(
            [this.k[0]*other.k[0], this.k[1]*other.k[1]],
            [this.p[0]*other.k[0] + other.p[0], this.p[1]*other.k[1] + other.p[1]]
        );
    }

    toString(): string {
        return `translate(${this.p[0]}, ${this.p[1]}) scale(${this.k[0]}, ${this.k[1]})`;
    }
}