/*
import * as d3 from 'd3';

const prefixes: Map<number, string> = new Map([
    [-18, "a"],
    [-15, "f"],
    [-12, "p"],
    [-9, "n"],
    [-6, "µ"],
    [-3, "m"],
    [0, ""],
    [3, "k"],
    [6, "M"],
    [9, "G"],
    [12, "T"],
    [15, "P"],
    [18, "E"],
]);

// return quotient and modulus of two numbers
function divmod(dividend: number, divisor: number): [number, number] {
    const mod = dividend % divisor;
    const quot = (dividend - mod) / divisor;
    return [quot, mod];
}

type ScaleDecision = "same" | "none" | [number, string];

export class Scalebar<E extends SVGElement> {
    parent: d3.Selection<E, undefined, null, undefined>;
    bar: d3.Selection<SVGGElement, undefined, null, undefined>;

    parentWidth: number;
    parentHeight: number;

    unit: string = "m";
    plotScale: number = 1;

    // min and max size of the scalebar
    minFrac: number = 0.2;
    maxFrac: number = 0.5;

    mantissas: Array<number> = [1, 2, 5];

    currentSize: number = 0;
    currentText: string = "0 m";

    margin: number = 10;

    constructor(
        parent: d3.Selection<E, undefined, null, undefined>,
        parentWidth: number, parentHeight: number, unit: string = "m",
        plotScale: number = 1.0,
    ) {
        const scalebarHeight = 15;
        const scalebarRadius = 4;
        const width = 10;

        const scalebar = parent.append("g")
            .attr("class", "scalebar")
            .attr("display", "none");

        const x = parentWidth - this.margin - width;
        const y = parentHeight - this.margin - scalebarHeight;

        scalebar.append("rect")
            .attr("rx", `${scalebarRadius}`)
            .attr("x", `${x}`)
            .attr("y", `${y}`)
            .attr("height", `${scalebarHeight}`)
            .attr("width", `${width}`);

        scalebar.append("text")
            .attr("x", `${x + width/2}`)
            .attr("y", `${y}`)
            .attr("text-anchor", "middle")
            .attr("dy", "-5")
            .html("1 nm");

        this.parent = parent;
        this.bar = scalebar;
        this.parentWidth = parentWidth;
        this.parentHeight = parentHeight;
        this.unit = unit;
        this.plotScale = plotScale;
    }

    private update(frameWidth: number) {
        const width = this.currentSize / frameWidth * this.parentWidth;
        const x = this.parentWidth - this.margin - width;

        this.bar.selectChild("rect")
            .attr("x", `${x}`)
            .attr("width", `${width}`);

        this.bar.selectChild("text")
            .attr("x", `${x + width/2}`);
    }

    private decideScale(
        frameWidth: number,
        force: boolean = false,
    ): ScaleDecision {
        let frac = this.currentSize / frameWidth;
        if (!force && this.minFrac <= frac && frac <= this.maxFrac) {
            return "same";
        }

        const orderOfMagnitude = Math.floor(Math.log10(this.maxFrac * frameWidth));

        const [engOrder, rem] = divmod(orderOfMagnitude, 3);
        const prefix = prefixes.get(3 * engOrder);
        if (prefix === undefined) {
            return "none";
        }

        const mult = Math.pow(10, rem);

        for (const mantissa of this.mantissas) {
            const size = Math.pow(10, orderOfMagnitude) * mantissa;
            frac = size / frameWidth;
            if (this.minFrac <= frac && frac <= this.maxFrac) {
                const val = mantissa * mult;
                const text = `${val} ${prefix}${this.unit}`

                return [size, text];
            }
        }
        return "none";
    }

    scale(
        frameWidth: number | d3.ScaleLinear<number, number, never>,
        force: boolean = false,
    ) {
        if (typeof frameWidth != "number") {
            let [min, max] = frameWidth.domain();
            frameWidth = Math.abs(max - min);
        }
        frameWidth *= this.plotScale;

        const result = this.decideScale(frameWidth, force);

        if (result === "none") {
            this.bar.attr("display", "none");
            return;
        }
        if (result === "same") {
            this.update(frameWidth);
            return;
        }

        const [size, text] = result;
        this.currentSize = size;
        this.bar.attr("display", "initial");
        this.bar.selectChild("text").html(text);
        this.update(frameWidth);
    }
}
*/