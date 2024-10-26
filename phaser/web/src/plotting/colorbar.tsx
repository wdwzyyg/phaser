import React from "react";
import { useAtom } from 'jotai';
import * as d3_color from 'd3-color';

import { FigureContext, Axis, Plot } from "./plot"
import { PlotScale } from "./scale";

interface ColorBarProps {
    scale: string

    length?: number
    width?: number
}


export function Colorbar(props: ColorBarProps) {
    const fig = React.useContext(FigureContext);
    if (fig === undefined) {
        throw new Error("Component 'ColorBar' must be used inside a 'Figure'");
    }

    if (!fig.scales.has(props.scale)) {
        throw new Error("Invalid scale passed to component 'ColorBar'");
    }
    const [scale, setScale] = useAtom(fig.scales.get(props.scale)!);

    const width = props.width ?? 20;
    const height = props.length ?? 200;

    let xaxis: Axis = {
        scale: new PlotScale([0, width], [0, width]),
        translateExtent: [-Infinity, Infinity],
        show: false,
    };
    let yaxis: Axis = {
        scale: new PlotScale(scale.scale?.domain()!, [0, height]),
        label: scale.label,
        translateExtent: [-Infinity, Infinity],
        show: true,
    };

    const canvasRef: React.MutableRefObject<HTMLCanvasElement | null> = React.useRef(null);

    React.useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !scale.scale) return;

        const interp = scale.scale.interpolator();

        const ctx = canvas.getContext('2d')!;

        const imageData = ctx.createImageData(width, height);
        let i = 0;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let color = d3_color.color(interp(y / height))!.rgb();
                if (y == height - 1 && x == 0) console.log(`color: ${color}`);
                imageData.data[i] = color.r;
                imageData.data[i + 1] = color.g;
                imageData.data[i + 2] = color.b;
                imageData.data[i + 3] = 255;
                //imageData[i + 3] = color.opacity;
                i += 4;
            }
        }
        ctx.putImageData(imageData, 0, 0);
    }, []);

    return <Plot xaxis={xaxis} yaxis={yaxis} yaxis_pos="right">
        <foreignObject x={0} y={0} width={width} height={height}>
            <canvas width={width} height={height} ref={canvasRef}></canvas>
        </foreignObject>
    </Plot>;
}