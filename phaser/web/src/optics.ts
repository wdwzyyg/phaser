import * as np from 'wasm-array';

function fftfreq(size: number, n: number): np.NArray {
    if (n % 2 == 0) {
        // even
        return np.ifftshift(np.linspace(-n/2 / size, (n/2 - 1) / size, n, 'float32'))
    } else {
        // odd
        return np.ifftshift(np.linspace(-(n - 1)/2 / size, (n - 1)/2 / size, n, 'float32'))
    }
}

export function make_recip_grid(size: [number, number], n: number = 1024): [np.NArray, np.NArray] {
    let ky = fftfreq(size[0], n);
    let kx = fftfreq(size[1], n);
    [ky, kx] = np.meshgrid(ky, kx);
    return [ky, kx];
}

export function make_focused_probe(ky: np.NArray, kx: np.NArray, wavelength: number,
                                   aperture: number, defocus: number = 0.0): np.NArray {
    const lambda = np.array(wavelength, 'float32');

    const theta2 = np.expr`(${ky}**2 + ${kx}**2) * ${lambda}**2`;
    const phase = np.expr`${defocus}/(2.*${lambda}) * ${theta2}`;

    const mask = np.expr`${theta2} <= (${aperture}*1e-3)**2`;
    let probe = np.expr`exp(-2.j*pi * ${phase})`.astype('complex64');
    probe = np.expr`${probe} * ${mask}`;
    let probe_int = np.sqrt(np.sum(np.abs(probe)));
    return np.fft2shift(np.ifft2(np.expr`${probe} / ${probe_int}`));
}

export function fresnel_propagator(ky: np.NArray, kx: np.NArray, wavelength: number,
                                   delta_z: number, tilt: [number, number] = [0., 0.]): np.NArray {
    const k2 = np.expr`${ky}**2 + ${kx}**2`;

    const tiltx = np.expr`tan(${tilt[0]} * 1e-3)`.astype(k2.dtype);
    const tilty = np.expr`tan(${tilt[1]} * 1e-3)`.astype(k2.dtype);

    const phase = np.expr`-1.j*pi * ${delta_z} * (
        ${wavelength} * ${k2} - 2.*(${kx}*${tiltx} + ${ky}*${tilty})
    )`.astype(k2.dtype);
    return np.expr`exp(-1.j*pi * ${phase})`.astype('complex64');
}
