"""
Code for analysis of ptychographic reconstructions
"""

from dataclasses import dataclass
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike

ScalarT = t.TypeVar('ScalarT', bound=numpy.generic)


@dataclass
class FRCResult:
    """Fourier Ring Correlation result"""

    freq: NDArray[numpy.float64]
    """List of frequencies in k-space (units of 1/px)"""

    corr: NDArray[numpy.float64]
    """Fourier ring correlations corresponding to `freq`/`r`"""

    r: NDArray[numpy.float64]
    """List of radii in k-space (proportional to `freq`, units of reciprocal space px)"""

    r_count: NDArray[numpy.int64]
    """Count of pixels contained in each radius bin"""

    single_image: bool = False
    """
    Whether the FRC was calculated from a single image checkerboard.
    This affects the final resolution calculation.
    """

    def half_bit_threshold(self) -> NDArray[numpy.float64]:
        """Return the half-bit FRC threshold"""
        snr = 2**0.5 - 1
        return self.snr_threshold(snr)

    def const_threshold(self, corr: float = 0.5) -> NDArray[numpy.float64]:
        """Return the FRC threshold for a constant correlation `corr`."""
        return numpy.full_like(self.freq, corr)

    def snr_threshold(self, snr: float) -> NDArray[numpy.float64]:
        """Return the FRC threshold for the given signal-to-noise ratio `snr`."""
        n = 1/numpy.sqrt(self.r_count)

        return (snr + (2*numpy.sqrt(snr) + 1)*n) / (1 + snr + 2*numpy.sqrt(snr)*n)

    def intersect(self, threshold: ArrayLike) -> t.Tuple[float, float]:
        """
        Return the (freq, corr) point where the FRC crosses `threshold`.
        """
        diff = self.corr - numpy.array(threshold, dtype=float)
        lastdiff = numpy.roll(diff, shift=1)
        try:
            # find negative zero crossing
            i = numpy.nonzero((diff[1:] <= 0) & (lastdiff[1:] > 0))[0][0]
        except IndexError:
            raise ValueError("No crossing found when evaluating FRC resolution")

        # calculate linear intersection between `frc.corr` and `threshold`
        # slope of line
        m = (diff[i+1] - diff[i]) / (self.freq[i+1] - self.freq[i])
        # frequency difference to reach x-axis
        x_d = -diff[i]/m  
        # slope of self line
        m_corr = (self.corr[i+1] - self.corr[i]) / (self.freq[i+1] - self.freq[i])
        # (x, y)
        return (float(self.freq[i] + x_d), float(self.corr[i] + x_d * m_corr))

    def resolution(self, threshold: ArrayLike, pixel_size: float = 1.0) -> float:
        """
        Return the FRC resolution using threshold `threshold`.

        `pixel_size` is the pixel size in units/px.
        Returns a value in the same units as `pixel_size`.
        """
        freq, _ = self.intersect(threshold)
        if self.single_image:
            # upscale frequency (to undo our downsampling)
            freq *= float(numpy.sqrt(2))
        return pixel_size / freq


def fourier_ring_correlate(img: NDArray[numpy.generic], *, r_spacing: float = 1.0,
                           inscribed: bool = True, hann_window: bool = True) -> FRCResult:
    """
    Fourier-ring correlate a single image `img`.

    This uses `checkerboard_split` to split an image up into two pairs,
    performs FRC on each, and averages the results.
    """
    subimages = checkerboard_split(img)
    # pair up along diagonals
    pair1, pair2 = (subimages[0], subimages[3]), (subimages[1], subimages[2])
    result1 = fourier_ring_correlate_pair(*pair1, r_spacing=r_spacing, inscribed=inscribed, hann_window=hann_window)
    result2 = fourier_ring_correlate_pair(*pair2, r_spacing=r_spacing, inscribed=inscribed, hann_window=hann_window)

    return FRCResult(
        freq=result1.freq, corr=(result1.corr + result2.corr) / 2., r=result1.r, r_count=result1.r_count,
        single_image=True
    )


def fourier_ring_correlate_pair(img1: NDArray[numpy.generic], img2: NDArray[numpy.generic], *,
                           r_spacing: float = 1.0, inscribed: bool = True, hann_window: bool = True) -> FRCResult:
    """
    Fourier-ring correlate `img1` and `img2`.

    Parameters:
      `r_spacing`: The spacing of ring radii to return correlations at.
      `inscribed`: If specified, frequencies are returned only to the edge of k-space (1/2 px^-1).
                   Otherwise, frequencies are returned to the corner of k-space.
      `hann_window`: If specified, a Hann window is applied to each image before cross-correlating.
    """

    # convert images to float
    img1f = img1.astype(numpy.float64)
    img2f = img2.astype(numpy.float64)

    # apply hann window if desired
    if hann_window:
        from skimage.filters import window
        win = numpy.asarray(window('hann', img1f.shape), dtype=numpy.float64)
        img1f *= win
        img2f *= win

    # take FFTs, and find magnitudes
    fft1 = numpy.fft.fftshift(numpy.fft.fft2(img1f), axes=(-2, -1))
    fft2 = numpy.fft.fftshift(numpy.fft.fft2(img2f), axes=(-2, -1)).conj()
    fft1_mag = fft1.real**2 + fft1.imag**2
    fft2_mag = fft2.real**2 + fft2.imag**2

    # calculate radii in reciprocal space
    y, x = numpy.indices(img1.shape[-2:])
    c_y, c_x = tuple(int(s//2) for s in img1.shape[-2:])
    r = numpy.sqrt((y - c_y)**2 + (x - c_x)**2)

    # assign radii to bins
    r_i = numpy.floor(r / r_spacing).astype(numpy.int_).ravel()
    # sum correlation and magnitude along each bin
    real_count = numpy.bincount(r_i, (fft1 * fft2).real.ravel())
    mag1_count = numpy.bincount(r_i, fft1_mag.ravel())
    mag2_count = numpy.bincount(r_i, fft2_mag.ravel())
    r_count = numpy.bincount(r_i)

    with numpy.errstate(invalid='ignore', divide='ignore'):
        # compute correlations
        vals = numpy.abs(real_count) / numpy.sqrt(mag1_count * mag2_count)
    # 0/0 => 1.0, 1/0 => 1.0
    vals = numpy.nan_to_num(vals, posinf=1.0, nan=1.0)

    # calculate radii and frequencies sampled at
    rs = numpy.linspace(0., len(vals) * r_spacing, len(vals), endpoint=False)
    freq = rs / numpy.sqrt(numpy.prod(img1.shape[-2:]))

    if inscribed:
        # crop to inner radii
        n_r = int(numpy.floor(min(c_y, c_x) / r_spacing))
        return FRCResult(freq=freq[:n_r], corr=numpy.abs(vals[:n_r]), r=rs[:n_r], r_count=r_count[:n_r])

    return FRCResult(freq=freq, corr=numpy.abs(vals), r=rs, r_count=r_count)


def checkerboard_split(img: NDArray[ScalarT]) -> t.Tuple[NDArray[ScalarT], NDArray[ScalarT], NDArray[ScalarT], NDArray[ScalarT]]:
    """
    Split image into 4 interlaced subimages (upper left, upper right, lower left, lower right).

    This is useful for single-image Fourier Ring Correlation
    """
    shape = img.shape[-2:]
    # crop image divisible by two
    shape = tuple(s - s%2 for s in shape)
    img = img[..., *(slice(0, s) for s in shape)]

    return tuple(
        img[..., slice(row_start, shape[0], 2), slice(col_start, shape[1], 2)]
        for row_start in (0, 1)
        for col_start in (0, 1)
    )  # type: ignore


def fourier_correlate(img1: NDArray[ScalarT], img2: NDArray[ScalarT], *, hann_window: bool = True) -> NDArray[numpy.complex128]:
    """
    Cross-correlate `img1` and `img2` in Fourier space.

    If `hann_window`, a Hann window is applied to each image before cross-correlating.
    """

    img1f = img1.astype(numpy.float64)
    img2f = img2.astype(numpy.float64)

    if hann_window:
        from skimage.filters import window
        win = numpy.asarray(window('hann', img1f.shape), dtype=numpy.float64)
        img1f *= win
        img2f *= win

    fft1 = numpy.fft.fftshift(numpy.fft.fft2(img1f), axes=(-2, -1))
    fft2 = numpy.fft.fftshift(numpy.fft.fft2(img2f), axes=(-2, -1)).conj()
    fft1_mag = fft1.real**2 + fft1.imag**2
    fft2_mag = fft2.real**2 + fft2.imag**2

    with numpy.errstate(divide='ignore', invalid='ignore'):
        corr = fft1 * fft2 / numpy.sqrt(fft1_mag * fft2_mag)
    return numpy.nan_to_num(corr, nan=1., posinf=0.)