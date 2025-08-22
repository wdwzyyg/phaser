from __future__ import annotations
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from itertools import product
import typing as t

import numpy
from pydantic import Field, validator, root_validator, PrivateAttr
from pydantic.types import NonNegativeInt, PositiveInt

from .models import ModelConfig, ValueOrList, WrapperModel
from .metadata import Metadata, AnyMetadata


class FormatError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


def _iter_dict(params: t.Dict[str, t.Any], keys: t.Iterable[str], sparse: bool = False) -> t.Iterable[t.Dict[str, t.Any]]:
    """
    Return copies of 'params', iterating through each key in `keys`.
    If `sparse`, sparse (plus-shaped) combinations of iterators are returned.
    The first value of each iterator is used as the default
    Otherwise, dense (grid-shaped, cartesian product) combinations are returned.
    """
    # filter keys which aren't in params
    keys = tuple(k for k in keys if k in params)
    vals = tuple(params[k] for k in keys)

    # dense (cartesian product)
    if not sparse:
        # dense (cartesian product)
        for update_vals in product(*vals):
            d = params.copy()
            d.update(zip(keys, update_vals))
            yield d

        return

    # TODO sparse update must be done with deep keys
    if math.prod(len(v) for v in vals) == 0:
        # empty, return
        return

    # construct iterator objects for every non-missing pair
    # propagate 'sparse' to child iterators
    iters = tuple(v.iter(sparse=True) if hasattr(v, 'iter') else iter(v) for v in vals)

    # use the first member of each iterator to define the center point
    base_vals = tuple((k, next(it)) for (k, it) in zip(keys, iters))
    params = params.copy()
    params.update(base_vals)
    # start by yielding center point
    yield params

    # perturb one key at a time
    for (update_key, it) in zip(keys, iters):
        for update_val in it:
            d = params.copy()
            d[update_key] = update_val
            yield d


@dataclass
class SaveRecord:
    """Stores which (and how many) files have already been saved, and avoids duplicates."""
    paths: t.Dict[str, int] = field(default_factory=dict)
    names: t.Dict[str, int] = field(default_factory=dict)
    i: int = 0

    def deduplicate_path(self, base_path: str) -> str:
        if base_path in self.paths:
            # handle duplicate paths
            self.paths[base_path] += 1
            base_path += f'_{self.paths[base_path]}'
        else:
            self.paths[base_path] = 0
        return base_path

    def deduplicate_name(self, base_name: str) -> str:
        if base_name in self.names:
            # handle duplicate names
            self.names[base_name] += 1
            base_name += f'_{self.names[base_name]}'
        else:
            self.names[base_name] = 0
        return base_name


class QueueParams(ModelConfig):
    lockfile: bool = False
    """If true, writes a lock file which prevents multiple reconstructions from running."""


class DetectorParams(ModelConfig):
    name: t.Literal['empad', 'empad_lebeau'] = 'empad_lebeau'
    """Detector type"""
    check_2_detpos: t.Literal[None] = None
    data_prefix: str = ""
    binning: bool = False
    upsampling: t.Union[NonNegativeInt, bool] = False
    burst_frames: PositiveInt = 1
 
    circ_mask: t.Optional[int] = None
    """If specified, applies a circular mask of the radius to each diffraction pattern."""

    bg_sub: t.Union[bool, float] = False
    """Subtract background intensity from diffraction patterns."""

    crop: t.Optional[t.Tuple[int, int, int, int]] = None
    """
    Scan dimensions to crop to (min_x, max_x, min_y, max_y).
    Matlab-style slicing, so [1, 128, 1, 128] == an entire 128x128 scan
    """

    step: t.Optional[t.Tuple[int, int]] = None
    """
    Only use every n x m (x, y) scan positions. Useful for testing reconstruction at multiples of step size.
    """

    tile: t.Optional[t.Tuple[int, int]] = None
    """
    Tile the scan n x m (x, y) times prior to reconstruction. Useful for increasing the size of simulated data.

    Care must be taken to ensure the scan is periodic
    """

    fill_nan: bool = False
    """
    Whether to check for and fill NaN values in raw dataset
    """

    sim: bool = False
    """Whether data is simulated (single-electron intensity)"""

    beam_dose: t.Optional[float] = None
    """Total electron dose (in e/A^2) to scale simulated data by. Overrides `beam_current` if specified."""
    beam_current: float = 30.
    """Beam current (in pA) to scale simulated data by."""

    psf_sigma: t.Optional[float] = None
    """
    Apply Gaussian point spread function to the raw data before reconstruction.
    Mainly useful for simulating a non-ideal detector.
    """

    poisson: bool = False
    """
    Apply Poisson noise to the raw data before reconstruction.
    If using this with simulated data, make sure `beam_current`
    is set correctly.
    """


class PrepareParams(ModelConfig):
    data_preparator: t.Literal['matlab_aps'] = 'matlab_aps'
    auto_prepare_data: bool = True
    """If true, prepare dataset from raw measurements"""
    force_preparation_data: bool = False
    """Prepare dataset from raw measurements even if prepared data exists"""
    store_prepared_data: bool = True
    """Store prepared data to HDF5"""
    prepare_data_function: t.Literal[""] = ""
    #auto_center_data: bool = False
    #"""Try to automatically center cbed center-of-mass"""


class RasterScanParams(ModelConfig):
    type: t.Literal['raster'] = 'raster'

    nx: PositiveInt
    """Number of scan positions in x."""
    ny: PositiveInt
    """Number of scan positions in y"""

    step: float = Field(None)
    """Scan step size (angstroms)"""
    step_size_x: float = Field(None)
    """Scan X step size (angstroms)"""
    step_size_y: float = Field(None)
    """Scan Y step size (angstroms)"""

    @root_validator
    def validate_scan_step(cls, values):
        if values.get('step') is not None:
            if values.get('step_size_x') is None:
                values['step_size_x'] = values['step']
            if values.get('step_size_y') is None:
                values['step_size_y'] = values['step']
        elif values.get('step_size_x') is not None:
            values['step'] = values['step_size_x']
            if values.get('step_size_y') is None:
                values['step_size_y'] = values['step_size_x']
        else:
            raise ValueError("Missing parameter 'step'/'step_size_x'")
        return values

    custom_flip: t.Tuple[bool, bool, bool] = (False, False, False)
    """Custom data flip (left-right, up-down, tranpose)."""

    step_randn_offset: float = 0.
    """Random offset to apply to raster positions (relative to step size?)"""

    roi: t.Optional[t.Tuple[int, int, int, int]] = None
    """
    Real-space dimensions to crop to (min_x, max_x, min_y, max_y).
    Currently broken
    """


class CustomScanParams(ModelConfig):
    type: t.Literal['custom_GPU'] = 'custom_GPU'

    custom_positions_source: Path
    """Niter.mat file to load positions from. Can be specified relative to base_path."""


class ListScanParams(ModelConfig):
    type: t.Literal['list'] = 'list'

    scan_positions: t.List[t.Tuple[float, float]]
    """List of scan positions."""

    def is_default(self) -> bool:
        return False

    def apply_metadata(self, metadata: t.Optional[Metadata] = None) -> ListScanParams:
        return self

    def iter(self, sparse: bool = False) -> t.Iterator[ListScanParams]:
        yield self

    def __iter__(self) -> t.Iterator[ListScanParams]:
        return self.iter()


class ModelParams(ModelConfig):
    object_type: t.Literal['rand', 'amplitude'] = 'rand'

    probe_alpha_max: float
    """Model probe convergence angle (mrad)"""
    probe_df: float
    """Model probe defocus (angstroms, overfocus is negative)."""
    probe_C3: float = 0.
    """Model probe C3 (angstrom)"""
    probe_C5: float = 0.
    """Model probe C5 (angstrom)"""
    probe_C7: float = 0.
    """Model probe C7 (angstrom)"""
    probe_f_a2: float = 0.
    probe_theta_a2: float = 0.
    probe_f_a3: float = 0.
    probe_theta_a3: float = 0.
    probe_f_c3: float = 0.
    probe_theta_c3: float = 0.


class ModelParamSet(ModelParams):
    probe_alpha_max: ValueOrList[float]
    probe_df: ValueOrList[float]

    _iterable_keys = ('probe_df', 'probe_alpha_max')
    _len: t.Optional[int] = PrivateAttr(None)

    def __len__(self) -> int:
        if self._len is None:
            self._len = math.prod(len(getattr(self, k) or (None,)) for k in self._iterable_keys)
        return self._len

    def iter(self, sparse: bool = False) -> t.Iterator[ModelParams]:
        params = dict(filter(lambda t: t[0] in self.__fields_set__, self.__dict__.items()))

        for d in _iter_dict(params, self._iterable_keys, sparse):
            yield ModelParams.parse_obj(d)

    def __iter__(self) -> t.Iterator[ModelParams]:
        return self.iter()


class IOParams(ModelConfig):
    default_mask_file: str = ""
    default_mask_type: t.Literal['binary', 'indices'] = 'binary'
    file_compression: NonNegativeInt = 0
    data_compression: PositiveInt = 3
    load_prep_pos: bool = False
    """Load positions from prepared data, ignoring metadata"""


class SlicesInterp(ModelConfig):
    old_n: PositiveInt
    """# of layers in the last engine to interpolate from."""

    new_n: PositiveInt
    """# of layers in the new engine to interpolate to."""

    def layers(self) -> t.List[float]:
        return list(numpy.linspace(1, self.old_n, self.new_n, endpoint=True))


class Slices(ModelConfig):
    n: PositiveInt
    """Number of layers. 1 for single-slice ptychography."""
    delta_z: float = Field(None)
    """Slice thickness (in angstroms)."""
    thickness: float = Field(None)
    """Total object thickness (in angstroms)."""

    def delta_zs(self) -> t.List[float]:
        return [self.delta_z] * self.n

    def zs(self) -> t.List[float]:
        return [self.delta_z * i for i in range(self.n)]

    @root_validator
    def validate_slices(cls, values: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        if values.get('delta_z') is not None:
            if values.get('thickness') is not None:
                raise ValueError("'delta_z' and 'thickness' can't both be specified.")
            values['thickness'] = values['delta_z'] * values['n']
            #values['delta_z'] = ValueOrList[float].parse_obj([values['delta_z']] * values['n'])
        elif values.get('thickness') is not None:
            #values['delta_z'] = ValueOrList[float].parse_obj([values['thickness'] / values['n']] * values['n'])
            values['delta_z'] = values['thickness'] / values['n']
        else:
            raise ValueError("Either 'delta_z' or 'thickness' must be specified.")

        return values


class SlicesSet(ModelConfig):
    n: ValueOrList[PositiveInt]
    """Number of layers. 1 for single-slice ptychography."""
    delta_z: ValueOrList[float] = Field(None)
    """Slice thickness (in angstroms)."""
    thickness: ValueOrList[float] = Field(None)
    """Total thickness (in angstroms)."""

    @root_validator
    def validate_slices(cls, values: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        if values.get('delta_z') is None and values.get('thickness') is None:
            raise ValueError("Either 'delta_z' or 'thickness' must be specified.")
        elif values.get('delta_z') is not None and values.get('thickness') is not None:
            raise ValueError("'delta_z' and 'thickness' can't both be specified.")
        return values

    _iterable_keys = ('n', 'delta_z', 'thickness')
    _len: t.Optional[int] = PrivateAttr(None)

    def __len__(self) -> int:
        if self._len is None:
            self._len = math.prod(len(getattr(self, k) or (None,)) for k in self._iterable_keys)
        return self._len

    def iter(self, sparse: bool = False) -> t.Iterator[Slices]:
        params = dict(filter(lambda t: t[0] in self.__fields_set__, self.__dict__.items()))
        for d in _iter_dict(params, self._iterable_keys, sparse):
            yield Slices.parse_obj(d)

    def __iter__(self) -> t.Iterator[Slices]:
        return self.iter()


class EngineParams(ModelConfig):
    name: t.Literal['GPU_MS'] = 'GPU_MS'
    """Reconstruction engine. GPU_MS = GPU multislice"""
    fout: str = "{method}_{opt_errmetric}_p{probe_modes}_g{grouping}_step{i}"
    """Output path. Can be specified relative to base_path."""
    use_gpu: bool = True
    """Use GPU for reconstruction."""
    keep_on_gpu: bool = True
    """Keep data + projections on GPU"""
    compress_data: bool = False
    """Use online memory compression"""
    gpu_id: t.Optional[int] = None
    """GPU id to use"""
    check_gpu_load: bool = True
    """Check GPU memory before starting engine"""

    number_iterations: PositiveInt = 200
    """Number of iterations for selected method"""

    save_results_every: t.Optional[PositiveInt] = 10
    """Save partial results every n iterations."""
    plot_results_every: t.Optional[PositiveInt] = None
    """Plot partial results every n iterations."""

    auto_center_data: bool = False
    """Center diffraction patterns using average center of mass."""

    slices: t.Optional[Slices] = None

    delta_z: t.List[float] = Field(None)
    """List of slice z"""

    @validator('delta_z')
    def validate_delta_z(cls, v, values: t.Dict[str, t.Any], **kwargs) -> t.Any:
        if v is not None:
            return v
        if values.get('slices') is not None:
            return values['slices'].delta_zs()
        raise ValueError("Either 'slices' or 'delta_z' must be specified.")

    # multislice options
    regularize_layers: float = Field(1., ge=0., le=1.)
    """Apply regularization on the reconstructed object layers. 0 = no regularization, 0.01 = weak regularization."""
    init_layer_preprocess: t.Literal['all', 'avg', 'avg1', 'interp'] = 'all'
    """
    How to preprocess layers from previous reconstruction step.
    'all' (default): No pre-processing
    'avg': Average layers together
    'avg1': Average layers and keep one (the rest initialized with 'init_layer_append_mode')
    'interp': Interpolate old object layers to 'init_slices'/'init_layer_interp'.
    """

    init_slices: t.Optional[SlicesInterp] = None
    """
    Slices to interpolate with. Required when 'init_layer_preprocess: interp'.
    """

    init_layer_interp: t.List[float] = Field(default_factory=list)
    """List of slice z positions. Initialized from 'init_slices' in python."""

    init_layer_scaling_factor: float = 1.
    """Scaling factor applied to phase of previous object layers. Useful when 'delta_z' is changed."""

    @validator('init_layer_interp')
    def validate_init_layer_interp(cls, v, values: t.Dict[str, t.Any], **kwargs) -> t.Any:
        if v is None or len(v) == 0:
            if values.get('init_slices') is not None:
                return values['init_slices'].layers()
            elif values.get('init_layer_preprocess') == 'interp':
                raise ValueError("'init_slices' or 'init_layer_interp' required when 'init_layer_preprocess: interp'")
        return v

    tilt_x: float = 0.
    """
    Propagator x tilt (in mrad).

    Positive causes a shift rightwards (+x) when moving into the sample.
    Unrelated to `sample_rotation_angles` and `apply_tilted_plane_correction`.
    """
    tilt_y: float = 0.
    """
    Propagator y tilt (in mrad).

    Positive causes a shift downwards (+y) when moving into the sample.
    Unrelated to `sample_rotation_angles` and `apply_tilted_plane_correction`.
    """

    init_layer_append_mode: t.Literal['vac', 'edge', 'avg'] = 'vac'
    """
    How to initialize extra object layers.
    'vac' (default): Add vacuum layers
    'edge': Copy edge layers
    'avg': Copy average of layers
    """

    preshift_ML_probe: bool = False
    """
    If true, the provided probe is at the center of the object. If false, it is at the top of object.

    Doesn't work in combination with ``sample_rotation_angles``.
    """

    asize_presolve: t.Optional[t.Tuple[PositiveInt, PositiveInt]] = None
    """Crop data to get low resolution estimate for next engine"""
    align_shared_objects: bool = False
    """Whether to align shared objects"""

    method: t.Literal['MLs', 'MLc', 'DM', 'ePIE', 'hPIE'] = 'MLs'
    """Optimization method. MLs = maximum-likelihood sparse, MLc = maximum-likelihood compact, DM = difference map"""

    opt_errmetric: t.Literal['L1', 'poisson'] = 'L1'
    """Optimization likelihood metric"""
    grouping: PositiveInt = 64
    """Size of processed blocks. Memory/efficiency trade-off, but smaller may lead to faster convergence for MLs"""
    probe_modes: PositiveInt = 8
    """Number of coherent probe modes"""
    object_modes: PositiveInt = 1
    """Number of coherent object modes"""
    object_change_start: PositiveInt = 1
    """Start updating object potential at this iteration."""
    probe_change_start: PositiveInt = 20
    """Start updating probe at this iteration."""
    probe_position_search: PositiveInt = 50
    """Iteration number to start probe position update at"""

    reg_mu: float = 0.
    """Object smoothness regularization constant. 0 for no regularization"""
    delta: float = 0.
    """Press values to zero out of the illumination area in the object, usually 1e-2 is enough"""
    positivity_constraint_object: float = 0.
    """Enforce weak positivity in object. 1e-2 should be enough."""
    amplitude_threshold_object: float = 1.5
    """Clamp object amplitude. Set to 'inf' to disable."""

    apply_multimodal_update: bool = False
    """If true, update all probe modes. If false, only update first probe mode."""
    probe_backpropagate: float = 0.
    """Backpropagation distance for the probe mask, 0 == apply in the object plane"""
    probe_support_radius: t.Union[None, float] = None
    """Normalized radius of circular support."""
    probe_support_fft: bool = False
    """Assume that there is not illumination intensity out of the central FZP zone."""
    probe_support_tem: bool = False
    """Limit the reconstructed probe support based on initial_probe. Added by ZC."""

    # Orthogonal Probe Relaxation
    variable_probe: bool = False
    """Enable OPR (vary probe modes throughout a single scan)."""
    variable_probe_modes: PositiveInt = 1
    """Number of OPR probe modes."""
    variable_probe_smooth: NonNegativeInt = 0
    """Order of polynomial fit used to smooth spatial evolution of OPR modes. 0 = no smoothing"""
    variable_intensity: bool = False
    """Whether to account for changes in probe intensity."""

    # PIE / ML reconstruction parameters
    beta_object: float = Field(1., gt=0., le=1.)
    """Object step size, <= 1"""
    beta_probe: float = Field(1., gt=0., le=1.)
    """Probe step size, <= 1"""
    delta_p: float = 0.1
    """LSQ damping constant"""
    beta_LSQ: float = Field(0.9, gt=0., le=1.)
    """Least-squares step size. Should be ~0.5 for noisy data, ~0.9 for clean data."""

    # MLc reconstruction
    momentum: float = 0.
    """Add momentum to the MLc method"""
    accelerated_gradients_start: int = 1000000
    """Iteration number to start Nesterov gradient acceleration at"""

    # DM reconstruction parameters
    pfft_relaxation: float = 0.05
    """Relaxation in the Fourier domain"""
    probe_regularization: float = 0.1
    """Weight factor for the probe update (inertia)"""

    apply_relaxed_position_constraint: bool = True
    """
    When true, slowly relax probe positions towards the affine/geometry model.
    When false, update random probe error and geometry model independently.
    """
    max_pos_update_shift: float = 0.1
    """Maximum position update allowed each iteration (px)."""
    probe_position_error_max: float = 2.
    """Maximum random position error (px). Set to 0 to disable random position correction."""
    probe_position_search_momentum: float = 0.
    """Momentum acceleration for probe position update."""

    save_images: t.List[t.Literal[
        'obj_ph', 'obj_ph_sum', 'obj_ph_stack',
        'obj_mag', 'obj_mag_sum', 'obj_mag_stack',
        'probe_mag', 'probe',
    ]] = Field(default_factory=lambda: [
        'obj_ph_sum', 'obj_ph_stack', 'obj_mag_sum', 'obj_mag_stack', 'probe', 'probe_mag'
    ])
    """Intermediate results to save as TIFF images."""


class PlotParams(ModelConfig):
    prepared_data: bool = False
    interval: t.Optional[int] = None
    log_scale: t.Tuple[bool, bool] = (False, False)
    realaxes: bool = True
    remove_phase_ramp: bool = False
    fov_box: bool = False
    fov_box_color: str = 'r'
    positions: bool = True
    mask_bool: bool = True
    windowautopos: bool = True
    obj_apod: bool = False
    prop_obj: float = 0.
    """meters"""
    show_layers: bool = True
    show_layers_stack: bool = False
    object_spectrum: t.Optional[bool] = False
    probe_spectrum: t.Optional[bool] = False
    conjugate: bool = False
    horz_fact: float = 2.5
    FP_maskdim: float = 180e-6
    calc_FSC: bool = False
    show_FSC: bool = False
    residua: bool = False


class SaveParams(ModelConfig):
    external: bool = True
    """Save in external matlab session"""
    store_images: bool = False
    """Save images of final reconstructions (to '$base_path/analysis/online/ptycho')."""
    store_images_intermediate: bool = False
    """Save images of reconstructions after each engine (to '$base_path/analysis/online/ptycho')."""
    store_images_ids: t.List[PositiveInt] = Field(default_factory=lambda: [1, 2, 3, 4])
    """IDs of images to be stored. 1 = obj amplitude, 2 = obj phase, 3 = probes, 4 = errors, 5 = probes spectrum, 6 = object spectrum"""
    store_images_format: t.Literal['png', 'jpg'] = 'png'
    store_images_dpi: PositiveInt = 300
    exclude: t.List[str] = Field(default_factory=lambda: ['fmag', 'fmask', 'illum_sum'])
    """Variables to exclude from saved output"""
    save_reconstructions_intermediate: bool = False
    save_reconstructions: bool = False
    output_file: t.Literal['h5', 'mat'] = 'h5'


class Params(ModelConfig):
    name: str
    """Name of reconstruction. Used by python only."""

    file_type: t.Literal['recons_params'] = 'recons_params'

    engine: t.Literal['fold_slice'] = 'fold_slice'
    """Reconstruction engine to use. For future use"""

    # Display/output params
    verbose_level: int = 2
    """0-1 for loops, 2-3 for testing, >=4 for debugging."""
    use_display: bool = False
    """Whether to display plots"""
    scan_number: ValueOrList[int] = ValueOrList.parse_obj(1)
    """Scan number for shared scans?"""

    # Geometry
    z: t.Literal[1] = 1
    """Distance from object to detector."""
    asize: t.Tuple[PositiveInt, PositiveInt] = (128, 128)
    """Diffraction pattern size in px (y, x)."""
    ctr: t.Optional[t.Tuple[PositiveInt, PositiveInt]] = None
    """Diffraction pattern center coordinates (y, x)."""

    beam_source: t.Literal['electron'] = 'electron'
    """Beam source."""
    d_alpha: float
    """Diffraction pixel size (mrad)"""
    prop_regime: t.Literal['farfield', 'nearfield'] = 'farfield'
    """Wave propagator to use. Nearfield = Fresnel, Farfield (default) = Fraunhofer"""
    focus_to_sample_distance: t.Optional[float] = None
    """Focus to sample distance, used for near-field propagator."""
    energy: float = 200.
    """Beam energy (in keV)."""

    apply_tilted_plane_correction: t.Literal['', 'propagation', 'diffraction'] = ''
    """
    If enabled, applies 'sample_rotation_angles' to the propagator ('propagation') or raw data ('diffraction')
    """
    sample_rotation_angles: t.Tuple[float, float, float] = (0., 0., 0.)
    """
    Sample mistilt in [X, Y, Z (rotation)], in degrees.

    Applied if ``apply_tilted_plane_correction = 'propagation'``.
    """


    auto_center_data: bool = True
    """Auto center cbed patterns. Crops diffraction pixels accordingly"""

    #thickness: float
    #"""Object thickness (in angstroms)."""

    #n_layers: PositiveInt
    #"""Multislice layers. 1 for single slice"""

    affine_angle: t.Optional[float] = 0.
    """Angle (in degrees) to rotate probe positions by."""
    affine_matrix: t.List[t.List[float]] = []
    """Affine matrix to apply to probe positions. Generated from `affine_angle` if not specified."""

    @root_validator
    def validate_transform(cls, values):
        if len(values['affine_matrix']) == 0:
            if values.get('affine_angle') is None:
                raise ValueError("Either affine_matrix or affine_angle must be specified.")
            a = values['affine_angle'] * math.pi / 180.
            values['affine_matrix'] = [[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]
        elif len(values['affine_matrix']) != 2 or len(values['affine_matrix'][0]) != 2 or len(values['affine_matrix'][1]) != 2:
            raise ValueError("Invalid shape for 'affine_matrix'. Expected a 2x2 matrix.")
        return values

    @root_validator(pre=False)
    def valdiate_paths(cls, params):
        # expand paths
        for k in ('base_path', 'ptycho_matlab_path', 'cSAXS_matlab_path'):
            params[k] = params[k].expanduser()

        # if base_path is relative, expand it
        params['base_path'] = params['base_path'].absolute()

        # make certain paths relative to base_path if specified
        for k in ('prepare_data_path', 'save_path', 'raw_data_path', 'initial_probe_file', 'initial_iterate_object_file'):
            if params[k] is None:
                params[k] = Path()
            else:
                params[k] = params[k].expanduser()
                if not params[k].is_absolute():
                    params[k] = params['base_path'] / params[k]

        if 'scan' in params and isinstance(params['scan'], CustomScanParams):
            path = params['scan'].custom_positions_source.expanduser()
            if not path.is_absolute():
                path = params['base_path'] / path
            params['scan'] = CustomScanParams(custom_positions_source=path)

        return params

    @root_validator
    def update_engine_defaults(cls, values):
        # update engines using default values from all_engines
        engines = t.cast(t.List[EngineParams], values['engines'])
        all_engines = t.cast(EngineParams, values['all_engines'])
        default_fields = all_engines.dict(exclude_unset=True)
        default_fields['delta_z'] = all_engines.delta_z
        default_fields['init_layer_interp'] = all_engines.init_layer_interp

        def update_engine(engine: EngineParams) -> EngineParams:
            d = default_fields.copy()
            d.update(engine.dict(exclude_unset=True))
            engine = EngineParams(**d)
            if engine.delta_z is None:  # type: ignore
                raise ValueError("'slices' or 'delta_z' must be specified for all engines.")
            return engine

        values['engines'] = list(map(update_engine, engines))

        for (i, engine) in enumerate(values['engines']):
            try:
                fout = engine.fout.format(i=i+1, **engine.dict())
                fout = Path(fout).expanduser()
                if not fout.is_absolute():
                    fout = values['base_path'] / fout
                fout = str(fout) + os.sep
                object.__setattr__(engine, 'fout', fout)
            except KeyError as e:
                raise FormatError(f"Invalid format string in 'fout' (unknown key {e})") from None
            except Exception as e:
                raise FormatError("Invalid format string in 'fout'") from e
        #for engine in engines:
            #if engine.delta_z is None and 'thickness' in values and 'n_layers' in values:
            #    # hack to get around Engine immutability
            #    object.__setattr__(engine, 'delta_z', [values['thickness'] / values['n_layers']] * values['n_layers'])
            #if engine.probe_modes is None:
            #    object.__setattr__(engine, 'probe_modes', values['probe_modes'])
        #values['engines'] = engines
        return values

    src_metadata: t.Literal['none'] = 'none'
    """Not currently used"""

    queue: QueueParams = QueueParams.parse_obj({})

    detector: DetectorParams = DetectorParams.parse_obj({})

    prepare: PrepareParams = PrepareParams.parse_obj({})

    all_engines: EngineParams = Field(default_factory=lambda: EngineParams.parse_obj({}))

    engines: t.List[EngineParams] = Field(default_factory=lambda: [EngineParams.parse_obj({})])

    src_positions: t.Literal['matlab_pos', 'load_from_file'] = 'matlab_pos'
    """Where to get scan positions from"""
    positions_file: str = ""
    """When src_positions=load_from_file, position file to load from. Formatted with the scan number"""

    scan: t.Union[RasterScanParams, CustomScanParams, ListScanParams] = Field(..., discriminator='type')
    """Scan settings"""

    prefix: str = ""
    """Prefix for outputs. If empty, scan number is used"""
    suffix: str = 'ML_recon'
    """Suffix for reconstruction outputs"""
    scan_string_format: str = '%01d'
    """Matlab format string used to format scan number"""

    base_path: Path = Path("")
    """Directory to reconstruct into."""
    ptycho_matlab_path: Path = Path("")
    """Path to fold_slice/ptycho folder."""
    cSAXS_matlab_path: Path = Path("")
    """Path to csolver. Shouldn't be necessary."""
    raw_data_path: Path = Field(None)
    """Path to get raw data from. Can be specified relative to base_path."""
    raw_data_filename: str = ""
    """Raw data filename. Defaults to 'scan_x%d_y%d.raw'."""

    prepare_data_filename: str = ""
    """Filename to write prepared data (as HDF5) to."""

    save_path: Path = Field(None)
    """
    Filename to save analysis to. Can be specified relative to base_path.
    Defaults, in matlab, to '{base_path}/analysis/S00000-00999/S{scan_number:05}/'.
    """
    specfile: t.Literal[''] = ''
    """Metadata filename. Currently unused"""

    prepare_data_path: Path = Field(None)
    """
    Filename to write prepared data into. Can be specified relative to base_path.
    Defaults to save_path.
    """

    io: IOParams = IOParams.parse_obj({})

    model_object: bool = True
    """If true, model initial object using 'object_type'. If false, load from file ('initial_iterate_object_file')."""
    model_probe: bool = True
    """Whether to model probe or load it from a file"""

    model: ModelParams
    """Probe & Object model settings"""

    initial_iterate_object_file: t.Optional[Path] = None
    """
    File to load initial object from. Used only if model_object=false.
    Can be specified relative to base_path.
    """
    multiple_layers_obj: bool = True
    """Whether 'initial_iterate_object_file' is multi- or single-slice"""

    initial_probe_file: t.Optional[Path] = None
    """
    File to load initial probe from. Used only if model_probe=false.
    Can be specified relative to base_path.
    """
    normalize_init_probe: bool = True
    """Whether to normalize initial probe. Should be disabled when loading an existing reconstruction."""
    crop_pad_init_probe: bool = False
    """Whether to crop/pad (True) or interpolate (False) real-space probe to match the reconstruction size."""
    probe_file_propagation: float = 0.
    """Distance to propagate the probe from the initial position (meters)"""

    share_probe: bool = False
    """Whether to share probes between scans."""
    share_object: bool = False
    """Whether to share object between scans."""

    mode_start_pow: t.Union[float, t.List[float]] = 0.02
    """Normalized intensity to start higher probe modes at."""
    mode_start: t.Literal['rand', 'herm', 'hermver', 'hermhor'] = 'herm'
    """Higher mode probe initialization."""
    ortho_probes: bool = True
    """Orthogonalize probe modes after each engine"""
    ortho_probe_modes: bool = False
    """Orthogonalize probe modes after each iteration"""

    plot: PlotParams = PlotParams.parse_obj({})
    save: SaveParams = SaveParams.parse_obj({})


class EngineParamSet(EngineParams):
    number_iterations: ValueOrList[PositiveInt] = ValueOrList.parse_obj(200)

    fout: str = "{method}_{opt_errmetric}_p{probe_modes}_g{grouping}_step{i}"
    """Output path. Format string, can be specified relative to base_path."""

    method: ValueOrList[t.Literal['MLs', 'MLc', 'DM', 'ePIE', 'hPIE']] = ValueOrList.parse_obj('MLs')
    """Optimization method. MLs = maximum-likelihood sparse, MLc = maximum-likelihood compact, DM = difference map"""

    grouping: ValueOrList[PositiveInt] = ValueOrList.parse_obj(64)
    """Size of processed blocks. Memory/efficiency trade-off, but smaller may lead to faster convergence for MLs"""

    object_change_start: ValueOrList[PositiveInt] = ValueOrList.parse_obj(1)
    """Start updating object potential at this iteration."""
    probe_change_start: ValueOrList[PositiveInt] = ValueOrList.parse_obj(20)
    """Start updating probe wavefunctions at this iteration."""
    probe_position_search: ValueOrList[PositiveInt] = ValueOrList.parse_obj(50)
    """Start updating probe positions at this iteration."""

    # Orthogonal Probe Relaxation
    variable_probe: ValueOrList[bool] = ValueOrList.parse_obj(False)
    """Enable OPR (vary probe modes throughout a single scan)."""
    variable_probe_modes: ValueOrList[PositiveInt] = ValueOrList.parse_obj(1)
    """Number of OPR probe modes."""
    variable_probe_smooth: ValueOrList[NonNegativeInt] = ValueOrList.parse_obj(0)
    """Order of polynomial fit used to smooth spatial evolution of OPR modes. 0 = no smoothing"""
    variable_intensity: ValueOrList[bool] = ValueOrList.parse_obj(False)
    """Whether to account for changes in probe intensity."""

    beta_object: ValueOrList[float] = ValueOrList.parse_obj(1.0)
    """Object step size, <= 1"""
    beta_probe: ValueOrList[float] = ValueOrList.parse_obj(1.0)
    """Probe step size, <= 1"""
    delta_p: ValueOrList[float] = ValueOrList.parse_obj(0.1)
    """LSQ damping constant"""
    beta_LSQ: ValueOrList[float] = ValueOrList.parse_obj(0.9)
    """Least-squares step size. Should be ~0.5 for noisy data, ~0.9 for clean data."""

    reg_mu: ValueOrList[float] = ValueOrList.parse_obj(0.0)
    """Object smoothness regularization constant. 0 for no regularization"""
    delta: ValueOrList[float] = ValueOrList.parse_obj(0.0)
    """Press values to zero out of the illumination area in the object, usually 1e-2 is enough"""
    positivity_constraint_object: ValueOrList[float] = ValueOrList.parse_obj(0.0)
    """Enforce weak positivity in object. 1e-2 should be enough."""
    amplitude_threshold_object: ValueOrList[float] = ValueOrList.parse_obj(1.5)
    """Clamp object amplitude. Set to 'inf' to disable."""

    probe_modes: ValueOrList[PositiveInt] = ValueOrList.parse_obj(8)
    """Number of coherent probe modes."""
    object_modes: ValueOrList[PositiveInt] = ValueOrList.parse_obj(8)
    """Number of coherent object modes."""
    regularize_layers: ValueOrList[float] = ValueOrList.parse_obj(1.)
    """Apply regularization on the reconstructed object layers. 0 = no regularization, 0.01 = weak regularization."""

    asize_presolve: ValueOrList[t.Optional[t.Tuple[PositiveInt, PositiveInt]]] = ValueOrList[t.Optional[t.Tuple[PositiveInt, PositiveInt]]].parse_obj(None)
    """Crop data to get low resolution estimate for next engine"""

    slices: t.Optional[SlicesSet] = None

    delta_z: t.Optional[t.List[float]] = None

    @validator('delta_z')
    def validate_delta_z(cls, v, values: t.Dict[str, t.Any], **kwargs) -> t.Any:
        return v

    init_layer_preprocess: ValueOrList[t.Literal['all', 'avg', 'avg1', 'interp']] = ValueOrList.parse_obj('all')
    """
    How to preprocess layers from previous reconstruction step.
    'all' (default): No pre-processing
    'avg': Average layers together
    'avg1': Average layers and keep one (the rest initialized with 'init_layer_append_mode')
    'interp': Interpolate old object layers to 'init_slices'/'init_layer_interp'.
    """

    init_slices: t.Optional[SlicesInterp] = None
    """
    Slices to interpolate with. Required when 'init_layer_preprocess: interp'.
    """

    init_layer_interp: t.List[float] = Field(default_factory=list)
    """List of slice z positions. Initialized from 'init_slices' in python."""

    @validator('init_layer_interp')
    def validate_init_layer_interp(cls, v, values: t.Dict[str, t.Any], **kwargs) -> t.Any:
        return v

    init_layer_scaling_factor: ValueOrList[float] = ValueOrList.parse_obj(1.)
    """Scaling factor applied to phase of previous object layers. Useful when 'delta_z' is changed."""

    tilt_x: ValueOrList[float] = ValueOrList.parse_obj(0.)
    """
    Propagator x tilt (in mrad).

    Positive causes a shift rightwards (+x) when moving into the sample.
    Unrelated to `sample_rotation_angles` and `apply_tilted_plane_correction`.
    """
    tilt_y: ValueOrList[float] = ValueOrList.parse_obj(0.)
    """
    Propagator y tilt (in mrad).

    Positive causes a shift downwards (+y) when moving into the sample.
    Unrelated to `sample_rotation_angles` and `apply_tilted_plane_correction`.
    """

    _iterable_keys = (
        'slices', 'number_iterations', 'method', 'grouping', 'probe_modes', 'object_modes', 'object_change_start', 'probe_change_start',
        'probe_position_search', 'beta_object', 'beta_probe', 'delta_p', 'beta_LSQ', 'regularize_layers', 'asize_presolve',
        'reg_mu', 'delta', 'positivity_constraint_object', 'amplitude_threshold_object',
        'init_layer_preprocess', 'init_layer_scaling_factor', 'tilt_x', 'tilt_y',
        'variable_probe', 'variable_probe_modes', 'variable_probe_smooth', 'variable_intensity',
    )
    _len: t.Optional[int] = PrivateAttr(None)

    def __len__(self) -> int:
        if self._len is None:
            self._len = math.prod(len(getattr(self, k) or (None,)) for k in self._iterable_keys)
        return self._len

    def iter(self, sparse: bool = False) -> t.Iterator[EngineParams]:
        params = dict(filter(lambda t: t[0] in self.__fields_set__, self.__dict__.items()))
        for d in _iter_dict(params, self._iterable_keys, sparse):
            yield EngineParams.parse_obj(d)

    def __iter__(self) -> t.Iterator[EngineParams]:
        return self.iter()


class RasterScanParamSet(RasterScanParams):
    step: ValueOrList[float] = Field(None)
    """Scan step size (angstroms)"""
    step_size_x: float = Field(None)
    """Scan X step size (angstroms)"""
    step_size_y: float = Field(None)
    """Scan Y step size (angstroms)"""

    nx: PositiveInt
    """Number of scan positions in x"""
    ny: PositiveInt
    """Number of scan positions in y"""

    custom_flip: ValueOrList[t.Tuple[bool, bool, bool]] = ValueOrList[t.Tuple[bool, bool, bool]].parse_obj((False, False, False))
    """Custom data flip (left-right, up-down, tranpose)."""

    step_randn_offset: ValueOrList[float] = ValueOrList.parse_obj(0.)
    """Random offset to apply to raster positions (relative to step size?)"""

    _iterable_keys = ('step', 'custom_flip', 'step_randn_offset')
    _len: t.Optional[int] = PrivateAttr(None)

    def __len__(self) -> int:
        if self._len is None:
            self._len = math.prod(len(getattr(self, k) or (None,)) for k in self._iterable_keys)
        return self._len

    def iter(self, sparse: bool = False) -> t.Iterator[RasterScanParams]:
        params = dict(filter(lambda t: t[0] in self.__fields_set__, self.__dict__.items()))
        for d in _iter_dict(params, self._iterable_keys, sparse):
            yield RasterScanParams.parse_obj(d)

    def __iter__(self) -> t.Iterator[RasterScanParams]:
        return self.iter()


class CustomScanParamSet(CustomScanParams):
    custom_positions_source: ValueOrList[Path]
    """Niter.mat file to load positions from. Can be specified relative to base_path."""

    _iterable_keys = ('custom_positions_source',)
    _len: t.Optional[int] = PrivateAttr(None)

    def __len__(self) -> int:
        if self._len is None:
            self._len = math.prod(len(getattr(self, k) or (None,)) for k in self._iterable_keys)
        return self._len

    def iter(self, sparse: bool = False) -> t.Iterator[CustomScanParams]:
        params = dict(filter(lambda t: t[0] in self.__fields_set__, self.__dict__.items()))
        for d in _iter_dict(params, self._iterable_keys, sparse):
            yield CustomScanParams.parse_obj(d)

    def __iter__(self) -> t.Iterator[CustomScanParams]:
        return self.iter()

    def apply_metadata(self, metadata: t.Optional[Metadata] = None) -> CustomScanParamSet:
        return self

    def is_default(self) -> bool:
        return False


class DetectorParamSet(DetectorParams):
    upsampling: ValueOrList[t.Union[NonNegativeInt, bool]] = ValueOrList.parse_obj(False)

    bg_sub: ValueOrList[t.Union[bool, float]] = ValueOrList.parse_obj(False)
    """Subtract background intensity from diffraction patterns."""

    crop: ValueOrList[t.Optional[t.Tuple[int, int, int, int]]] = ValueOrList[t.Optional[t.Tuple[int, int, int, int]]].parse_obj(None)
    """
    Scan dimensions to crop to (min_x, max_x, min_y, max_y).
    Matlab-style slicing, so [1, 128, 1, 128] == an entire 128x128 scan
    """

    step: ValueOrList[t.Optional[t.Tuple[int, int]]] = ValueOrList[t.Optional[t.Tuple[int, int]]].parse_obj(None)
    """
    Only use every n x m scan positions (x, y). Useful for testing reconstruction at multiples of step size.
    """

    tile: ValueOrList[t.Optional[t.Tuple[int, int]]] = ValueOrList[t.Optional[t.Tuple[int, int]]].parse_obj(None)
    """
    Tile the scan n x m times prior to reconstruction (x, y). Useful for increasing the size of simulated data.

    Care must be taken to ensure the scan is periodic
    """

    circ_mask: ValueOrList[t.Optional[int]] = ValueOrList[t.Optional[int]].parse_obj(None)
    """If specified, applies a circular mask of the radius to each diffraction pattern."""

    psf_sigma: ValueOrList[t.Optional[float]] = ValueOrList[t.Optional[float]].parse_obj(None)
    """
    Apply Gaussian point spread function to the raw data before reconstruction.
    Mainly useful for simulating a non-ideal detector.
    """

    beam_dose: ValueOrList[t.Optional[float]] = ValueOrList[t.Optional[float]].parse_obj(None)
    """Total electron dose (in e/A^2) to scale simulated data by. Overrides `beam_current` if specified."""

    beam_current: ValueOrList[float] = ValueOrList[float].parse_obj(30.)
    """Beam current (in pA) to scale simulated data by."""

    poisson: ValueOrList[bool] = ValueOrList[bool].parse_obj(False)
    """
    Apply Poisson noise to the raw data before reconstruction.
    If using this with simulated data, make sure `beam_current`
    is set correctly.
    """

    _iterable_keys = ('upsampling', 'bg_sub', 'crop', 'step', 'tile', 'circ_mask', 'psf_sigma', 'poisson', 'beam_dose', 'beam_current')
    _len: t.Optional[int] = PrivateAttr(None)

    def __len__(self) -> int:
        if self._len is None:
            self._len = math.prod(len(getattr(self, k, (None,))) for k in self._iterable_keys)
        return self._len

    def iter(self, sparse: bool = False) -> t.Iterator[DetectorParams]:
        params = dict(filter(lambda t: t[0] in self.__fields_set__, self.__dict__.items()))
        for d in _iter_dict(params, self._iterable_keys, sparse):
            yield DetectorParams.parse_obj(d)

    def __iter__(self) -> t.Iterator[DetectorParams]:
        return self.iter()

    def apply_metadata(self, metadata: t.Optional[Metadata] = None) -> DetectorParamSet:
        from_meta: t.Dict[str, t.Any] = {}
        if metadata is not None:
            from_meta['sim'] = metadata.is_simulated()
            if metadata.crop is not None:
                from_meta['crop'] = ValueOrList[t.Optional[t.Tuple[int, int, int, int]]].parse_obj(metadata.crop)

        return DetectorParamSet.parse_obj({**from_meta, **{k: self.__dict__[k] for k in self.__fields_set__}})


class ParamSet(Params):
    name: str
    """Name of reconstruction. Used by python only. May be a format string"""

    file_type: t.Literal['recons_param_set'] = 'recons_param_set'

    base_path: Path = Path("{name}")
    """Directory to reconstruct into. May be a Python format string, which is expanded for each reconstruction."""

    sample_rotation_angles: ValueOrList[t.Tuple[float, float, float]] = ValueOrList[t.Tuple[float, float, float]].parse_obj((0., 0., 0.))
    """
    Sample mistilt in [X, Y, Z (rotation)], in degrees.

    Applied if ``apply_tilted_plane_correction = 'propagation'``.
    """

    energy: ValueOrList[float] = ValueOrList.parse_obj(200.)
    """Beam energy (in keV)."""

    asize: ValueOrList[t.Tuple[PositiveInt, PositiveInt]] = ValueOrList[t.Tuple[PositiveInt, PositiveInt]].parse_obj((128, 128))
    """Diffraction pattern size in px (y, x)."""

    detector: DetectorParamSet = DetectorParamSet.parse_obj({})
    """Detector settings"""

    scan: t.Union[RasterScanParamSet, CustomScanParamSet, ListScanParams]
    """Scan settings"""

    model: ModelParamSet
    """Probe & Object model settings"""

    prepare_data_path: t.Optional[Path] = None
    save_path: t.Optional[Path] = None
    raw_data_path: t.Optional[Path] = None

    all_engines: EngineParamSet = Field(default_factory = lambda: EngineParamSet.parse_obj({}))

    engines: t.List[EngineParamSet] = Field(default_factory = lambda: [EngineParamSet.parse_obj({})])

    d_alpha: ValueOrList[float] = Field(None)
    """Diffraction pixel size (mrad)."""

    affine_angle: ValueOrList[t.Optional[float]] = ValueOrList[t.Optional[float]].parse_obj(None)
    """Angle (in degrees) to rotate probe positions by."""

    affine_matrix: ValueOrList[t.List[t.List[float]]] = ValueOrList.parse_obj([[]])
    """Affine matrix to apply to probe positions. Generated from `affine_angle` if not specified."""

    sparse: bool = False
    """Whether to return a sparse or dense combination of parameters."""

    @root_validator
    def valdiate_paths(cls, params):
        return params

    @root_validator
    def update_engine_defaults(cls, values):
        return values

    @root_validator
    def validate_transform(cls, values):
        return values

    _iterable_keys = (
        'energy', 'scan', 'detector', 'd_alpha', 'affine_angle', 'affine_matrix',
        'all_engines', 'model', 'asize', 'sample_rotation_angles',
    )
    _len: t.Optional[int] = PrivateAttr(None)

    def __len__(self) -> int:
        if self._len is None:
            self._len = (math.prod(len(getattr(self, k) or (None,)) for k in self._iterable_keys) *
                         math.prod(len(engine) for engine in self.engines))
        return self._len

    def iter(self,
        _save_record: t.Optional[SaveRecord] = None, metadata: t.Optional[Metadata] = None,
        sparse: t.Optional[bool] = None
    ) -> t.Iterator[Params]:
        # keep a list of used paths, names, and a counter of iterations.
        if _save_record is None:
            _save_record = SaveRecord()
        sparse = sparse if sparse is not None else self.sparse

        params = dict(filter(lambda t: t[0] in self.__fields_set__, self.__dict__.items()))
        params.pop('file_type', None)
        params.pop('sparse', None)

        # If keys are missing from `vals`, replace with a sentinel which gets filtered out later
        for d in _iter_dict(params, self._iterable_keys, sparse):
            # apply values for this run (skipping those which are _missing)
            for engines in product(*self.engines):
                d.update(engines=list(engines))

                # format run-specific name and base_path
                try:
                    name = self.name.format(i=_save_record.i, meta=metadata, sparse=sparse, **d)
                    d['name'] = _save_record.deduplicate_name(name)
                    path = str(self.base_path).format(i=_save_record.i, meta=metadata, **d)
                    d['base_path'] = _save_record.deduplicate_path(path)
                except KeyError as e:
                    raise FormatError(f"Invalid format string (unknown key {e})") from None
                except Exception as e:
                    raise FormatError("Invalid format string") from e

                # finally, yield a parsed object
                yield Params.parse_obj(d)
                _save_record.i += 1

    def __iter__(self) -> t.Iterator[Params]:
        return self.iter()


class RasterScanParamMetaSet(RasterScanParamSet):
    nx: t.Optional[PositiveInt] = None
    """Number of scan positions in x. Required if metadata not given."""
    ny: t.Optional[PositiveInt] = None
    """Number of scan positions in y. Required if metadata not given."""

    @root_validator
    def validate_scan_step(cls, values):
        # save validation for after we have any metadata
        return values

    def apply_metadata(self, metadata: t.Optional[Metadata] = None) -> RasterScanParamSet:
        from_meta: t.Dict[str, t.Any] = {}
        if metadata is not None:
            from_meta.update(
                nx=metadata.scan_shape[0],
                ny=metadata.scan_shape[1],
                step=ValueOrList.parse_obj(metadata.scan_step[0]*1e10),  # m to A
                step_size_x=metadata.scan_step[0]*1e10,                  # m to A
                step_size_y=metadata.scan_step[1]*1e10,                  # m to A
            )
        else:
            if self.nx is None or self.ny is None:
                raise ValueError("Scan 'nx' and 'ny' must be specified in parameters or metadata.")
            if self.step is None and (self.step_size_x is None or self.step_size_y is None):
                raise ValueError("Scan 'step' or 'step_size_x' and 'step_size_y' must be specified in parameters or metadata.")

        d = {**from_meta, **{k: self.__dict__[k] for k in self.__fields_set__}}
        return RasterScanParamSet.parse_obj({k: v.__root__ if isinstance(v, WrapperModel) else v for (k, v) in d.items()})

    def is_default(self) -> bool:
        return not len(self.__fields_set__)


class ModelParamMetaSet(ModelParamSet):
    probe_alpha_max: t.Optional[ValueOrList[float]] = None
    """Model probe convergence angle (mrad). Required if metadata not given."""
    probe_df: t.Optional[ValueOrList[float]] = None
    """Model probe defocus (angstroms, overfocus is negative). Required if metadata not given."""

    def apply_metadata(self, metadata: t.Optional[Metadata] = None) -> ModelParamSet:
        from_metadata: t.Dict[str, t.Any] = {}
        if metadata is not None:
            if metadata.conv_angle is not None:
                from_metadata['probe_alpha_max'] = ValueOrList[float].parse_obj(metadata.conv_angle)
            if metadata.defocus is not None:
                # sign convention opposite, m -> angstrom
                from_metadata['probe_df'] = ValueOrList[float].parse_obj(-1e10 * metadata.defocus)

        new = {**from_metadata, **{k: self.__dict__[k] for k in self.__fields_set__}}

        if new.get('probe_alpha_max') is None:
            raise ValueError("'probe_alpha_max' must be specified in parameters or metadata.")
        if new.get('probe_df') is None:
            raise ValueError("'probe_df' must be specified in parameters or metadata.")

        return ModelParamSet.parse_obj({k: v.__root__ if isinstance(v, WrapperModel) else v for (k, v) in new.items()})


class ParamMetaSet(ParamSet):
    file_type: t.Literal['recons_param_meta'] = 'recons_param_meta'

    meta: ValueOrList[t.Union[AnyMetadata, Path, None]] = ValueOrList[t.Union[AnyMetadata, Path, None]].parse_obj(None)
    """
    Metadata object or path(s) to metadata files. If a relative path is specified,
    it will be resolved relative to the path of the reconstruction file.
    """

    @validator('meta', pre=False)
    def validate_meta_path(cls, meta: ValueOrList[t.Union[AnyMetadata, Path, None]]) -> ValueOrList[t.Union[AnyMetadata, Path, None]]:
        if meta is None:
            return ValueOrList[t.Union[AnyMetadata, Path, None]].parse_obj(None)
        return meta.map(lambda p: p.expanduser() if isinstance(p, Path) else p, t.Union[AnyMetadata, Path, None])

    energy: t.Optional[ValueOrList[float]] = None
    """Beam energy (in keV). Required if metadata not given."""
    raw_data_path: t.Optional[Path] = None
    """Path to raw data. Can be specified relative to 'base_path'."""
    raw_data_filename: t.Optional[str] = None
    """Raw data filename. Matlab format string. Defaults to 'scan_x%d_y%d.raw' if metadata not given."""
    d_alpha: t.Optional[ValueOrList[float]] = None
    """Diffraction pixel size (mrad). Required if metadata not given."""

    scan: t.Union[RasterScanParamMetaSet, CustomScanParamSet, ListScanParams] = RasterScanParamMetaSet.parse_obj({})
    """Scan settings"""

    model: ModelParamMetaSet = ModelParamMetaSet.parse_obj({})
    """Probe & Object model settings"""

    def with_metadata(self, metadata: t.Union[None, Metadata, t.Sequence[Metadata]]) -> ParamMetaSet:
        if 'meta' in self.__fields_set__:
            print("Overriding metadata specified in ParamMetaSet.", file=sys.stderr)
        return self.copy(update={'meta': ValueOrList[t.Union[AnyMetadata, Path, None]].parse_obj(metadata)})

    def apply_metadata(self, metadata: t.Optional[Metadata] = None) -> ParamSet:
        from_meta = {}
        if metadata is not None:
            from_meta.update(
                energy=ValueOrList.parse_obj(metadata.voltage * 1e-3),  # V to kV
                raw_data_path=metadata.path.resolve(),  # type: ignore
                raw_data_filename=metadata.raw_filename,
            )
            if metadata.diff_step is not None:
                from_meta['d_alpha'] = ValueOrList.parse_obj(metadata.diff_step)

            if hasattr(metadata, 'scan_correction') and metadata.scan_correction is not None:
                a = metadata.scan_rotation * math.pi / 180.
                rot = numpy.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
                # apply rotation after scan correction (is this right?)
                # also, flip to [y, x] coordinates for ptycho shelves
                from_meta['affine_matrix'] = ValueOrList.parse_obj((rot @ numpy.array(metadata.scan_correction)[::-1, ::-1]).tolist())
                from_meta['affine_angle'] = 0.
            else:
                from_meta['affine_angle'] = ValueOrList.parse_obj(metadata.scan_rotation)
        else:
            if self.d_alpha is None:
                raise ValueError("'d_alpha' must be specified in parameters or metadata.")
            if self.energy is None:
                raise ValueError("'energy' must be specified in parameters or metadata.")

        if metadata is not None and hasattr(metadata, 'scan_positions') and metadata.scan_positions is not None \
            and self.scan.is_default():
            # scale m -> A, flip 180 degrees, transpose
            positions = [(-y*1e10, -x*1e10) for (x, y) in metadata.scan_positions]

            # center around 0, 0
            xshift = (min([p[0] for p in positions]) + max([p[0] for p in positions]))/2.
            yshift = (min([p[1] for p in positions]) + max([p[1] for p in positions]))/2.
            positions = [(x - xshift, y - yshift) for (x, y) in positions]

            from_meta['scan'] = ListScanParams(scan_positions=positions)
        else:
            from_meta['scan'] = self.scan.apply_metadata(metadata)

        from_meta.update(
            model = self.model.apply_metadata(metadata),
            detector = self.detector.apply_metadata(metadata),
        )

        fields = self.__fields_set__ - {'meta', 'scan', 'model', 'detector'}
        d = {**from_meta, **{k: self.__dict__[k] for k in fields}}
        return ParamSet.parse_obj({k: v.__root__ if isinstance(v, WrapperModel) else v for (k, v) in d.items()})

    def iter(
            self, _save_record: t.Optional[SaveRecord] = None,
            path: t.Optional[Path] = None, sparse: t.Optional[bool] = None
        ) -> t.Iterator[Params]:
        """
        Iterate through the contained Params objects. Metadata files are loaded relative to `path`.
        """
        # keep a list of used paths, names, and a counter of iterations.
        if _save_record is None:
            _save_record = SaveRecord()

        for meta in self.meta:
            if isinstance(meta, (str, Path)):
                if not meta.is_absolute() and path is not None:
                    meta = path / meta
                print(f"Loading '{meta}' as metadata...")
                meta = AnyMetadata.parse_file(meta)

            yield from self.apply_metadata(meta).iter(_save_record, meta, sparse)


if __name__ == '__main__':
    import yaml

    with open('test.yaml', 'r') as f:
        obj = yaml.safe_load(f)
        params = Params.parse_obj(obj)
        print(params.json())
