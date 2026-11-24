"""Load and validate a complete, reproducible experiment configuration.

All physical/numerical parameters, output switches and paths live in JSON.
Nothing in the computation modules reads configuration from global variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import json
import math

_UNSET = object()


@dataclass(frozen=True)
class Config:
    source: Path
    sections: Mapping[str, Mapping[str, Any]]
    _flat: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._flat[name]
        except KeyError:
            raise AttributeError(f"Unknown configuration parameter: {name}") from None

    @property
    def input_images(self) -> Path:
        return resolve_path(self.source, self.sections['paths']['input_images'])

    @property
    def output_root(self) -> Path:
        return resolve_path(self.source, self.sections['paths']['output_root'])

    def snapshot(self) -> dict:
        """Return the original grouped JSON, suitable for an exact run snapshot."""
        return {group: dict(items) for group, items in self.sections.items()}


def resolve_path(config_path: Path, value: str) -> Path:
    """Resolve relative paths against the JSON file location, not the CWD."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser().resolve()
    with path.open('r', encoding='utf-8') as handle:
        groups = json.load(handle)
    if not isinstance(groups, dict):
        raise ValueError('Top-level configuration must be a JSON object.')
    expected = {
        'paths': {'input_images', 'output_root'},
        'run': {'runID', 'UseParallelFlow', 'MaxFlowWorkers', 'MaxFramePairs'},
        'preprocessing': {'USE_SVD', 'KillIDs'},
        'farneback': {
            'PyramidDownsampleFactor', 'PyramidLevels', 'WindowSize',
            'IterationCount', 'PolyFitKernelSize', 'GaussKernelStdev', 'FilterFlags',
            'FarnebackPyramidMinSize', 'ExplicitFarnebackRegularization',
            'ExplicitMaxResidualIncrement', 'FarnebackWarpMode',
            'CoefficientWarpOpenCVBorders',
        },
        'information_solver': {
            'UseInformationAwareSolver', 'InformationRidgeFraction',
            'UseInformationSpatialPrior', 'InformationPriorWindow',
        },
        'consistency': {
            'ConsistencyErrorThreshold', 'HardThresholdK',
            'RelativeConsistencyAlpha', 'MedianOfMeansGroupSize',
        },
        'confidence': {
            'UseConfidenceSmoothing', 'ConfidenceSigma', 'ConfidenceKernelSize',
            'MinConfidence', 'UseLocalConfidenceSigma', 'LocalSigmaKernelSize',
            'MinimumConfidenceSigma',
        },
        'visualization': {
            'PlotRejectedPoints', 'ArrowDensityStep', 'ArrowScale',
            'VideoFPS', 'VideoDPI', 'VelocityInversion',
        },
        'diagnostics': {
            'SaveDiagnosticMaps', 'SaveValidationData', 'SavePyramidVisualizations',
            'PrintPyramidWindowResolution', 'SavePyramidVelocityPlots',
            'ExportPyramidVelocityArrays', 'SaveStreamlinePlots', 'SaveStreamlineSVG',
            'DiagnosticFramePairs', 'PyramidVelocityDPI', 'PyramidQuiverVectorsAcross',
            'PyramidQuiverScale', 'PyramidVelocityColorMax', 'StreamlineDensity',
            'StreamlineMinSpeed', 'StreamlineMaxGridSize', 'ShowDiagnosticPlots',
            'PyramidVisualizationPairs', 'SavePyramidLevelImages',
            'PyramidVisualizationDPI', 'PyramidKnownU', 'PyramidKnownV',
            'SaveInformationMaps',
        },
        'outputs': {'WriteWXF', 'OverwriteExistingRun', 'SaveVelocityVideo', 'SaveQCPlot'},
    }
    if set(groups) != set(expected):
        raise ValueError(f'JSON sections: missing={sorted(set(expected)-set(groups))}; '
                         f'unknown={sorted(set(groups)-set(expected))}')
    values = {}
    for section, keys in expected.items():
        data = groups[section]
        if not isinstance(data, dict):
            raise ValueError(f'{section} must be a JSON object.')
        if set(data) != keys:
            raise ValueError(f'{section}: missing={sorted(keys-set(data))}; '
                             f'unknown={sorted(set(data)-keys)}')
        for name, value in data.items():
            if name in values:
                raise ValueError(f'Duplicate config key: {name}')
            values[name] = value
    validate(values)
    return Config(path, MappingProxyType({k: MappingProxyType(v) for k,v in groups.items()}),
                  MappingProxyType(values))


def validate(p: Mapping[str, Any]) -> None:
    def positive(name):
        x = p[name]
        if isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) or x <= 0:
            raise ValueError(f'{name} must be a positive finite number; got {x!r}')
    for key in ('PyramidLevels','IterationCount','PolyFitKernelSize',
                'FarnebackPyramidMinSize','ArrowDensityStep','VideoFPS','VideoDPI',
                'GaussKernelStdev','WindowSize','MedianOfMeansGroupSize',
                'MinimumConfidenceSigma','ConfidenceKernelSize','LocalSigmaKernelSize',
                'InformationPriorWindow'):
        positive(key)
    for key in ('PolyFitKernelSize','ConfidenceKernelSize','LocalSigmaKernelSize','InformationPriorWindow'):
        if not isinstance(p[key], int) or p[key] % 2 != 1:
            raise ValueError(f'{key} must be an odd integer.')
    if not (0 < p['PyramidDownsampleFactor'] < 1):
        raise ValueError('PyramidDownsampleFactor must lie in (0, 1).')
    if p['FarnebackWarpMode'] not in ('coefficients','image'):
        raise ValueError("FarnebackWarpMode must be 'coefficients' or 'image'.")
    if p['VelocityInversion'] not in (-1, 0, 1):
        raise ValueError('VelocityInversion must be -1, 0 or 1 (original behavior).')
    for name in ('MaxFlowWorkers', 'MaxFramePairs'):
        if p[name] is not None and (not isinstance(p[name],int) or p[name]<=0):
            raise ValueError(f'{name} must be null or a positive integer.')
    if not isinstance(p['runID'],int) or p['runID'] < 0:
        raise ValueError('runID must be a nonnegative integer.')
    if p['ExplicitMaxResidualIncrement'] is not None:
        positive('ExplicitMaxResidualIncrement')
    if p['PyramidVelocityColorMax'] is not None:
        positive('PyramidVelocityColorMax')
    if (p['PyramidKnownU'] is None) != (p['PyramidKnownV'] is None):
        raise ValueError('PyramidKnownU and PyramidKnownV must either both be null or both numerical.')
    for key in ('input_images','output_root'):
        if not isinstance(p[key],str) or not p[key].strip():
            raise ValueError(f'{key} must be a non-empty path string.')
    for key in ('DiagnosticFramePairs', 'PyramidVisualizationPairs', 'KillIDs'):
        if not isinstance(p[key],list) or not all(isinstance(v,int) and v>=0 for v in p[key]):
            raise ValueError(f'{key} must be an array of non-negative integers.')
    # A repeated SVD mode index would subtract that mode more than once.
    duplicates = sorted({v for v in p['KillIDs'] if p['KillIDs'].count(v) > 1})
    if duplicates:
        raise ValueError(f'KillIDs contains repeated mode indices: {duplicates}')
    if not isinstance(p['WriteWXF'],bool):
        raise ValueError('WriteWXF must be a JSON boolean.')
