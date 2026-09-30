from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from app.extraction.document_loader import ExtractedDocument, Table
from app.extraction.field_extractor import split_value_unit

NOT_SPECIFIED = "Not specified"
_NUMBER_PATTERN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
_UNIT_PATTERN = r"[A-Za-zµ%][A-Za-z0-9µ%*/^.-]*"
_CASE_CODES = {"ULS", "SLS", "SER", "STR", "EQ", "LSD", "LC", "LOAD"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _format_number(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return format(float(value), ".12g")


def _stage_from_text(text: str) -> str:
    value = _clean(text).strip(" \t:;,()[]")
    if not value:
        return ""
    match = re.search(r"\bstage\s*[-:]?\s*(\d+)\b", value, re.I)
    if match:
        return f"Stage {int(match.group(1))}"
    if re.fullmatch(r"\d+", value):
        return f"Stage {int(value)}"
    match = re.search(r"\b(ULS|SLS|SER|STR|EQ|LSD|LC|LOAD)[-_ ]?(\d+)\b", value, re.I)
    if match:
        return f"{match.group(1).upper()}-{int(match.group(2))}"
    match = re.fullmatch(r"(ULS|SLS|SER|STR|EQ|LSD|LC|LOAD)(?:\s*[-:]\s*(.+))?", value, re.I)
    if match:
        suffix = _clean(match.group(2) or "")
        return f"{match.group(1).upper()}-{suffix}" if suffix else match.group(1).upper()
    if re.search(r"\bstage\b", value, re.I) and re.search(r"\d", value):
        number = re.search(r"\d+", value)
        if number:
            return f"Stage {int(number.group(0))}"
    return value


def _stage_number(stage: str) -> tuple[int, str]:
    match = re.search(r"\d+", stage or "")
    return (int(match.group(0)) if match else 10**6, stage or "")


def _parse_cell(cell: str) -> float | str | None:
    value, _, _ = split_value_unit(cell or "")
    return value


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _unit_from_header(header: str) -> str | None:
    text = _clean(header or "")
    match = re.search(r"\(([^)]*)\)\s*$", text)
    if match:
        return match.group(1).strip() or None
    match = re.search(r"\[([^\]]*)\]\s*$", text)
    if match:
        return match.group(1).strip() or None
    return None


def _cell_has_number(cell: str) -> bool:
    return _is_numeric(_parse_cell(cell or ""))


def _is_stage_text(text: str) -> bool:
    value = _clean(text or "")
    return bool(
        re.search(r"\bstage\s*[-:]?\s*\d+\b", value, re.I)
        or re.fullmatch(r"\d+", value)
        or re.search(r"\b(?:ULS|SLS|SER|STR|EQ|LSD|LC)[-_ ]?\d+\b", value, re.I)
        or re.fullmatch(r"(?:ULS|SLS|SER|STR|EQ|LSD|LC)", value, re.I)
    )


def _is_headerish(text: str) -> bool:
    return _norm(text) in {
        "stage", "stageno", "loadcase", "loadcasetype", "case", "combination",
        "name", "number", "layer", "soil", "soiltype", "stratum", "type",
    }


@dataclass
class ReportValue:
    value: float | str | None
    unit: str | None = None
    source: str = ""
    location: str = ""
    raw: str = ""
    label: str = ""
    alternatives: list["ReportValue"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "location": self.location,
            "raw": self.raw,
            "label": self.label,
            "alternatives": [item.to_dict() for item in self.alternatives],
        }


@dataclass
class LoadCase:
    stage: str
    compression: ReportValue | None = None
    tension: ReportValue | None = None
    shear: ReportValue | None = None
    shear_other: ReportValue | None = None
    applied_moment: ReportValue | None = None
    applied_moment_other: ReportValue | None = None
    internal_moment: ReportValue | None = None
    internal_moment_other: ReportValue | None = None
    compression_capacity: ReportValue | None = None
    tension_capacity: ReportValue | None = None
    structural_capacity: ReportValue | None = None
    geotechnical_capacity: ReportValue | None = None
    r_geo: ReportValue | None = None
    r_str: ReportValue | None = None
    r_cap: ReportValue | None = None
    load_type: ReportValue | None = None
    axial_load: ReportValue | None = None
    candidates: dict[str, list[ReportValue]] = field(default_factory=dict, repr=False)
    priorities: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def fx(self) -> ReportValue | None:
        return self.shear

    @fx.setter
    def fx(self, value: ReportValue | None) -> None:
        self.shear = value

    @property
    def fy(self) -> ReportValue | None:
        return self.shear_other

    @fy.setter
    def fy(self, value: ReportValue | None) -> None:
        self.shear_other = value

    @property
    def fxx(self) -> ReportValue | None:
        return self.shear

    @fxx.setter
    def fxx(self, value: ReportValue | None) -> None:
        self.shear = value

    @property
    def fyy(self) -> ReportValue | None:
        return self.shear_other

    @fyy.setter
    def fyy(self, value: ReportValue | None) -> None:
        self.shear_other = value

    @property
    def mxx(self) -> ReportValue | None:
        return self.applied_moment

    @mxx.setter
    def mxx(self, value: ReportValue | None) -> None:
        self.applied_moment = value

    @property
    def myy(self) -> ReportValue | None:
        return self.applied_moment_other

    @myy.setter
    def myy(self, value: ReportValue | None) -> None:
        self.applied_moment_other = value

    @property
    def moment(self) -> ReportValue | None:
        return self.applied_moment

    @moment.setter
    def moment(self, value: ReportValue | None) -> None:
        self.applied_moment = value

    @property
    def type(self) -> ReportValue | None:
        return self.load_type

    @type.setter
    def type(self, value: ReportValue | None) -> None:
        self.load_type = value

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "compression": self.compression.to_dict() if self.compression else None,
            "tension": self.tension.to_dict() if self.tension else None,
            "shear": self.shear.to_dict() if self.shear else None,
            "shear_other": self.shear_other.to_dict() if self.shear_other else None,
            "applied_moment": self.applied_moment.to_dict() if self.applied_moment else None,
            "applied_moment_other": self.applied_moment_other.to_dict() if self.applied_moment_other else None,
            "internal_moment": self.internal_moment.to_dict() if self.internal_moment else None,
            "internal_moment_other": self.internal_moment_other.to_dict() if self.internal_moment_other else None,
            "compression_capacity": self.compression_capacity.to_dict() if self.compression_capacity else None,
            "tension_capacity": self.tension_capacity.to_dict() if self.tension_capacity else None,
            "structural_capacity": self.structural_capacity.to_dict() if self.structural_capacity else None,
            "geotechnical_capacity": self.geotechnical_capacity.to_dict() if self.geotechnical_capacity else None,
            "r_geo": self.r_geo.to_dict() if self.r_geo else None,
            "r_str": self.r_str.to_dict() if self.r_str else None,
            "r_cap": self.r_cap.to_dict() if self.r_cap else None,
            "load_type": self.load_type.to_dict() if self.load_type else None,
            "type": self.load_type.to_dict() if self.load_type else None,
            "axial_load": self.axial_load.to_dict() if self.axial_load else None,
            "fx": self.shear.to_dict() if self.shear else None,
            "fy": self.shear_other.to_dict() if self.shear_other else None,
            "fxx": self.shear.to_dict() if self.shear else None,
            "fyy": self.shear_other.to_dict() if self.shear_other else None,
            "mxx": self.applied_moment.to_dict() if self.applied_moment else None,
            "myy": self.applied_moment_other.to_dict() if self.applied_moment_other else None,
        }


@dataclass
class RatioCase:
    stage: str
    overall: ReportValue | None = None
    structural: ReportValue | None = None
    geotechnical: ReportValue | None = None
    failure_mode: ReportValue | None = None
    governing: bool = False

    @property
    def r_geo(self) -> ReportValue | None:
        return self.geotechnical

    @r_geo.setter
    def r_geo(self, value: ReportValue | None) -> None:
        self.geotechnical = value

    @property
    def r_str(self) -> ReportValue | None:
        return self.structural

    @r_str.setter
    def r_str(self, value: ReportValue | None) -> None:
        self.structural = value

    @property
    def r_cap(self) -> ReportValue | None:
        return self.overall

    @r_cap.setter
    def r_cap(self, value: ReportValue | None) -> None:
        self.overall = value

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "overall": self.overall.to_dict() if self.overall else None,
            "structural": self.structural.to_dict() if self.structural else None,
            "geotechnical": self.geotechnical.to_dict() if self.geotechnical else None,
            "r_geo": self.geotechnical.to_dict() if self.geotechnical else None,
            "r_str": self.structural.to_dict() if self.structural else None,
            "r_cap": self.overall.to_dict() if self.overall else None,
            "failure_mode": self.failure_mode.to_dict() if self.failure_mode else None,
            "governing": self.governing,
        }


@dataclass
class CapacityCase:
    stage: str
    compression_capacity: ReportValue | None = None
    tension_capacity: ReportValue | None = None
    geotechnical_capacity: ReportValue | None = None
    structural_capacity: ReportValue | None = None
    r_geo: ReportValue | None = None
    r_str: ReportValue | None = None
    r_cap: ReportValue | None = None
    failure_mode: ReportValue | None = None
    governing: bool = False

    @property
    def compression(self) -> ReportValue | None:
        return self.compression_capacity

    @property
    def tension(self) -> ReportValue | None:
        return self.tension_capacity

    @property
    def geotechnical(self) -> ReportValue | None:
        return self.geotechnical_capacity

    @property
    def structural(self) -> ReportValue | None:
        return self.structural_capacity

    @property
    def overall(self) -> ReportValue | None:
        return self.r_cap

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "compression_capacity": self.compression_capacity.to_dict() if self.compression_capacity else None,
            "tension_capacity": self.tension_capacity.to_dict() if self.tension_capacity else None,
            "geotechnical_capacity": self.geotechnical_capacity.to_dict() if self.geotechnical_capacity else None,
            "structural_capacity": self.structural_capacity.to_dict() if self.structural_capacity else None,
            "compression": self.compression_capacity.to_dict() if self.compression_capacity else None,
            "tension": self.tension_capacity.to_dict() if self.tension_capacity else None,
            "geotechnical": self.geotechnical_capacity.to_dict() if self.geotechnical_capacity else None,
            "structural": self.structural_capacity.to_dict() if self.structural_capacity else None,
            "r_geo": self.r_geo.to_dict() if self.r_geo else None,
            "r_str": self.r_str.to_dict() if self.r_str else None,
            "r_cap": self.r_cap.to_dict() if self.r_cap else None,
            "overall": self.r_cap.to_dict() if self.r_cap else None,
            "failure_mode": self.failure_mode.to_dict() if self.failure_mode else None,
            "governing": self.governing,
        }


@dataclass
class SoilLayer:
    name: str = ""
    values: dict[str, ReportValue] = field(default_factory=dict)
    source: str = ""
    location: str = ""
    raw: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            for value in self.values.values():
                if value.source:
                    self.source = value.source
                    break
        if not self.location:
            for value in self.values.values():
                if value.location:
                    self.location = value.location
                    break
        if not self.raw:
            for value in self.values.values():
                if value.raw:
                    self.raw = value.raw
                    break

    def get(self, key: str, default: ReportValue | None = None) -> ReportValue | None:
        if key in self.values:
            return self.values[key]
        wanted = _norm(key)
        for name, value in self.values.items():
            if _norm(name) == wanted:
                return value
        return default

    def __getitem__(self, key: str) -> ReportValue | None:
        return self.get(key)

    @property
    def depth(self) -> ReportValue | None:
        return self.get("depth")

    @property
    def depth_top(self) -> ReportValue | None:
        return self.get("depth_top")

    @property
    def depth_bottom(self) -> ReportValue | None:
        return self.get("depth_bottom")

    @property
    def thickness(self) -> ReportValue | None:
        return self.get("thickness")

    @property
    def phi(self) -> ReportValue | None:
        return self.get("phi") or self.get("friction_angle")

    @property
    def cohesion(self) -> ReportValue | None:
        return self.get("cohesion") or self.get("c")

    @property
    def su(self) -> ReportValue | None:
        return self.get("su") or self.get("undrained_shear_strength")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "values": {key: value.to_dict() for key, value in self.values.items()},
            "source": self.source,
            "location": self.location,
            "raw": self.raw,
        }


@dataclass
class SettlementCase:
    stage: str
    settlement: ReportValue | None = None
    design_settlement: ReportValue | None = None
    ultimate_settlement: ReportValue | None = None
    max_settlement: ReportValue | None = None
    warning: ReportValue | None = None
    failure_mode: ReportValue | None = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "settlement": self.settlement.to_dict() if self.settlement else None,
            "design_settlement": self.design_settlement.to_dict() if self.design_settlement else None,
            "ultimate_settlement": self.ultimate_settlement.to_dict() if self.ultimate_settlement else None,
            "max_settlement": self.max_settlement.to_dict() if self.max_settlement else None,
            "warning": self.warning.to_dict() if self.warning else None,
            "failure_mode": self.failure_mode.to_dict() if self.failure_mode else None,
        }


@dataclass
class LateralCase:
    stage: str
    fx: ReportValue | None = None
    fy: ReportValue | None = None
    mxx: ReportValue | None = None
    myy: ReportValue | None = None
    max_deflection: ReportValue | None = None
    rotation: ReportValue | None = None
    lateral_capacity: ReportValue | None = None
    governing: bool = False

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "fx": self.fx.to_dict() if self.fx else None,
            "fy": self.fy.to_dict() if self.fy else None,
            "mxx": self.mxx.to_dict() if self.mxx else None,
            "myy": self.myy.to_dict() if self.myy else None,
            "max_deflection": self.max_deflection.to_dict() if self.max_deflection else None,
            "rotation": self.rotation.to_dict() if self.rotation else None,
            "lateral_capacity": self.lateral_capacity.to_dict() if self.lateral_capacity else None,
            "governing": self.governing,
        }


@dataclass
class MemberSpecification:
    name: str = ""
    diameter: ReportValue | None = None
    thickness: ReportValue | None = None
    yield_strength: ReportValue | None = None
    length: ReportValue | None = None
    material: ReportValue | None = None
    values: dict[str, ReportValue] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "diameter": self.diameter.to_dict() if self.diameter else None,
            "thickness": self.thickness.to_dict() if self.thickness else None,
            "yield_strength": self.yield_strength.to_dict() if self.yield_strength else None,
            "length": self.length.to_dict() if self.length else None,
            "material": self.material.to_dict() if self.material else None,
            "values": {key: value.to_dict() for key, value in self.values.items()},
        }


@dataclass
class ReportReference:
    text: str
    source: str = ""
    location: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "location": self.location, "raw": self.raw}


@dataclass
class EngineeringReport:
    project: dict[str, ReportValue | None] = field(default_factory=dict)
    organization: ReportValue | None = None
    pile_diameter: ReportValue | None = None
    pile_thickness_initial: ReportValue | None = None
    pile_thickness_table: ReportValue | None = None
    pile_thickness_effective: ReportValue | None = None
    yield_strength: ReportValue | None = None
    helix_count: ReportValue | None = None
    helix_diameters: dict[str, ReportValue] = field(default_factory=dict)
    helix_spacing: list[ReportValue] = field(default_factory=list)
    design_life: ReportValue | None = None
    corrosion_loss: ReportValue | None = None
    steel_loss_percent: ReportValue | None = None
    loads: dict[str, LoadCase] = field(default_factory=dict)
    ratios: dict[str, RatioCase] = field(default_factory=dict)
    settlements: dict[str, ReportValue] = field(default_factory=dict)
    soil_parameters: dict[str, ReportValue] = field(default_factory=dict)
    settlement_warnings: list[str] = field(default_factory=list)
    references: list[ReportReference] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pile_length: ReportValue | None = None
    member_type: ReportValue | None = None
    member_material: ReportValue | None = None
    ultimate_strength: ReportValue | None = None
    member_specifications: list[MemberSpecification] = field(default_factory=list)
    soil_layers: list[SoilLayer] = field(default_factory=list)
    capacities: dict[str, CapacityCase] = field(default_factory=dict)
    settlement_cases: dict[str, SettlementCase] = field(default_factory=dict)
    lateral_cases: dict[str, LateralCase] = field(default_factory=dict)
    corrosion: dict[str, ReportValue] = field(default_factory=dict)
    corrosion_cases: dict[str, ReportValue] = field(default_factory=dict)
    governing_stage: str | None = None
    governing_failure_mode: str | None = None
    governing_stage_value: ReportValue | None = None
    evaluation: list[str] = field(default_factory=list)
    missing_values: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for key in ("project_name", "project_no", "structure", "location", "client", "submission_date", "engineer"):
            self.project.setdefault(key, None)

    @property
    def project_info(self) -> dict[str, ReportValue | None]:
        return self.project

    @property
    def capacity_cases(self) -> dict[str, CapacityCase]:
        return self.capacities

    @property
    def lateral_analysis(self) -> dict[str, LateralCase]:
        return self.lateral_cases

    @property
    def soil(self) -> list[SoilLayer]:
        return self.soil_layers

    @property
    def pile_specifications(self) -> list[MemberSpecification]:
        return self.member_specifications

    def add_reference(self, text: str, source: str, location: str, raw: str | None = None) -> None:
        text = _clean(text)
        if not text:
            return
        raw_value = raw if raw is not None else text
        item = ReportReference(text=text, source=source, location=location, raw=raw_value)
        signature = (item.text, item.source, item.location, item.raw)
        if not any((r.text, r.source, r.location, r.raw) == signature for r in self.references):
            self.references.append(item)

    def add_note(self, note: str) -> None:
        if note and note not in self.notes:
            self.notes.append(note)

    def add_soil_layer(self, layer: SoilLayer) -> None:
        if layer is None:
            return
        signature = (layer.name, layer.location, layer.raw)
        if not any((item.name, item.location, item.raw) == signature for item in self.soil_layers):
            self.soil_layers.append(layer)

    def to_dict(self) -> dict:
        return {
            "project": {key: value.to_dict() if value else None for key, value in self.project.items()},
            "organization": self.organization.to_dict() if self.organization else None,
            "pile": {
                "diameter": self.pile_diameter.to_dict() if self.pile_diameter else None,
                "thickness_initial": self.pile_thickness_initial.to_dict() if self.pile_thickness_initial else None,
                "thickness_table": self.pile_thickness_table.to_dict() if self.pile_thickness_table else None,
                "thickness_effective": self.pile_thickness_effective.to_dict() if self.pile_thickness_effective else None,
                "yield_strength": self.yield_strength.to_dict() if self.yield_strength else None,
                "ultimate_strength": self.ultimate_strength.to_dict() if self.ultimate_strength else None,
                "length": self.pile_length.to_dict() if self.pile_length else None,
                "type": self.member_type.to_dict() if self.member_type else None,
                "material": self.member_material.to_dict() if self.member_material else None,
                "helix_count": self.helix_count.to_dict() if self.helix_count else None,
                "helix_diameters": {key: value.to_dict() for key, value in self.helix_diameters.items()},
                "helix_spacing": [value.to_dict() for value in self.helix_spacing],
                "design_life": self.design_life.to_dict() if self.design_life else None,
                "corrosion_loss": self.corrosion_loss.to_dict() if self.corrosion_loss else None,
                "steel_loss_percent": self.steel_loss_percent.to_dict() if self.steel_loss_percent else None,
            },
            "member_specifications": [item.to_dict() for item in self.member_specifications],
            "loads": [case.to_dict() for case in self.loads.values()],
            "ratios": [case.to_dict() for case in self.ratios.values()],
            "capacities": [case.to_dict() for case in self.capacities.values()],
            "settlements": {key: value.to_dict() for key, value in self.settlements.items()},
            "settlement_cases": {key: value.to_dict() for key, value in self.settlement_cases.items()},
            "lateral_cases": {key: value.to_dict() for key, value in self.lateral_cases.items()},
            "soil_parameters": {key: value.to_dict() for key, value in self.soil_parameters.items()},
            "soil_layers": [layer.to_dict() for layer in self.soil_layers],
            "corrosion": {key: value.to_dict() for key, value in self.corrosion.items()},
            "corrosion_cases": {key: value.to_dict() for key, value in self.corrosion_cases.items()},
            "settlement_warnings": list(self.settlement_warnings),
            "governing_stage": self.governing_stage,
            "governing_failure_mode": self.governing_failure_mode,
            "governing_stage_value": self.governing_stage_value.to_dict() if self.governing_stage_value else None,
            "evaluation": list(self.evaluation),
            "missing_values": list(self.missing_values),
            "references": [reference.to_dict() for reference in self.references],
            "notes": list(self.notes),
        }


def _row_values(table: Table) -> list[tuple[int, list[str]]]:
    return [(index + 2, list(row)) for index, row in enumerate(table.rows)]


def _data_start(table: Table) -> int:
    for index, row in enumerate(table.rows):
        if not row:
            continue
        first = _clean(row[0])
        if first and _is_stage_text(first) and not _is_headerish(first):
            return index
        if re.search(r"\d+\s*(?:-|–|to)\s*\d+", first, re.I):
            return index
        if _cell_has_number(first) and any(_cell_has_number(cell) for cell in row[1:]):
            return index
        if index and any(_cell_has_number(cell) for cell in row[1:]):
            return index
    return 0


def _table_width(table: Table) -> int:
    return max([len(table.headers)] + [len(row) for row in table.rows]) if table.rows else len(table.headers)


def _column_text(table: Table, column: int, data_start: int | None = None) -> str:
    start = _data_start(table) if data_start is None else data_start
    values: list[str] = []
    if column < len(table.headers):
        values.append(table.headers[column])
    for row in table.rows[:start]:
        if column < len(row):
            values.append(row[column])
    return _clean(" ".join(value for value in values if value and value.strip()))


def _column_headers(table: Table, data_start: int | None = None) -> list[str]:
    start = _data_start(table) if data_start is None else data_start
    return [_norm(_column_text(table, column, start)) for column in range(_table_width(table))]


def _find_columns(table: Table, predicate: Callable[[str], bool], data_start: int) -> list[int]:
    return [column for column, value in enumerate(_column_headers(table, data_start)) if predicate(value)]


def _find_column(table: Table, predicate: Callable[[str], bool], data_start: int) -> int | None:
    columns = _find_columns(table, predicate, data_start)
    return columns[0] if columns else None


def _cell_value(
    document: ExtractedDocument,
    table: Table,
    row_number: int,
    row: list[str],
    column: int | None,
    label: str,
    unit: str | None = None,
) -> ReportValue | None:
    if column is None or column >= len(row):
        return None
    raw = row[column].strip()
    if not raw:
        return None
    value, cell_unit, _ = split_value_unit(raw)
    if unit is None:
        unit = _unit_from_header(_column_text(table, column, _data_start(table)))
    return ReportValue(value=value, unit=cell_unit or unit, source=document.path.name,
                       location=f"{table.location}, row {row_number}", raw=raw, label=label)


def _text_value(
    document: ExtractedDocument,
    location: str,
    raw: str,
    label: str,
    unit: str | None = None,
    value_text: str | None = None,
) -> ReportValue:
    parsed_raw = raw if value_text is None else value_text
    value, parsed_unit, _ = split_value_unit(parsed_raw)
    return ReportValue(value=value, unit=parsed_unit or unit, source=document.path.name,
                       location=location, raw=raw, label=label)


def _add_alternative(primary: ReportValue | None, candidate: ReportValue) -> ReportValue | None:
    if primary is None:
        return candidate
    if primary.value == candidate.value and primary.unit == candidate.unit:
        return primary
    if not any(item.value == candidate.value and item.unit == candidate.unit for item in primary.alternatives):
        primary.alternatives.append(candidate)
    return primary


def _first(current: ReportValue | None, candidate: ReportValue | None) -> ReportValue | None:
    if candidate is None:
        return current
    return candidate if current is None else _add_alternative(current, candidate)


_FIELD_ALIASES = {"fx": "shear", "fxx": "shear", "fy": "shear_other", "fyy": "shear_other",
                  "mxx": "applied_moment", "myy": "applied_moment_other", "type": "load_type"}


def _set_prioritized(case: LoadCase, name: str, candidate: ReportValue | None, priority: int) -> None:
    if candidate is None:
        return
    target = _FIELD_ALIASES.get(name, name)
    stored = case.candidates.setdefault(target, [])
    if not any(item.source == candidate.source and item.location == candidate.location and item.raw == candidate.raw
               and item.value == candidate.value and item.unit == candidate.unit for item in stored):
        stored.append(candidate)
    current = getattr(case, target)
    current_priority = case.priorities.get(target, 0)
    if current is None or priority >= current_priority:
        if current is None:
            selected = candidate
        elif current is not candidate and (current.value != candidate.value or current.unit != candidate.unit):
            selected = _add_alternative(current, candidate)
        else:
            selected = current
        setattr(case, target, selected)
        case.priorities[target] = max(priority, current_priority)
    case.priorities[name] = max(priority, case.priorities.get(name, 0))


def _case_for(report: EngineeringReport, stage: str) -> LoadCase:
    return report.loads.setdefault(stage, LoadCase(stage=stage))


def _ratio_for(report: EngineeringReport, stage: str) -> RatioCase:
    return report.ratios.setdefault(stage, RatioCase(stage=stage))


def _capacity_for(report: EngineeringReport, stage: str) -> CapacityCase:
    return report.capacities.setdefault(stage, CapacityCase(stage=stage))


def _settlement_for(report: EngineeringReport, stage: str) -> SettlementCase:
    return report.settlement_cases.setdefault(stage, SettlementCase(stage=stage))


def _lateral_for(report: EngineeringReport, stage: str) -> LateralCase:
    return report.lateral_cases.setdefault(stage, LateralCase(stage=stage))


def _signed_value(value: ReportValue) -> ReportValue:
    if isinstance(value.value, (int, float)) and value.value < 0:
        return ReportValue(abs(value.value), value.unit, value.source, value.location, value.raw,
                           value.label, list(value.alternatives))
    return value


def _is_diameter_header(value: str) -> bool:
    value = _norm(value)
    return value in {"dp", "diameter", "od", "outerdiameter", "outsidediameter", "pilediameter", "memberdiameter"} or "pilediameter" in value or "memberdiameter" in value or ("diameter" in value and "soil" not in value)


def _is_thickness_header(value: str) -> bool:
    value = _norm(value)
    return value in {"tp", "t", "thickness", "wallthickness", "pilethickness", "memberthickness"} or ("thickness" in value and "loss" not in value)


def _is_fy_header(value: str) -> bool:
    value = _norm(value)
    return value in {"fy", "yield", "yieldstrength", "steelyieldstrength"} or value.endswith("fy") or "yieldstrength" in value


def _is_fu_header(value: str) -> bool:
    value = _norm(value)
    return value in {"fu", "ultimate", "ultimatestrength", "steelultimatestrength"} or value.endswith("fu") or "ultimatestrength" in value


def _is_count_header(value: str) -> bool:
    value = _norm(value)
    return value in {"numbe", "number", "numberofhelices", "numberofplates", "platecount", "helixcount", "qty", "count"} or "numbe" in value or "numberof" in value


def _is_helix_diameter_header(value: str) -> bool:
    value = _norm(value)
    return value in {"dhel", "helixdiameter", "platediameter", "helixdia"} or "helixdiameter" in value or "platediameter" in value or value.startswith("dhel")


def _is_spacing_header(value: str) -> bool:
    value = _norm(value)
    return value in {"shel", "spacing", "helixspacing", "platespacing"} or "spacing" in value


def _is_stage_header(value: str) -> bool:
    value = _norm(value)
    return value in {"stage", "stageno", "stageid", "loadcase", "loadcaseno", "case", "caseno", "combination", "loadcombination"} or "loadcase" in value or "combination" in value


def _is_compression_header(value: str) -> bool:
    value = _norm(value)
    if "capacity" in value or "ratio" in value:
        return False
    return value in {"comp", "compression", "compressiveload", "axialcompression", "axialcomp", "axc", "c"} or "compression" in value or "compressiveload" in value or "axialcomp" in value


def _is_tension_header(value: str) -> bool:
    value = _norm(value)
    if "capacity" in value or "ratio" in value:
        return False
    return value in {"tension", "tensileload", "axialtension", "axialloadtension", "uplift", "upliftload", "pullout", "t"} or "tension" in value or "uplift" in value or "pullout" in value or "tensileload" in value


def _is_axial_header(value: str) -> bool:
    value = _norm(value)
    return value in {"axialload", "axialforce", "axial", "load", "pileheadload"} or "axialload" in value or "axialforce" in value


def _is_fx_header(value: str) -> bool:
    value = _norm(value)
    return "fxx" in value or "shearfx" in value or "lateralfx" in value or "forcefx" in value or bool(re.search(r"(?:^|[a-z])fx(?:$|[a-z])", value))


def _is_fy_header(value: str) -> bool:
    value = _norm(value)
    return "fyy" in value or "shearfy" in value or "lateralfy" in value or "forcefy" in value or bool(re.search(r"(?:^|[a-z])fy(?:$|[a-z])", value))


def _is_mxx_header(value: str) -> bool:
    value = _norm(value)
    return "mxx" in value or "mxmax" in value or "momentx" in value or "bendingx" in value or "shearmomentx" in value


def _is_myy_header(value: str) -> bool:
    value = _norm(value)
    return "myy" in value or "mymax" in value or "momenty" in value or "bendingy" in value or "shearmomenty" in value


def _is_type_header(value: str) -> bool:
    value = _norm(value)
    return value in {"type", "loadtype", "loadcasetype", "casetype", "action", "actiontype", "loadcategory"} or value.endswith("loadtype") or value.endswith("casetype")


def _ratio_kind(value: str) -> str | None:
    value = _norm(value)
    if not value:
        return None
    if "rgeo" in value or ("geo" in value and "ratio" in value) or ("geotechnical" in value and "ratio" in value):
        return "geo"
    if "rstr" in value or ("struct" in value and "ratio" in value) or ("structural" in value and "ratio" in value):
        return "str"
    if "rcap" in value or ("loadcap" in value and not any(token in value for token in ("geo", "struct"))) or "overallcriticalratio" in value or "overallratio" in value:
        return "cap"
    if value in {"ratio", "dc", "dcratio", "utilization", "demandcapacity"}:
        return "cap"
    return None


def _capacity_kind(value: str) -> str | None:
    value = _norm(value)
    if _ratio_kind(value) is not None:
        return None
    if "compressioncapacity" in value or ("compress" in value and "cap" in value):
        return "compression"
    if "tensioncapacity" in value or "upliftcapacity" in value or ("tension" in value and "cap" in value):
        return "tension"
    if "struct" in value and ("cap" in value or "capacity" in value):
        return "structural"
    if "geo" in value and ("cap" in value or "capacity" in value):
        return "geotechnical"
    if value in {"capacity", "geotechnicalcapacity", "designcapacity", "allowablecapacity", "qall", "qult", "qdesign"}:
        return "generic" if value == "capacity" else "geotechnical"
    if value.startswith("q") and any(token in value for token in ("ult", "all", "des", "cap")):
        return "geotechnical"
    if value.startswith("cap") and value != "capacity":
        return "generic"
    return None


def _is_settlement_header(value: str) -> bool:
    value = _norm(value)
    return "settlement" in value or value in {"settle", "smax", "predictedsettlement", "verticalsettlement"}


def _is_deflection_header(value: str) -> bool:
    value = _norm(value)
    return value in {"deflection", "maxdeflection", "lateraldeflection", "displacement", "lateraldisplacement", "rotation", "maxrotation"}


def _is_soil_header(value: str) -> bool:
    value = _norm(value)
    return value in {"soil", "soiltype", "layer", "stratum", "material", "description"} or "soiltype" in value or "stratum" in value


def _soil_parameter_name(value: str) -> str | None:
    value = _norm(value)
    if value in {"phi", "frictionangle", "angleoffriction", "friction"} or value.endswith("phi"):
        return "phi"
    if value in {"c", "cohesion", "cohesive", "cohesionvalue"} or "cohesion" in value:
        return "cohesion"
    if value in {"su", "undrained", "undrainedshearstrength", "undrainedstrength", "cu", "csu"} or value.startswith("su"):
        return "su"
    if value in {"gamma", "unitweight", "moistunitweight", "dryunitweight", "bulkunitweight", "density"} or "unitweight" in value or value.startswith("gamma"):
        return "unit_weight"
    if value in {"depth", "depthtop", "topdepth", "elev", "elevation", "layerdepth"} or value.startswith("depthtop") or value.startswith("topdepth"):
        return "depth_top"
    if value in {"depthbottom", "bottomdepth", "depthbase"} or value.startswith("depthbottom") or value.startswith("bottomdepth"):
        return "depth_bottom"
    if value in {"thickness", "layerthickness", "soilthickness"}:
        return "thickness"
    if value in {"watercontent", "moisturecontent", "w"}:
        return "water_content"
    if value in {"sppt", "nvalue", "n"}:
        return "spt"
    if value in {"type", "soil", "material", "description", "layername"}:
        return "name"
    return None


def _text_number_matches(line: str, aliases: str) -> list[tuple[re.Match[str], str, str | None]]:
    pattern = re.compile(
        rf"(?P<label>{aliases})(?:\s*\((?P<parenunit>[^)]*)\))?\s*(?:[:=]\s*|\s+)(?P<num>{_NUMBER_PATTERN})(?:\s*(?P<unit>{_UNIT_PATTERN}))?"
    )
    return [(match, match.group("num"), match.group("unit") or match.group("parenunit") or None)
            for match in pattern.finditer(line)]


def _set_project_value(report: EngineeringReport, key: str, candidate: ReportValue) -> None:
    current = report.project.get(key)
    if current is None:
        report.project[key] = candidate
    elif current.value != candidate.value or current.unit != candidate.unit:
        _add_alternative(current, candidate)


_PROJECT_TABLE_LABELS = {
    "project": "project_name", "projectname": "project_name", "nameofproject": "project_name",
    "projectno": "project_no", "projectnumber": "project_no", "jobno": "project_no", "jobnumber": "project_no",
    "structure": "structure", "projectstructure": "structure", "substructure": "structure",
    "descriptionofwork": "structure", "location": "location", "projectlocation": "location",
    "sitelocation": "location", "site": "location", "client": "client", "owner": "client", "customer": "client",
    "submissiondate": "submission_date", "dateofsubmission": "submission_date", "dateofissue": "submission_date",
    "issuedate": "submission_date", "engineer": "engineer", "preparedby": "engineer", "checkedby": "engineer",
}


def _extract_project(document: ExtractedDocument, report: EngineeringReport) -> None:
    patterns = {
        "project_name": re.compile(r"^\s*(?:project(?:\s+name)?|name\s+of\s+project)\s*[:\-]\s*(?P<value>.+)$", re.I),
        "project_no": re.compile(r"^\s*(?:project\s*(?:no\.?|number)|job\s*(?:no\.?|number)|document\s*(?:no\.?|number))\s*[:\-]\s*(?P<value>.+)$", re.I),
        "structure": re.compile(r"^\s*(?:structure|project\s+structure|sub[-\s]?structure|description\s+of\s+work)\s*[:\-]\s*(?P<value>.+)$", re.I),
        "location": re.compile(r"^\s*(?:project\s+location|location|site\s+location|site)\s*[:\-]\s*(?P<value>.+)$", re.I),
        "client": re.compile(r"^\s*(?:client|owner|customer)\s*[:\-]\s*(?P<value>.+)$", re.I),
        "submission_date": re.compile(r"^\s*(?:submission\s+date|date\s+of\s+(?:submission|issue)|date\s+(?:submitted|issued)|issue\s+date)\s*[:\-]\s*(?P<value>.+)$", re.I),
        "engineer": re.compile(r"^\s*(?:engineer|prepared\s+by|checked\s+by|designer)\s*[:\-]\s*(?P<value>.+)$", re.I),
    }
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            clean = _clean(line)
            for key, pattern in patterns.items():
                match = pattern.match(clean)
                if match:
                    value = _clean(match.group("value"))
                    if value:
                        _set_project_value(report, key, _text_value(document, chunk.location, clean, key, value_text=value))
            company = re.match(r"^\s*(?:company|organization|organisation)\s*[:\-]\s*(?P<value>.+)$", clean, re.I)
            if company and report.organization is None:
                value = _clean(company.group("value"))
                if value:
                    report.organization = _text_value(document, chunk.location, clean, "Company", value_text=value)
    for table in document.tables:
        for row_number, row in _row_values(table):
            if len(row) < 2 or not _clean(row[1]):
                continue
            value_text = _clean(row[1])
            key = _PROJECT_TABLE_LABELS.get(_norm(row[0]))
            raw = f"{_clean(row[0])} = {value_text}"
            if key:
                _set_project_value(report, key, _text_value(document, f"{table.location}, row {row_number}", raw, key, value_text=value_text))
            if re.match(r"^\s*(?:company|organization|organisation)\s*$", row[0], re.I) and report.organization is None:
                report.organization = _text_value(document, f"{table.location}, row {row_number}", raw, "Company", value_text=value_text)


def _table_title(document: ExtractedDocument, table: Table) -> str:
    page_match = re.search(r"page\s+(\d+)", table.location, re.I)
    page = page_match.group(1) if page_match else ""
    candidates: list[str] = []
    for chunk in document.text_chunks:
        if page and not re.search(rf"page\s+{re.escape(page)}\b", chunk.location, re.I):
            continue
        for line in chunk.text.splitlines():
            clean = _clean(line)
            if re.match(r"^table\s*:", clean, re.I):
                candidates.append(clean)
    if candidates:
        return candidates[-1]
    header = _clean(" | ".join(table.headers))
    return f"Table columns: {header}" if header else "Engineering table"


def _register_table_reference(report: EngineeringReport, document: ExtractedDocument, table: Table) -> None:
    report.add_reference(f"{_table_title(document, table)} — {document.path.name}, {table.location}.",
                         document.path.name, table.location, raw=table.to_text())


def _make_member_spec(name: str, diameter: ReportValue | None, thickness: ReportValue | None,
                      yield_strength: ReportValue | None, length: ReportValue | None = None,
                      material: ReportValue | None = None) -> MemberSpecification:
    values = {key: value for key, value in (("diameter", diameter), ("thickness", thickness),
                                             ("yield_strength", yield_strength), ("length", length),
                                             ("material", material)) if value is not None}
    return MemberSpecification(name, diameter, thickness, yield_strength, length, material, values)


def _add_member_spec(report: EngineeringReport, specification: MemberSpecification) -> None:
    signature = (specification.name,
                 specification.diameter.value if specification.diameter else None,
                 specification.thickness.value if specification.thickness else None,
                 specification.yield_strength.value if specification.yield_strength else None)
    if not any((item.name, item.diameter.value if item.diameter else None,
                 item.thickness.value if item.thickness else None,
                 item.yield_strength.value if item.yield_strength else None) == signature
                for item in report.member_specifications):
        report.member_specifications.append(specification)


def _section_regex(document: ExtractedDocument) -> re.Match[str] | None:
    pattern = re.compile(r"(?P<diameter>\d+(?:\.\d+)?)\s*[xX]\s*(?P<thickness>\d+(?:\.\d+)?)\s*[xX]\s*(?P<fy>\d+(?:\.\d+)?)\b")
    for chunk in document.text_chunks:
        match = pattern.search(chunk.text)
        if match:
            return match
    return None


def _extract_pile_tables(document: ExtractedDocument, report: EngineeringReport) -> None:
    for table in document.tables:
        start = _data_start(table)
        headers = _column_headers(table, start)
        joined = "".join(headers)
        is_section = (("section" in joined or "pile" in joined or "member" in joined or "plate" in joined)
                      and any(_is_diameter_header(value) for value in headers)
                      and (any(_is_thickness_header(value) for value in headers) or any(_is_fy_header(value) for value in headers)))
        is_helix = any(_is_helix_diameter_header(value) for value in headers) and any(
            _is_spacing_header(value) or _is_helix_diameter_header(value) for value in headers)
        is_corrosion = any("corrosion" in value or "thicknessloss" in value or "steelloss" in value for value in headers)
        if is_section:
            _register_table_reference(report, document, table)
            diameter_cols = _find_columns(table, _is_diameter_header, start)
            thickness_cols = _find_columns(table, _is_thickness_header, start)
            fy_cols = _find_columns(table, _is_fy_header, start)
            fu_cols = _find_columns(table, _is_fu_header, start)
            length_cols = _find_columns(table, lambda value: "length" in value or _norm(value) in {"l", "pile"}, start)
            count_cols = _find_columns(table, _is_count_header, start)
            for row_number, row in _row_values(table):
                if row_number - 2 < start or not row or not _clean(row[0]):
                    continue
                name = _clean(row[0])
                diameter = _cell_value(document, table, row_number, row, diameter_cols[0] if diameter_cols else None, "Pile/Member Diameter (O.D.)")
                thickness = _cell_value(document, table, row_number, row, thickness_cols[0] if thickness_cols else None, "Pile/Member Thickness")
                fy = _cell_value(document, table, row_number, row, fy_cols[0] if fy_cols else None, "Yield Strength (Fy)")
                fu = _cell_value(document, table, row_number, row, fu_cols[0] if fu_cols else None, "Ultimate Strength (Fu)")
                length = _cell_value(document, table, row_number, row, length_cols[0] if length_cols else None, "Pile/Member Length")
                if diameter is not None:
                    report.pile_diameter = _first(report.pile_diameter, diameter)
                    if report.pile_thickness_initial is None:
                        report.pile_thickness_initial = thickness
                    if report.yield_strength is None:
                        report.yield_strength = fy
                report.pile_thickness_table = _first(report.pile_thickness_table, thickness)
                report.ultimate_strength = _first(report.ultimate_strength, fu)
                report.pile_length = _first(report.pile_length, length)
                if report.helix_count is None and count_cols:
                    count = _cell_value(document, table, row_number, row, count_cols[0], "Number of Helices/Plates", "plates")
                    if count is not None:
                        report.helix_count = count
                _add_member_spec(report, _make_member_spec(name, diameter, thickness, fy, length))
                match = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", name)
                if match:
                    if report.pile_diameter is None:
                        report.pile_diameter = _text_value(document, f"{table.location}, row {row_number}", name, "Pile/Member Diameter (O.D.)", value_text=match.group(1))
                    if report.pile_thickness_initial is None:
                        report.pile_thickness_initial = _text_value(document, f"{table.location}, row {row_number}", name, "Pile/Member Thickness", value_text=match.group(2))
                    if report.yield_strength is None:
                        report.yield_strength = _text_value(document, f"{table.location}, row {row_number}", name, "Yield Strength (Fy)", value_text=match.group(3))
        if is_helix:
            _register_table_reference(report, document, table)
            diameter_cols = _find_columns(table, _is_helix_diameter_header, start)
            spacing_cols = _find_columns(table, _is_spacing_header, start)
            count_cols = _find_columns(table, _is_count_header, start)
            for row_number, row in _row_values(table):
                if row_number - 2 < start or not row or not _clean(row[0]):
                    continue
                plate = _clean(row[0])
                if diameter_cols:
                    diameter = _cell_value(document, table, row_number, row, diameter_cols[0], "Helix/Plate Diameter")
                    if diameter is not None:
                        key = f"helix_diameter_{plate}"
                        report.helix_diameters[key] = _first(report.helix_diameters.get(key), diameter) or diameter
                if spacing_cols:
                    spacing = _cell_value(document, table, row_number, row, spacing_cols[0], "Helix/Plate Spacing")
                    if spacing is not None and not any(item.value == spacing.value and item.unit == spacing.unit for item in report.helix_spacing):
                        report.helix_spacing.append(spacing)
                if report.helix_count is None and count_cols:
                    count = _cell_value(document, table, row_number, row, count_cols[0], "Number of Helices/Plates", "plates")
                    if count is not None:
                        report.helix_count = count
                if report.helix_count is None and plate.isdigit():
                    report.helix_count = ReportValue(float(plate), "plates", document.path.name,
                                                       f"{table.location}, row {row_number}", plate,
                                                       "Number of Helices/Plates")
        if is_corrosion:
            _register_table_reference(report, document, table)
            life_col = _find_column(table, lambda value: "designlife" in value or ("design" in value and "life" in value), start)
            loss_col = _find_column(table, lambda value: "corrosionloss" in value or "thicknessloss" in value or "allowableloss" in value, start)
            percent_col = _find_column(table, lambda value: "steelloss" in value or ("steel" in value and "loss" in value), start)
            rate_col = _find_column(table, lambda value: "corrosionrate" in value or ("rate" in value and "corrosion" in value), start)
            for row_number, row in _row_values(table):
                if row_number - 2 < start or not row:
                    continue
                stage = _stage_from_text(row[0]) if row else ""
                for key, column, label in (("design_life", life_col, "Design Life"),
                                            ("corrosion_loss", loss_col, "Corrosion Thickness Loss"),
                                            ("steel_loss_percent", percent_col, "Steel Loss"),
                                            ("corrosion_rate", rate_col, "Corrosion Rate")):
                    if column is None:
                        continue
                    value = _cell_value(document, table, row_number, row, column, label)
                    if value is None:
                        continue
                    if key == "design_life":
                        report.design_life = _first(report.design_life, value)
                    elif key == "corrosion_loss":
                        report.corrosion_loss = _first(report.corrosion_loss, value)
                    elif key == "steel_loss_percent":
                        report.steel_loss_percent = _first(report.steel_loss_percent, value)
                    else:
                        report.corrosion[key] = _first(report.corrosion.get(key), value) or value
                    if stage:
                        report.corrosion_cases[f"{stage}:{key}"] = value
    section_match = _section_regex(document)
    if section_match:
        raw = section_match.group(0)
        report.pile_diameter = report.pile_diameter or _text_value(document, "section text", raw, "Pile/Member Diameter (O.D.)", value_text=section_match.group("diameter"))
        report.pile_thickness_initial = report.pile_thickness_initial or _text_value(document, "section text", raw, "Pile/Member Thickness", value_text=section_match.group("thickness"))
        report.yield_strength = report.yield_strength or _text_value(document, "section text", raw, "Yield Strength (Fy)", value_text=section_match.group("fy"))


def _extract_text_pile(document: ExtractedDocument, report: EngineeringReport) -> None:
    aliases = {
        "diameter": r"pile\s*(?:diameter|dia(?:\\.?|meter)?|section|od)|member\s*(?:diameter|dia(?:\\.?|meter)?)|diameter\s*(?:od)?|outside\s+diameter",
        "thickness": r"(?:pile|member|wall)\s*(?:wall\s*)?thickness|pile\s+thickness|\btp\b",
        "fy": r"yield\s+strength(?:\s*\(\s*fy\s*\))?|\bfy\b",
        "fu": r"ultimate\s+strength(?:\s*\(\s*fu\s*\))?|\bfu\b",
        "length": r"(?:design\s+)?pile\s+length|pile\s+length|embedded\s+depth|member\s+length",
    }
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            clean = _clean(line)
            if not clean:
                continue
            section_match = re.search(r"(?P<diameter>\d+(?:\.\d+)?)\s*[xX]\s*(?P<thickness>\d+(?:\.\d+)?)\s*[xX]\s*(?P<fy>\d+(?:\.\d+)?)", clean, re.I)
            if section_match:
                report.pile_diameter = report.pile_diameter or _text_value(document, chunk.location, clean, "Pile/Member Diameter (O.D.)", value_text=section_match.group("diameter"))
                report.pile_thickness_initial = report.pile_thickness_initial or _text_value(document, chunk.location, clean, "Pile/Member Thickness", value_text=section_match.group("thickness"))
                report.yield_strength = report.yield_strength or _text_value(document, chunk.location, clean, "Yield Strength (Fy)", value_text=section_match.group("fy"))
            for key, label in (("diameter", "Pile/Member Diameter (O.D.)"), ("thickness", "Pile/Member Thickness"),
                               ("fy", "Yield Strength (Fy)"), ("fu", "Ultimate Strength (Fu)"),
                               ("length", "Pile/Member Length")):
                for _, number, unit in _text_number_matches(clean, aliases[key]):
                    value = _text_value(document, chunk.location, clean, label, value_text=f"{number}{(' ' + unit) if unit else ''}")
                    if key == "diameter":
                        report.pile_diameter = _first(report.pile_diameter, value)
                    elif key == "thickness":
                        report.pile_thickness_initial = _first(report.pile_thickness_initial, value)
                    elif key == "fy":
                        report.yield_strength = _first(report.yield_strength, value)
                    elif key == "fu":
                        report.ultimate_strength = _first(report.ultimate_strength, value)
                    else:
                        report.pile_length = _first(report.pile_length, value)
            material = re.search(r"\b(?:material|steel\s+grade|grade)\s*[:=]\s*(?P<value>[A-Za-z][A-Za-z0-9 ._/-]*)", clean, re.I)
            if material:
                report.member_material = _first(report.member_material, _text_value(document, chunk.location, clean, "Pile/Member Material", value_text=_clean(material.group("value"))))
            member_type = re.search(r"\b(?:member|pile)\s+type\s*[:=]\s*(?P<value>[A-Za-z][A-Za-z0-9 ._/-]*)", clean, re.I)
            if member_type:
                report.member_type = _first(report.member_type, _text_value(document, chunk.location, clean, "Pile/Member Type", value_text=_clean(member_type.group("value"))))
            for match in re.finditer(r"\bhelix\s*(?:no\.?|number|count)\s*[:=]\s*(?P<value>[A-Za-z0-9.]+)", clean, re.I):
                report.helix_count = _first(report.helix_count, _text_value(document, chunk.location, clean, "Number of Helices/Plates", value_text=match.group("value")))
            for match in re.finditer(r"\bhelix\s*(?:dia|diameter)\s*(?P<number>[1-9][0-9]*)\s*[:=]\s*(?P<num>" + _NUMBER_PATTERN + r")(?:\s*(?P<unit>" + _UNIT_PATTERN + r"))?", clean, re.I):
                value = _text_value(document, chunk.location, clean, f"Helix/Plate Diameter {match.group('number')}", value_text=f"{match.group('num')}{(' ' + match.group('unit')) if match.group('unit') else ''}")
                key = f"helix_diameter_{match.group('number')}"
                report.helix_diameters[key] = _first(report.helix_diameters.get(key), value) or value
            effective = re.search(r"\b(?:effective|corroded|net)\s+(?:pile\s+)?thickness\s*[:=]\s*(?P<num>" + _NUMBER_PATTERN + r")(?:\s*(?P<unit>" + _UNIT_PATTERN + r"))?", clean, re.I)
            if effective:
                report.pile_thickness_effective = _first(report.pile_thickness_effective, _text_value(document, chunk.location, clean, "Effective/Corroded Pile Thickness", value_text=f"{effective.group('num')}{(' ' + effective.group('unit')) if effective.group('unit') else ''}"))


def _line_stage(line: str) -> str:
    clean = _clean(line)
    patterns = (
        r"\bstage\s*[-:]?\s*(?P<stage>\d+)\b",
        r"\b(?:load\s*case|case|combination)\s*(?:no\.?|#)?\s*[:\-]?\s*(?P<stage>(?:ULS|SLS|SER|STR|EQ|LSD|LC)[-_ ]?\d+)\b",
        r"(?P<stage>(?:ULS|SLS|SER|STR|EQ|LSD|LC)[-_ ]?\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, clean, re.I)
        if match:
            return _stage_from_text(match.group("stage"))
    match = re.search(r"\b(ULS|SLS|SER|STR|EQ|LSD|LC)\b", clean, re.I)
    return match.group(1).upper() if match else ""


def _line_load_type(line: str) -> str:
    match = re.search(r"\(\s*(compression|tension|shear|lateral|moment|bending)\s*\)", line, re.I)
    if match:
        return match.group(1).title()
    match = re.search(r"\b(?:load\s*)?type\s*[:=]\s*(?P<value>[A-Za-z][A-Za-z0-9 _/-]*)", line, re.I)
    return _clean(match.group("value")).strip(" ,;") if match else ""


def _text_label_values(line: str) -> list[tuple[str, re.Match[str], str, str | None]]:
    definitions = [
        ("compression", r"compression(?:\s+load)?|comp(?:\s+load)?|axial\s+compression|axial\s+comp"),
        ("tension", r"tension(?:\s+load)?|uplift(?:\s+load)?|pullout(?:\s+load)?|tensile\s+load"),
        ("fx", r"fxx|fx|shear\s*(?:force\s*)?x|lateral\s*(?:force\s*)?x|force\s*x"),
        ("fy", r"fyy|fy|shear\s*(?:force\s*)?y|lateral\s*(?:force\s*)?y|force\s*y"),
        ("mxx", r"mxx|mx|bending\s+moment\s*x|moment\s*x|bending\s+moment"),
        ("myy", r"myy|my|bending\s+moment\s*y|moment\s*y"),
        ("settlement", r"settlement|settle|smax|predicted\s+settlement|design\s+settlement|ultimate\s+settlement|max\s+settlement"),
        ("corrosion_loss", r"corrosion\s+(?:thickness\s+)?loss|thickness\s+loss|steel\s+loss|corrosion\s+allowance"),
        ("design_life", r"design\s+life|service\s+life"),
        ("lateral_deflection", r"lateral\s+deflection|max\s+deflection|deflection|displacement"),
        ("lateral_rotation", r"rotation|max\s+rotation"),
    ]
    found: list[tuple[str, re.Match[str], str, str | None]] = []
    for key, label_pattern in definitions:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9])(?P<label>{label_pattern})(?:\s*\((?P<parenunit>[^)]*)\))?\s*(?:[:=]\s*|\s+)(?P<num>{_NUMBER_PATTERN})(?:\s*(?P<unit>{_UNIT_PATTERN}))?"
        )
        for match in pattern.finditer(line):
            tail = line[match.end():match.end() + 30].lower()
            if key in {"compression", "tension", "fx", "fy", "mxx", "myy"} and re.match(r"\s*(?:capacity|ratio|allowable|demand|resistance)\b", tail):
                continue
            if key == "settlement" and re.search(r"\b(?:curve|graph|ordinate|coordinate|settlement\s*y)\b", tail, re.I):
                continue
            found.append((key, match, match.group("num"), match.group("unit") or match.group("parenunit") or None))
    return found


def _text_capacity_values(line: str) -> list[tuple[str, re.Match[str], str, str | None]]:
    definitions = [
        ("compression_capacity", r"compression\s+capacity|capacity\s+compression|comp\s+capacity"),
        ("tension_capacity", r"tension\s+capacity|uplift\s+capacity|capacity\s+tension"),
        ("geotechnical_capacity", r"geotechnical\s+capacity|geo\s+capacity|design\s+capacity|allowable\s+capacity|q\s*ult|q\s*all"),
        ("structural_capacity", r"structural\s+capacity|struct\s+capacity|steel\s+capacity"),
        ("r_geo", r"r\s*/\s*geo|rgeo|geotechnical\s+ratio|geo\s+ratio"),
        ("r_str", r"r\s*/\s*str|rstr|structural\s+ratio|struct\s+ratio"),
        ("r_cap", r"r\s*/\s*cap|rcap|overall\s+(?:critical\s+)?ratio|load\s*/\s*cap|capacity\s+ratio"),
    ]
    found: list[tuple[str, re.Match[str], str, str | None]] = []
    for key, label_pattern in definitions:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9])(?P<label>{label_pattern})(?:\s*\((?P<parenunit>[^)]*)\))?\s*(?:[:=]\s*|\s+)(?P<num>{_NUMBER_PATTERN})(?:\s*(?P<unit>{_UNIT_PATTERN}))?"
        )
        for match in pattern.finditer(line):
            found.append((key, match, match.group("num"), match.group("unit") or match.group("parenunit") or None))
    return found


def _row_stage(table: Table, row: list[str], stage_columns: list[int]) -> str:
    for column in stage_columns:
        if column < len(row):
            value = row[column]
            stage = _stage_from_text(value)
            if stage and not _is_headerish(value):
                return stage
    for value in row:
        if _is_stage_text(value):
            return _stage_from_text(value)
    return ""


def _set_capacity_value(report: EngineeringReport, stage: str, key: str, value: ReportValue) -> None:
    capacity = _capacity_for(report, stage)
    setattr(capacity, key, _first(getattr(capacity, key), value) or value)
    load = _case_for(report, stage)
    load_key = {"compression_capacity": "compression_capacity", "tension_capacity": "tension_capacity",
                "geotechnical_capacity": "geotechnical_capacity", "structural_capacity": "structural_capacity"}.get(key)
    if load_key:
        setattr(load, load_key, _first(getattr(load, load_key), value) or value)
    ratio_key = {"r_geo": "geotechnical", "r_str": "structural", "r_cap": "overall"}.get(key)
    if ratio_key:
        ratio = _ratio_for(report, stage)
        setattr(ratio, ratio_key, _first(getattr(ratio, ratio_key), value) or value)
        setattr(load, key, value)
        setattr(capacity, key, getattr(ratio, ratio_key))


def _set_failure_mode(report: EngineeringReport, stage: str, value: ReportValue) -> None:
    ratio = _ratio_for(report, stage)
    ratio.failure_mode = _first(ratio.failure_mode, value) or value
    _capacity_for(report, stage).failure_mode = ratio.failure_mode


def _extract_load_table(document: ExtractedDocument, report: EngineeringReport, table: Table) -> None:
    start = _data_start(table)
    headers = _column_headers(table, start)
    joined = "".join(headers)
    stage_columns = _find_columns(table, _is_stage_header, start)
    has_stage = bool(stage_columns) or any(_is_stage_text(row[0]) for row in table.rows[start:] if row)
    compression_cols = _find_columns(table, _is_compression_header, start)
    tension_cols = _find_columns(table, _is_tension_header, start)
    axial_cols = _find_columns(table, _is_axial_header, start)
    fx_cols = _find_columns(table, _is_fx_header, start)
    fy_cols = _find_columns(table, _is_fy_header, start)
    mxx_cols = _find_columns(table, _is_mxx_header, start)
    myy_cols = _find_columns(table, _is_myy_header, start)
    type_cols = _find_columns(table, _is_type_header, start)
    capacity_cols = {key: _find_columns(table, lambda value, kind=kind: _capacity_kind(value) == kind, start)
                     for key, kind in (("compression_capacity", "compression"), ("tension_capacity", "tension"),
                                       ("geotechnical_capacity", "geotechnical"), ("structural_capacity", "structural"))}
    generic_capacity_cols = _find_columns(table, lambda value: _capacity_kind(value) == "generic", start)
    ratio_cols = {key: _find_columns(table, lambda value, kind=kind: _ratio_kind(value) == kind, start)
                  for key, kind in (("r_geo", "geo"), ("r_str", "str"), ("r_cap", "cap"))}
    has_values = bool(compression_cols or tension_cols or axial_cols or fx_cols or fy_cols or mxx_cols or myy_cols or type_cols or any(capacity_cols.values()) or generic_capacity_cols or any(ratio_cols.values()))
    if not has_stage or not has_values:
        return
    if not stage_columns and not any(token in joined for token in ("load", "stage", "case", "combination")):
        return
    _register_table_reference(report, document, table)
    for row_number, row in _row_values(table):
        if row_number - 2 < start or not row:
            continue
        stage = _row_stage(table, row, stage_columns)
        if not stage or _is_headerish(stage):
            continue
        case = _case_for(report, stage)
        for column in compression_cols:
            value = _cell_value(document, table, row_number, row, column, "Compression")
            if value is not None:
                _set_prioritized(case, "compression", value, 3)
        for column in tension_cols:
            value = _cell_value(document, table, row_number, row, column, "Tension")
            if value is not None:
                _set_prioritized(case, "tension", value, 3)
        for column in axial_cols:
            value = _cell_value(document, table, row_number, row, column, "Axial Load")
            if value is not None:
                _set_prioritized(case, "axial_load", value, 2)
                if isinstance(value.value, (int, float)) and value.value < 0:
                    _set_prioritized(case, "tension", _signed_value(value), 2)
                else:
                    _set_prioritized(case, "compression", value, 2)
        for column in fx_cols:
            value = _cell_value(document, table, row_number, row, column, "Fx")
            if value is not None:
                _set_prioritized(case, "fx", value, 3)
        for column in fy_cols:
            value = _cell_value(document, table, row_number, row, column, "Fy")
            if value is not None:
                _set_prioritized(case, "fy", value, 3)
        for column in mxx_cols:
            value = _cell_value(document, table, row_number, row, column, "Mxx")
            if value is not None:
                _set_prioritized(case, "mxx", value, 3)
        for column in myy_cols:
            value = _cell_value(document, table, row_number, row, column, "Myy")
            if value is not None:
                _set_prioritized(case, "myy", value, 3)
        for column in type_cols:
            value = _cell_value(document, table, row_number, row, column, "Load Type")
            if value is not None:
                _set_prioritized(case, "load_type", value, 3)
        for key, columns in capacity_cols.items():
            for column in columns:
                value = _cell_value(document, table, row_number, row, column, key.replace("_", " ").title())
                if value is not None:
                    _set_capacity_value(report, stage, key, value)
        for column in generic_capacity_cols:
            value = _cell_value(document, table, row_number, row, column, "Capacity")
            if value is not None:
                _set_capacity_value(report, stage, "geotechnical_capacity", value)
        for key, columns in ratio_cols.items():
            for column in columns:
                value = _cell_value(document, table, row_number, row, column, key.replace("_", "/").upper())
                if value is not None:
                    _set_capacity_value(report, stage, key, value)
        mode_col = _find_column(table, lambda value: "failuremode" in value or "governingmode" in value or value in {"mode", "failuretype"}, start)
        if mode_col is not None:
            mode = _cell_value(document, table, row_number, row, mode_col, "Failure Mode")
            if mode is not None:
                _set_failure_mode(report, stage, mode)


def _extract_load_text(document: ExtractedDocument, report: EngineeringReport) -> None:
    current_stage = ""
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            clean = _clean(line)
            if not clean:
                continue
            stage = _line_stage(clean)
            if stage:
                current_stage = stage
            if not current_stage:
                continue
            values = _text_label_values(clean)
            for key, _, number, unit in values:
                if key not in {"compression", "tension", "fx", "fy", "mxx", "myy"}:
                    continue
                value = _text_value(document, chunk.location, clean, key, value_text=f"{number}{(' ' + unit) if unit else ''}")
                if key == "compression" and isinstance(value.value, (int, float)) and value.value < 0:
                    _set_prioritized(_case_for(report, current_stage), "tension", _signed_value(value), 2)
                else:
                    _set_prioritized(_case_for(report, current_stage), key, value, 2)
            if re.search(r"\b(?:axial\s+)?load\b", clean, re.I):
                axial = next((item for item in values if item[0] in {"compression", "tension"}), None)
                if axial:
                    value = _text_value(document, chunk.location, clean, "Axial Load", value_text=f"{axial[2]}{(' ' + axial[3]) if axial[3] else ''}")
                    _set_prioritized(_case_for(report, current_stage), "axial_load", value, 1)
            load_type = _line_load_type(clean)
            if load_type:
                _set_prioritized(_case_for(report, current_stage), "load_type", _text_value(document, chunk.location, clean, "Load Type", value_text=load_type), 2)
            elif values:
                inferred = next((key for key, _, _, _ in values if key in {"compression", "tension", "fx", "fy", "mxx", "myy"}), "")
                if inferred:
                    _set_prioritized(_case_for(report, current_stage), "load_type", _text_value(document, chunk.location, clean, "Load Type", value_text=inferred.title()), 1)


def _extract_soil_tables(document: ExtractedDocument, report: EngineeringReport) -> None:
    for table in document.tables:
        start = _data_start(table)
        headers = _column_headers(table, start)
        if not any(_is_soil_header(value) for value in headers) and not any("phi" in value or "cohesion" in value or "frictionangle" in value for value in headers):
            continue
        if any("settlement" in value for value in headers) and not any("phi" in value or "cohesion" in value for value in headers):
            continue
        _register_table_reference(report, document, table)
        name_columns = _find_columns(table, _is_soil_header, start)
        name_column = name_columns[0] if name_columns else 0
        parameters = [(column, _soil_parameter_name(header)) for column, header in enumerate(headers)]
        parameters = [(column, name) for column, name in parameters if name and name != "name"]
        for row_number, row in _row_values(table):
            if row_number - 2 < start or not row:
                continue
            name = _clean(row[name_column]) if name_column < len(row) else ""
            values: dict[str, ReportValue] = {}
            for column, key in parameters:
                value = _cell_value(document, table, row_number, row, column, key.replace("_", " ").title())
                if value is not None:
                    values[key] = value
            if name and "name" not in values and name_column < len(row):
                values["name"] = ReportValue(name, None, document.path.name, f"{table.location}, row {row_number}", row[name_column].strip(), "Soil Layer Name")
            if not values:
                continue
            layer = SoilLayer(name, values, document.path.name, f"{table.location}, row {row_number}", " | ".join(cell.strip() for cell in row if cell.strip()))
            report.add_soil_layer(layer)
            for key, value in values.items():
                if key in {"phi", "cohesion", "su", "unit_weight"}:
                    report.soil_parameters.setdefault(f"soil_parameters_{key}", value)
                    report.soil_parameters.setdefault(key, value)


def _extract_soil_text(document: ExtractedDocument, report: EngineeringReport) -> None:
    layer_pattern = re.compile(r"\b(?:soil\s+)?layer\s*(?P<name>[A-Za-z0-9][A-Za-z0-9 _-]*)?\s*[:\-]\s*(?P<rest>.+)$", re.I)
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            clean = _clean(line)
            match = layer_pattern.search(clean)
            if not match:
                continue
            name = _clean(match.group("name") or "")
            rest = _clean(match.group("rest"))
            values: dict[str, ReportValue] = {}
            if name:
                values["name"] = ReportValue(name, None, document.path.name, chunk.location, clean, "Soil Layer Name")
            definitions = (("phi", r"phi|friction\s+angle", "Friction Angle (Phi)"),
                           ("cohesion", r"cohesion|\bc\b", "Cohesion"),
                           ("su", r"\bsu\b|undrained\s+shear\s+strength", "Undrained Shear Strength"),
                           ("unit_weight", r"unit\s+weight|gamma", "Unit Weight"),
                           ("depth_top", r"top\s+depth|depth\s+from|depth", "Depth From"),
                           ("depth_bottom", r"bottom\s+depth|depth\s+to", "Depth To"),
                           ("thickness", r"thickness", "Layer Thickness"))
            for key, label_pattern, label in definitions:
                for _, number, unit in _text_number_matches(rest, label_pattern):
                    value = _text_value(document, chunk.location, clean, label, value_text=f"{number}{(' ' + unit) if unit else ''}")
                    values[key] = _first(values.get(key), value) or value
            if values:
                report.add_soil_layer(SoilLayer(name, values, document.path.name, chunk.location, clean))
                for key in ("phi", "cohesion", "su", "unit_weight"):
                    value = values.get(key)
                    if value is not None:
                        report.soil_parameters.setdefault(f"soil_parameters_{key}", value)
                        report.soil_parameters.setdefault(key, value)


def _extract_ratio_tables(document: ExtractedDocument, report: EngineeringReport) -> None:
    for table in document.tables:
        start = _data_start(table)
        stage_columns = _find_columns(table, _is_stage_header, start)
        columns = {kind: _find_columns(table, lambda value, expected=kind: _ratio_kind(value) == expected, start)
                   for kind in ("geo", "str", "cap")}
        if not any(columns.values()) or (not stage_columns and not any(_is_stage_text(row[0]) for row in table.rows[start:] if row)):
            continue
        _register_table_reference(report, document, table)
        mode_col = _find_column(table, lambda value: "failuremode" in value or "governingmode" in value, start)
        for row_number, row in _row_values(table):
            if row_number - 2 < start or not row:
                continue
            stage = _row_stage(table, row, stage_columns)
            if not stage or _is_headerish(stage):
                continue
            for kind, found in columns.items():
                for column in found:
                    value = _cell_value(document, table, row_number, row, column, {"geo": "R/GEO", "str": "R/STR", "cap": "R/CAP"}[kind])
                    if value is not None:
                        _set_capacity_value(report, stage, {"geo": "r_geo", "str": "r_str", "cap": "r_cap"}[kind], value)
            if mode_col is not None:
                mode = _cell_value(document, table, row_number, row, mode_col, "Failure Mode")
                if mode is not None:
                    _set_failure_mode(report, stage, mode)


def _extract_settlement_tables(document: ExtractedDocument, report: EngineeringReport) -> None:
    for table in document.tables:
        start = _data_start(table)
        headers = _column_headers(table, start)
        columns = _find_columns(table, _is_settlement_header, start)
        if not columns or (len(table.rows) > 10 and any("settlementy" in value or "ordinate" in value for value in headers)):
            continue
        _register_table_reference(report, document, table)
        stage_columns = _find_columns(table, _is_stage_header, start)
        for row_number, row in _row_values(table):
            if row_number - 2 < start or not row:
                continue
            stage = _row_stage(table, row, stage_columns)
            if not stage or _is_headerish(stage):
                continue
            case = _settlement_for(report, stage)
            for column in columns:
                value = _cell_value(document, table, row_number, row, column, "Settlement")
                if value is None:
                    continue
                header = _norm(_column_text(table, column, start))
                if "design" in header or "predicted" in header:
                    case.design_settlement = _first(case.design_settlement, value) or value
                elif "ultimate" in header:
                    case.ultimate_settlement = _first(case.ultimate_settlement, value) or value
                elif "max" in header:
                    case.max_settlement = _first(case.max_settlement, value) or value
                else:
                    case.settlement = _first(case.settlement, value) or value
                if not case.settlement:
                    case.settlement = value
                report.settlements[stage] = _first(report.settlements.get(stage), value) or value


def _extract_lateral_tables(document: ExtractedDocument, report: EngineeringReport) -> None:
    for table in document.tables:
        start = _data_start(table)
        headers = _column_headers(table, start)
        found = {
            "fx": _find_columns(table, _is_fx_header, start),
            "fy": _find_columns(table, _is_fy_header, start),
            "mxx": _find_columns(table, _is_mxx_header, start),
            "myy": _find_columns(table, _is_myy_header, start),
            "max_deflection": _find_columns(table, _is_deflection_header, start),
            "rotation": _find_columns(table, lambda value: _norm(value) in {"rotation", "maxrotation"}, start),
            "lateral_capacity": _find_columns(table, lambda value: ("lateralcapacity" in value or "latcapacity" in value) and "load" not in value, start),
        }
        if not any(found.values()):
            continue
        if len(table.rows) > 10 and not (found["max_deflection"] or found["lateral_capacity"] or "lateral" in "".join(headers) or "latforce" in "".join(headers)):
            continue
        _register_table_reference(report, document, table)
        stage_columns = _find_columns(table, _is_stage_header, start)
        for row_number, row in _row_values(table):
            if row_number - 2 < start or not row:
                continue
            stage = _row_stage(table, row, stage_columns)
            if not stage or _is_headerish(stage):
                continue
            lateral = _lateral_for(report, stage)
            for key, columns in found.items():
                for column in columns:
                    value = _cell_value(document, table, row_number, row, column, key.replace("_", " ").title())
                    if value is None:
                        continue
                    setattr(lateral, key, _first(getattr(lateral, key), value) or value)
                    if key in {"fx", "fy", "mxx", "myy"}:
                        load_key = {"fx": "fx", "fy": "fy", "mxx": "internal_moment", "myy": "internal_moment_other"}[key]
                        _set_prioritized(_case_for(report, stage), load_key, value, 2)


def _extract_settlement_text(document: ExtractedDocument, report: EngineeringReport) -> None:
    current_stage = ""
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            clean = _clean(line)
            stage = _line_stage(clean)
            if stage:
                current_stage = stage
            if not current_stage or not re.search(r"settle", clean, re.I):
                continue
            for key, _, number, unit in _text_label_values(clean):
                if key != "settlement":
                    continue
                value = _text_value(document, chunk.location, clean, "Settlement", value_text=f"{number}{(' ' + unit) if unit else ''}")
                case = _settlement_for(report, current_stage)
                case.settlement = _first(case.settlement, value) or value
                report.settlements[current_stage] = _first(report.settlements.get(current_stage), value) or value
            if re.search(r"\b(?:warning|settlement|response)\b", clean, re.I):
                warning = _text_value(document, chunk.location, clean, "Settlement Warning", value_text=clean)
                _settlement_for(report, current_stage).warning = _first(_settlement_for(report, current_stage).warning, warning) or warning
                message = f"{clean} ({document.path.name}, {chunk.location})"
                if message not in report.settlement_warnings:
                    report.settlement_warnings.append(message)


def _extract_lateral_text(document: ExtractedDocument, report: EngineeringReport) -> None:
    current_stage = ""
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            clean = _clean(line)
            stage = _line_stage(clean)
            if stage:
                current_stage = stage
            if not current_stage:
                continue
            for key, _, number, unit in _text_label_values(clean):
                if key not in {"fx", "fy", "mxx", "myy", "lateral_deflection", "lateral_rotation"}:
                    continue
                value = _text_value(document, chunk.location, clean, key, value_text=f"{number}{(' ' + unit) if unit else ''}")
                lateral = _lateral_for(report, current_stage)
                target = "max_deflection" if key == "lateral_deflection" else "rotation" if key == "lateral_rotation" else key
                setattr(lateral, target, _first(getattr(lateral, target), value) or value)
                load_key = {"fx": "fx", "fy": "fy", "mxx": "internal_moment", "myy": "internal_moment_other"}.get(key)
                if load_key:
                    _set_prioritized(_case_for(report, current_stage), load_key, value, 2)
            match = re.search(r"\blateral\s+capacity\s*[:=]?\s*(?P<num>" + _NUMBER_PATTERN + r")(?:\s*(?P<unit>" + _UNIT_PATTERN + r"))?", clean, re.I)
            if match:
                _lateral_for(report, current_stage).lateral_capacity = _text_value(document, chunk.location, clean, "Lateral Capacity", value_text=f"{match.group('num')}{(' ' + match.group('unit')) if match.group('unit') else ''}")


def _extract_ratio_text(document: ExtractedDocument, report: EngineeringReport) -> None:
    current_stage = ""
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            clean = _clean(line)
            stage = _line_stage(clean)
            if stage:
                current_stage = stage
            if not current_stage:
                continue
            for key, _, number, unit in _text_capacity_values(clean):
                value = _text_value(document, chunk.location, clean, key, value_text=f"{number}{(' ' + unit) if unit else ''}")
                _set_capacity_value(report, current_stage, key, value)


def _extract_corrosion_text(document: ExtractedDocument, report: EngineeringReport) -> None:
    definitions = (("design_life", r"design\s+life|service\s+life", "Design Life"),
                   ("corrosion_loss", r"corrosion\s+(?:thickness\s+)?loss|thickness\s+loss|steel\s+loss|corrosion\s+allowance", "Corrosion Thickness Loss"),
                   ("steel_loss_percent", r"steel\s+loss(?:\s+percent)?", "Steel Loss"),
                   ("corrosion_rate", r"corrosion\s+rate", "Corrosion Rate"))
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            clean = _clean(line)
            for key, label_pattern, label in definitions:
                for _, number, unit in _text_number_matches(clean, label_pattern):
                    value = _text_value(document, chunk.location, clean, label, value_text=f"{number}{(' ' + unit) if unit else ''}")
                    if key == "design_life":
                        report.design_life = _first(report.design_life, value)
                    elif key == "corrosion_loss":
                        report.corrosion_loss = _first(report.corrosion_loss, value)
                    elif key == "steel_loss_percent":
                        report.steel_loss_percent = _first(report.steel_loss_percent, value)
                    else:
                        report.corrosion[key] = _first(report.corrosion.get(key), value) or value
