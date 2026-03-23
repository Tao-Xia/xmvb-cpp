from __future__ import annotations

import argparse
import json
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an F2 bond-length scan input set")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("/home/xiatao/vb_project/xmvb-cpp/test_molecule/F2.xmi"),
        help="Path to the template F2 .xmi file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_inputs/F2_scan"),
        help="Directory where the scanned inputs will be written",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="",
        help="Comma-separated bond lengths. Overrides the default scan grid when provided.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="F2 VBSCF scan",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    return parser.parse_args()


def _quantize_length(length: float) -> Decimal:
    return Decimal(str(length)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def _default_bond_lengths() -> list[Decimal]:
    lengths: list[Decimal] = []

    def add_range(start: str, stop: str, step: str) -> None:
        current = Decimal(start)
        stop_decimal = Decimal(stop)
        step_decimal = Decimal(step)
        while current <= stop_decimal + Decimal("1e-12"):
            lengths.append(current.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
            current += step_decimal

    add_range("1.100", "1.800", "0.050")
    add_range("1.900", "3.000", "0.100")
    add_range("3.200", "4.000", "0.200")
    return lengths


def _parse_lengths_arg(value: str) -> list[Decimal]:
    if not value.strip():
        return _default_bond_lengths()
    lengths: list[Decimal] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        lengths.append(_quantize_length(float(stripped)))
    if not lengths:
        raise ValueError("no valid bond lengths were provided")
    unique_sorted = sorted(set(lengths))
    return unique_sorted


def _stem_for_length(length: Decimal) -> str:
    text = f"{length:.3f}"
    return "F2_r" + text.replace("-", "m").replace(".", "p")


def _load_template(template_path: Path) -> list[str]:
    if template_path.suffix != ".xmi":
        raise ValueError(f"template must be a .xmi file: {template_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"template does not exist: {template_path}")
    return template_path.read_text().splitlines()


def _replace_geometry_and_title(
    template_lines: list[str],
    *,
    bond_length: Decimal,
    title_prefix: str,
) -> str:
    lines = list(template_lines)
    if not lines:
        raise ValueError("template .xmi is empty")
    lines[0] = f"{title_prefix} r={bond_length:.3f}"

    try:
        geo_start = lines.index("$GEO")
    except ValueError as error:
        raise ValueError("template .xmi does not contain a $GEO block") from error

    geo_end = None
    for index in range(geo_start + 1, len(lines)):
        if lines[index].strip() == "$END":
            geo_end = index
            break
    if geo_end is None:
        raise ValueError("template .xmi has an unterminated $GEO block")

    geometry_lines = [line for line in lines[geo_start + 1 : geo_end] if line.strip()]
    if len(geometry_lines) != 2:
        raise ValueError("the F2 scan generator expects exactly two atoms in the $GEO block")

    first_tokens = geometry_lines[0].split()
    second_tokens = geometry_lines[1].split()
    if len(first_tokens) != 4 or len(second_tokens) != 4:
        raise ValueError("expected Cartesian geometry lines with four columns")
    if first_tokens[0] != "F" or second_tokens[0] != "F":
        raise ValueError("the F2 scan generator expects an F2 template")

    first_tokens[1:] = ["0.0", "0.0", "0.0"]
    second_tokens[1:] = ["0.0", "0.0", f"{bond_length:.3f}"]
    lines[geo_start + 1] = " ".join(first_tokens)
    lines[geo_start + 2] = " ".join(second_tokens)
    return "\n".join(lines) + "\n"


def _copy_sidecars(template_path: Path, output_stem: Path) -> list[str]:
    copied_extensions: list[str] = []
    for extension in (".orb", ".str", ".xdat"):
        source_path = template_path.with_suffix(extension)
        if source_path.exists():
            destination_path = output_stem.with_suffix(extension)
            destination_path.write_bytes(source_path.read_bytes())
            copied_extensions.append(extension)
    return copied_extensions


def main() -> None:
    args = _parse_args()
    template_path = args.template.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"output directory already exists and is non-empty: {output_dir}; "
                "pass --overwrite to reuse it"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    template_lines = _load_template(template_path)
    bond_lengths = _parse_lengths_arg(args.lengths)

    manifest_records: list[dict[str, object]] = []
    for bond_length in bond_lengths:
        stem_name = _stem_for_length(bond_length)
        output_stem = output_dir / stem_name
        xmi_contents = _replace_geometry_and_title(
            template_lines,
            bond_length=bond_length,
            title_prefix=args.title_prefix,
        )
        output_stem.with_suffix(".xmi").write_text(xmi_contents)
        copied_extensions = _copy_sidecars(template_path, output_stem)
        manifest_records.append(
            {
                "sample_name": stem_name,
                "bond_length": float(bond_length),
                "xmi_path": str(output_stem.with_suffix(".xmi")),
                "copied_sidecars": copied_extensions,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_records, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "template": str(template_path),
                "output_dir": str(output_dir),
                "n_geometries": len(manifest_records),
                "manifest_path": str(manifest_path),
                "first_sample": None if not manifest_records else manifest_records[0]["sample_name"],
                "last_sample": None if not manifest_records else manifest_records[-1]["sample_name"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
