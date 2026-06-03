#!/usr/bin/env python3
"""Create an XLSX review sheet with dropdowns from a review CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


DROPDOWNS = {
    "has_crack": ["yes", "no", "unclear"],
    "detect_quality": ["good", "usable", "poor", "fail"],
    "miss_level": ["none", "slight", "medium", "severe"],
    "false_positive_level": ["none", "slight", "medium", "severe"],
    "fragment_level": ["none", "slight", "medium", "severe"],
    "width_length_reliable": ["yes", "no", "unclear"],
    "typical_case": ["good", "miss", "false_positive", "fragmented", "empty", "duplicate", "unclear"],
    "next_action": ["keep", "threshold_down", "threshold_up", "postprocess", "need_label", "ignore"],
}

WIDTHS = {
    "review_order": 12,
    "image_stem": 30,
    "original_path": 48,
    "preview_path": 72,
    "overlay_path": 72,
    "length_overlay_path": 78,
    "mask_path": 68,
    "prob_path": 68,
    "mask_ratio": 12,
    "skeleton_components": 18,
    "length_m": 10,
    "auto_priority": 12,
    "auto_risk": 54,
    "notes": 48,
}


def col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_ref(row: int, col: int) -> str:
    return f"{col_name(col)}{row}"


def xml_text(value: object) -> str:
    return escape(str(value), {'"': "&quot;"})


def cell_xml(row: int, col: int, value: str, style: int | None = None) -> str:
    attrs = [f'r="{cell_ref(row, col)}"']
    if style is not None:
        attrs.append(f's="{style}"')
    if value == "":
        return f"<c {' '.join(attrs)}/>"
    attrs.append('t="inlineStr"')
    return f"<c {' '.join(attrs)}><is><t>{xml_text(value)}</t></is></c>"


def sheet_xml(headers: list[str], rows: list[dict[str, str]]) -> str:
    max_row = len(rows) + 1
    max_col = len(headers)
    dimensions = f"A1:{cell_ref(max_row, max_col)}"

    cols = []
    for index, header in enumerate(headers, start=1):
        width = WIDTHS.get(header, 16)
        cols.append(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')

    sheet_rows = []
    header_cells = [cell_xml(1, col, header, style=1) for col, header in enumerate(headers, start=1)]
    sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')
    for row_index, row in enumerate(rows, start=2):
        cells = [
            cell_xml(row_index, col, row.get(header, ""))
            for col, header in enumerate(headers, start=1)
        ]
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    validations = []
    for header, values in DROPDOWNS.items():
        if header not in headers:
            continue
        col = col_name(headers.index(header) + 1)
        formula = ",".join(values)
        validations.append(
            (
                f'<dataValidation type="list" allowBlank="1" showErrorMessage="1" '
                f'sqref="{col}2:{col}{max_row}">'
                f"<formula1>&quot;{xml_text(formula)}&quot;</formula1>"
                "</dataValidation>"
            )
        )
    validation_xml = ""
    if validations:
        validation_xml = f'<dataValidations count="{len(validations)}">{"".join(validations)}</dataValidations>'

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimensions}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>{"".join(cols)}</cols>
  <sheetData>{"".join(sheet_rows)}</sheetData>
  <autoFilter ref="{dimensions}"/>
  {validation_xml}
</worksheet>
'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>
'''


def write_xlsx(csv_path: Path, xlsx_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
''',
        )
        archive.writestr(
            "_rels/.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
''',
        )
        archive.writestr(
            "xl/workbook.xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="review" sheetId="1" r:id="rId1"/></sheets>
</workbook>
''',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
''',
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml(headers, rows))
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr(
            "docProps/core.xml",
            f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>
''',
        )
        archive.writestr(
            "docProps/app.xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
''',
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-xlsx", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_xlsx(args.input_csv, args.output_xlsx)


if __name__ == "__main__":
    main()
