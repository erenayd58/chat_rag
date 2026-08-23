"""Regenerate the seam fixture PDF.

Run with: py -3.11 tests/fixtures/seam/build_mini_report.py

The fixture is committed, so this script exists only to document exactly how it
was produced and to allow deliberate regeneration.  Regenerating changes the
sha256 in manifest.json.
"""

from __future__ import annotations

import pathlib

import pymupdf as fitz

OUTPUT = pathlib.Path(__file__).with_name("mini-report.pdf")


def build() -> pathlib.Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    state = {"y": 70}

    def write(text: str, size: int, bold: bool = False, dy: int | None = None) -> None:
        page.insert_text(
            (60, state["y"]), text, fontsize=size, fontname="hebo" if bold else "helv"
        )
        state["y"] += dy if dy is not None else size + 8

    write("Kredi Risk Yonetimi", 22, bold=True, dy=40)
    write("Bu bolum, kurumun kredi riskini nasil olctugunu ve yonettigini aciklar.", 10)
    write("Risk istahi yillik olarak yonetim kurulu tarafindan onaylanir ve ceyreklik", 10)
    write("olarak gozden gecirilir.", 10, dy=30)

    write("Risk Gostergeleri", 16, bold=True, dy=30)
    write("Asagidaki tablo son iki yilin temel gostergelerini karsilastirir.", 10, dy=25)

    rows = [
        ("Gosterge", "2024", "2023"),
        ("Takipteki alacak orani", "2,4%", "2,9%"),
        ("Karsilik orani", "78,1%", "74,5%"),
        ("Sermaye yeterliligi", "18,3%", "17,2%"),
    ]
    widths = [200, 90, 90]
    for row_index, row in enumerate(rows):
        x = 60
        for column_index, cell in enumerate(row):
            page.draw_rect(
                fitz.Rect(x, state["y"] - 12, x + widths[column_index], state["y"] + 6),
                color=(0, 0, 0),
                width=0.6,
            )
            page.insert_text(
                (x + 5, state["y"]),
                cell,
                fontsize=9,
                fontname="hebo" if row_index == 0 else "helv",
            )
            x += widths[column_index]
        state["y"] += 18
    state["y"] += 25

    write("Kontrol Adimlari", 16, bold=True, dy=30)
    for item in (
        "Limit tahsisi ikinci bir birim tarafindan dogrulanir.",
        "Teminat degerlemesi bagimsiz eksperlerce yapilir.",
        "Erken uyari sinyalleri haftalik izlenir.",
    ):
        write("\u2022 " + item, 10)

    second = doc.new_page(width=595, height=842)
    second.insert_text((60, 70), "Operasyonel Risk", fontsize=22, fontname="hebo")
    second.insert_text(
        (60, 120),
        "Operasyonel risk, yetersiz ic surecler veya dis olaylardan",
        fontsize=10,
        fontname="helv",
    )
    second.insert_text(
        (60, 134), "kaynaklanan kayip riskidir.", fontsize=10, fontname="helv"
    )

    doc.save(OUTPUT)
    doc.close()
    return OUTPUT


if __name__ == "__main__":
    print(build())
