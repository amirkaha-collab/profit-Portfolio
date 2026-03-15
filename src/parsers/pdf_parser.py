"""
PDF parser with three extraction strategies (in priority order):

1. pdfplumber  – works well for text-based PDFs with clear table structure
2. camelot     – lattice / stream mode for gridded tables
3. pymupdf     – faster text extraction fallback (no table awareness)

Each strategy is attempted in order; the one that returns the most plausible
holdings table wins.  If all fail, an error is recorded and an empty result
is returned so the caller can surface a clear message to the user.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from .base import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """
    Extract holdings tables from PDF statements.
    """

    # Columns that strongly suggest a holdings table row
    HOLDINGS_KEYWORDS = [
        "שווי", "market value", "שם נייר", "security", "isin",
        "אחזקות", "holdings", "balance", "יתרה", "סכום",
        "שווי שוק", "שווי שוק בש\"ח", "שווי נוכחי",
    ]

    def parse(self, source: Union[Path, bytes, io.BytesIO]) -> ParseResult:
        buf = self._to_bytesio(source)
        warnings: list[str] = []
        errors: list[str] = []

        # Strategy 1: pdfplumber
        try:
            result = self._parse_pdfplumber(buf)
            if result.success:
                logger.info("PDFParser: pdfplumber succeeded")
                return result
            warnings.extend(result.warnings)
        except Exception as exc:
            warnings.append(f"pdfplumber failed: {exc}")

        # Strategy 2: camelot
        buf.seek(0)
        try:
            result = self._parse_camelot(buf)
            if result.success:
                logger.info("PDFParser: camelot succeeded")
                return result
            warnings.extend(result.warnings)
        except Exception as exc:
            warnings.append(f"camelot failed: {exc}")

        # Strategy 3: pymupdf text extraction
        buf.seek(0)
        try:
            result = self._parse_pymupdf(buf)
            if result.success:
                logger.info("PDFParser: pymupdf text-extraction succeeded")
                result.warnings.extend(warnings)
                return result
            warnings.extend(result.warnings)
        except Exception as exc:
            warnings.append(f"pymupdf failed: {exc}")

        errors.append(
            "All PDF extraction strategies failed.  "
            "Please check that the file is not scanned/image-only, "
            "or try uploading as CSV/XLSX."
        )
        return ParseResult(
            primary_df=pd.DataFrame(),
            warnings=warnings,
            errors=errors,
            parse_method="all_failed",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy implementations
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_pdfplumber(self, buf: io.BytesIO) -> ParseResult:
        import pdfplumber  # local import keeps startup fast

        buf.seek(0)
        all_tables: list[pd.DataFrame] = []
        warnings: list[str] = []

        with pdfplumber.open(buf) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 3,
                    }
                )
                if not tables:
                    # Fallback to text extraction on this page
                    tables = page.extract_tables()

                for raw_table in tables:
                    if not raw_table or len(raw_table) < 2:
                        continue
                    df = pd.DataFrame(raw_table[1:], columns=raw_table[0])
                    df = self._clean_header(df)
                    if self._looks_like_holdings(df):
                        all_tables.append(df)

        if not all_tables:
            return ParseResult(
                pd.DataFrame(),
                warnings=["pdfplumber: no holdings-like tables detected"],
                parse_method="pdfplumber",
            )

        best = self._select_best_table(all_tables)
        return ParseResult(
            primary_df=best,
            raw_tables=all_tables,
            parse_method="pdfplumber",
            warnings=warnings,
        )

    def _parse_camelot(self, buf: io.BytesIO) -> ParseResult:
        import camelot  # optional – may not be installed

        buf.seek(0)
        # camelot needs a file path, so we write to a temp file
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name

        try:
            tables = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")
            if not tables or tables.n == 0:
                tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream")
        finally:
            os.unlink(tmp_path)

        all_dfs: list[pd.DataFrame] = []
        for t in tables:
            df = t.df
            if df.shape[0] < 2:
                continue
            # First row as header
            df.columns = [str(c).strip() for c in df.iloc[0]]
            df = df.iloc[1:].reset_index(drop=True)
            df = self._clean_header(df)
            if self._looks_like_holdings(df):
                all_dfs.append(df)

        if not all_dfs:
            return ParseResult(
                pd.DataFrame(),
                warnings=["camelot: no holdings-like tables detected"],
                parse_method="camelot",
            )

        best = self._select_best_table(all_dfs)
        return ParseResult(
            primary_df=best,
            raw_tables=all_dfs,
            parse_method="camelot",
        )

    def _parse_pymupdf(self, buf: io.BytesIO) -> ParseResult:
        import fitz  # pymupdf

        buf.seek(0)
        doc = fitz.open(stream=buf.read(), filetype="pdf")
        rows: list[dict] = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    text = " ".join(span["text"] for span in line["spans"]).strip()
                    if text:
                        rows.append({"raw_text": text})

        if not rows:
            return ParseResult(
                pd.DataFrame(),
                warnings=["pymupdf: no text blocks found"],
                parse_method="pymupdf_text",
            )

        df = pd.DataFrame(rows)
        # Attempt to parse columnar lines as CSV-like
        # This is a best-effort fallback; user should review
        return ParseResult(
            primary_df=df,
            raw_tables=[df],
            parse_method="pymupdf_text",
            warnings=[
                "pymupdf text extraction used – table structure may not be preserved. "
                "Please review and correct the holdings in the confirmation step."
            ],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _looks_like_holdings(self, df: pd.DataFrame) -> bool:
        """Heuristic: does this DataFrame look like a holdings table?"""
        if df.empty or df.shape[1] < 2:
            return False
        col_text = " ".join(str(c) for c in df.columns).lower()
        return any(kw.lower() in col_text for kw in self.HOLDINGS_KEYWORDS)

    def _select_best_table(self, tables: list[pd.DataFrame]) -> pd.DataFrame:
        """Return the largest (most rows) plausible holdings table."""
        return max(tables, key=lambda t: len(t))
