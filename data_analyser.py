#!/usr/bin/env python3
"""
Unified Data Quality Analyzer & Intelligent Schema Evolution Assistant (CLI / Local)

- CSV-only ingestion (files, folder, or zip)
- Preprocessing: column standardization, missing symbol handling, type coercion
- Data Quality:
    * Completeness (missing)
    * Duplicates (exact + optional fuzzy)
    * Consistency (email/phone)
    * Validity (derived from consistency)
    * Uniqueness per column (candidate key detection)
    * Numeric profiling (min/max/mean/etc.)
    * Text profiling (length and distinct stats)
    * Aggregate Data Quality Score
- Schema change detection between versions (CLI multi-file mode)
- History persisted in SQLite (data_quality_history.db)
- Exports Power BI–ready CSVs to output_dir

Usage examples:
    python data_analyser.py --files data/jan.csv data/feb.csv --output_dir ./dq_output
    python data_analyser.py --folder ./datasets_history --output_dir ./dq_output
    python data_analyser.py --zip datasets.zip --output_dir ./dq_output
"""

from __future__ import annotations
import os
import sys
import json
import zipfile
import argparse
import tempfile
import shutil
import sqlite3
from typing import List, Iterable, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype  # Import added for the fix

HISTORY_DB_PATH = "data_quality_history.db"
PERF_THRESHOLD_ROWS = 100  # auto performance mode when rows > 100


# ---------------------- Utilities ----------------------


def now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec='seconds')


def to_snake_case(name: str) -> str:
    import re
    name = re.sub(r'[^a-zA-Z0-9_ \-]', '', str(name)).strip()
    name = re.sub(r'[\s\-]+', '_', name)
    return name.lower()


def safe_filename(s: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', s)


def detect_file_type_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return 'csv'
    return 'unknown'


def init_history_db():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT,
            file_source TEXT,
            rows INTEGER,
            cols INTEGER,
            missing_score REAL,
            duplicate_percent REAL,
            health_score REAL,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# Minimal in-memory DB for audit logs
class MockDB:
    def __init__(self):
        self.data = defaultdict(dict)

    def save_audit(self, user_id: str, entry: dict):
        if 'audit_log' not in self.data[user_id]:
            self.data[user_id]['audit_log'] = []
        self.data[user_id]['audit_log'].append(entry)

    def save_report(self, user_id: str, report: dict):
        self.data[user_id]['latest_report'] = report


def initialize_firestore():
    print("[info] Mock DB initialized")
    return MockDB()


EMAIL_REGEX = __import__('re').compile(r'^[^@]+@[^@]+\.[^@]+$')
PHONE_DIGIT_RE = __import__('re').compile(r'\d{7,15}')


class UnifiedDataQualityAnalyzer:
    # Insert or replace the prepare_data method inside the UnifiedDataQualityAnalyzer class in data_analyser.py
    # Insert or replace the prepare_data method inside the UnifiedDataQualityAnalyzer class in data_analyser.py

    def prepare_data(self, version: str):
        """
        Performs preliminary cleaning steps, now with robust duplicate detection.
        - Identifies and removes exact duplicate rows based on non-ID columns.
        - Stores the count and indices of removed duplicates in the quality report.
        """
        if version not in self.data_versions:
            print(f"[error] Version '{version}' not found for preparation.")
            return

        df = self.data_versions[version].copy()
        total_rows_original = len(df)

        # --- ROBUST DUPLICATE DETECTION LOGIC ---
        id_columns_to_ignore_for_dup_check = ['CustomerID', 'CustomerIdentifier']

        # 1. Create a temporary DataFrame for a robust duplicate check
        cols_to_check = [col for col in df.columns if col not in id_columns_to_ignore_for_dup_check]
        df_temp = df[cols_to_check].copy()

        # 2. Convert all checking columns to stripped strings to normalize data for a clean match
        for col in df_temp.columns:
            try:
                # Convert to string, replace NaN/None with a placeholder, and strip whitespace
                df_temp[col] = df_temp[col].astype(str).str.strip().replace({'nan': '', 'None': ''})
            except Exception as e:
                # Fallback conversion
                df_temp[col] = df_temp[col].apply(lambda x: str(x).strip())

        # 3. Identify all duplicate rows in the normalized temporary data
        duplicate_mask = df_temp.duplicated(keep=False)
        duplicate_indices = df_temp[duplicate_mask].index.tolist()

        # 4. Identify the rows to be removed (all duplicates except the first occurrence)
        rows_to_be_removed_mask = df_temp.duplicated(keep='first')
        removed_duplicate_rows_count = rows_to_be_removed_mask.sum()

        # --- END ROBUST DUPLICATE DETECTION LOGIC ---

        # Calculate percentage for display
        dup_percent = round((removed_duplicate_rows_count / total_rows_original) * 100,
                            2) if total_rows_original > 0 else 0

        # 5. Store ALL duplicate metrics in the report structure (Crucial for Flet UI)
        report = self.quality_reports.setdefault(version, {})
        dup_report = report.setdefault('duplicates_exact', {})

        dup_report['duplicate_row_indices'] = duplicate_indices
        dup_report['duplicate_rows_count'] = removed_duplicate_rows_count
        dup_report['duplicate_rows_percent'] = dup_percent

        # 6. Remove one of the functionally duplicate columns
        if 'CustomerIdentifier' in df.columns:
            df = df.drop(columns=['CustomerIdentifier'])

        # 7. Store the final deduplicated data (using the mask from the robust check)
        self.data_versions[version] = df[~rows_to_be_removed_mask]

        print(
            f"[info] Data version '{version}' prepared. Found and removed {removed_duplicate_rows_count} duplicate rows.")

    def prepare_data(self, version: str):
        """
        Performs preliminary cleaning steps based on business rules.
        - Identifies and removes exact duplicate rows based on non-ID columns.
        - Drops one of the functionally duplicate columns ('CustomerIdentifier').
        - Stores the count and indices of removed duplicates in the quality report.
        """
        if version not in self.data_versions:
            print(f"[error] Version '{version}' not found for preparation.")
            return

        df = self.data_versions[version].copy()
        total_rows_original = len(df)  # Get total rows BEFORE removal

        # NOTE: Columns to ignore when checking for duplicate ROWS (e.g., unique IDs)
        id_columns_to_ignore_for_dup_check = ['CustomerID', 'CustomerIdentifier']

        # Create a DataFrame copy without ID columns for duplicate check
        df_for_dup_check = df.drop(
            columns=[col for col in id_columns_to_ignore_for_dup_check if col in df.columns],
            errors='ignore'
        )

        # 1. Identify all duplicate rows (keep=False marks ALL involved rows as True)
        duplicate_mask = df_for_dup_check.duplicated(keep=False)
        duplicate_indices = df_for_dup_check[duplicate_mask].index.tolist()

        # 2. Calculate the count of rows that are duplicates (i.e., every duplicate row MINUS the first instance kept)
        rows_to_be_removed_mask = df_for_dup_check.duplicated(keep='first')
        removed_duplicate_rows_count = rows_to_be_removed_mask.sum()

        # Calculate percentage for display
        if total_rows_original > 0:
            dup_percent = round((removed_duplicate_rows_count / total_rows_original) * 100, 2)
        else:
            dup_percent = 0

        # 3. Store ALL duplicate metrics in the report structure
        report = self.quality_reports.setdefault(version, {})
        dup_report = report.setdefault('duplicates_exact', {})

        dup_report['duplicate_row_indices'] = duplicate_indices  # Indices of ALL rows involved in a duplication
        dup_report['duplicate_rows_count'] = removed_duplicate_rows_count  # Count of extra rows removed
        dup_report['duplicate_rows_percent'] = dup_percent

        # 4. Remove one of the functionally duplicate columns
        if 'CustomerIdentifier' in df.columns:
            df = df.drop(columns=['CustomerIdentifier'])

        # 5. Store the final deduplicated data
        self.data_versions[version] = df[~rows_to_be_removed_mask]

        print(f"[info] Data version '{version}' prepared. Removed {removed_duplicate_rows_count} duplicate rows.")

    # Insert this method inside the UnifiedDataQualityAnalyzer class in data_analyser.py

    def __init__(self, user_id: str = 'local_user'):
        self.db = initialize_firestore()
        self.user_id = user_id
        self.data_versions: dict[str, pd.DataFrame] = {}
        self.quality_reports: dict[str, dict] = {}
        self.schema_reports: list[dict] = []
        self.transform_audit_log: list[dict] = []
        self.default_missing_symbols = [
            'NA', 'N/A', 'null', '-', '?',
            'Unknown', 'none', 'None', 'nan', ''
        ]
        self._last_ingested_order: list[str] = []

        init_history_db()
        self.version_sources: dict[str, str] = {}  # version -> original file path / name

    # ----------------- Internal helpers -----------------

    def _log_audit(self, action: str, details: dict):
        entry = {'timestamp': now_iso(), 'action': action, 'details': details}
        self.transform_audit_log.append(entry)
        self.db.save_audit(self.user_id, entry)

    def _save_history_entry(self, version: str):
        """Persist per-version results to SQLite history."""
        df = self.data_versions.get(version)
        if df is None:
            return

        stats_missing = self.quality_reports.get(version, {}).get('missing', {})
        stats_dup = self.quality_reports.get(version, {}).get('duplicates_exact', {})
        health = self.quality_reports.get(version, {}).get('health_score', None)
        src = self.version_sources.get(version, "")

        rows = int(stats_missing.get('total_rows', len(df))) if stats_missing else len(df)
        cols = int(stats_missing.get('total_columns', len(df.columns))) if stats_missing else len(df.columns)
        missing_score = stats_missing.get('overall_missing_score', None)
        dup_pct = stats_dup.get('duplicate_rows_percent', None)

        conn = sqlite3.connect(HISTORY_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO analysis_history
            (version, file_source, rows, cols, missing_score, duplicate_percent, health_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version,
                src,
                rows,
                cols,
                float(missing_score) if missing_score is not None else None,
                float(dup_pct) if dup_pct is not None else None,
                float(health) if health is not None else None,
                now_iso(),
            ),
        )
        conn.commit()
        conn.close()

    # ----------------- Ingestion -----------------

    def _read_single_input(self, inp, given_name: Optional[str] = None):
        """
        CSV-only reader:
        - Accepts file paths with .csv
        - Accepts in-memory DataFrames
        - Optionally accepts JSON-like dict/list
        """

        if isinstance(inp, pd.DataFrame):
            name = given_name or f"df_{len(self.data_versions) + 1}"
            return inp.copy(), name

        if isinstance(inp, str) and os.path.exists(inp):
            ftype = detect_file_type_from_path(inp)
            base_name = os.path.splitext(os.path.basename(inp))[0]
            name = given_name or base_name
            if ftype != 'csv':
                print(f"[ERROR] Only .csv files are supported. Skipping: {inp}")
                return None, name
            try:
                df = pd.read_csv(inp)
                return df, name
            except Exception as e:
                print(f"[ERROR] Failed to read CSV file '{inp}': {e}")
                return None, name

        if isinstance(inp, (dict, list)):
            try:
                df = pd.DataFrame(inp)
                name = given_name or f"in_memory_{len(self.data_versions) + 1}"
                return df, name
            except Exception as e:
                print(f"[ERROR] cannot convert object to DataFrame: {e}")
                return None, given_name or f"input_{len(self.data_versions) + 1}"

        print(f"[WARN] unsupported input type or non-existing path: {repr(inp)}")
        return None, given_name or f"input_{len(self.data_versions) + 1}"

    def _make_unique_version_name(self, name: str) -> str:
        base = to_snake_case(name)
        if base not in self.data_versions:
            return base
        i = 1
        while f"{base}_{i}" in self.data_versions:
            i += 1
        return f"{base}_{i}"

    # ----------------- Preprocessing -----------------

    def _preprocess_dataframe(self, df: pd.DataFrame, missing_symbols: List[str]) -> pd.DataFrame:
        df = df.copy()

        # rename cols
        rename_map = {col: to_snake_case(col) for col in df.columns}
        df.rename(columns=rename_map, inplace=True)
        if rename_map:
            self._log_audit(
                'standardize_columns',
                {'renamed': {k: v for k, v in rename_map.items() if k != v}},
            )

        # normalize missing tokens (using map instead of deprecated applymap)
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            df[col] = df[col].map(
                lambda x: np.nan
                if isinstance(x, str)
                   and x.strip().lower() in {t.strip().lower() for t in missing_symbols}
                else (x.strip() if isinstance(x, str) else x)
            )

        self._log_audit('missing_symbol_normalization', {'tokens': missing_symbols})

        # date parsing
        for col in df.columns:
            if any(k in col.lower() for k in ('date', 'dt', 'time', 'timestamp')) and df[col].dtype == object:
                try:
                    # dayfirst=False is important for consistency (assumes M/D/Y by default)
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                    self._log_audit('date_parsing', {'column': col})
                except Exception:
                    pass

        # coerce numeric-like
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(200)
                if len(sample) > 0:
                    numeric_like_count = sum(
                        1
                        for v in sample
                        if __import__('re').fullmatch(
                            r'[-+]?\d+(\.\d+)?', v.replace(',', '')
                        )
                    )
                    if numeric_like_count / len(sample) > 0.8:
                        df[col] = df[col].astype(str).str.replace(',', '').replace('', np.nan)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        self._log_audit('coerce_numeric', {'column': col})

        return df

    # ----------------- MODULE 1: Data Quality -----------------

    def analyze_missing_data(self, version: str):
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version]
        total_rows = len(df)
        missing = df.isnull().sum().to_dict()
        summary = []
        for col, cnt in missing.items():
            if cnt > 0:
                summary.append(
                    {
                        'column': col,
                        'missing_count': int(cnt),
                        'missing_percent': round(100 * cnt / total_rows, 2)
                        if total_rows > 0
                        else None,
                    }
                )
        total_cells = total_rows * len(df.columns) if len(df.columns) > 0 else 0
        overall_missing_score = (
            100.0
            if total_cells == 0
            else round(100 - (df.isnull().sum().sum() / total_cells * 100), 2)
        )
        result = {
            'total_rows': total_rows,
            'total_columns': len(df.columns),
            'missing_summary': summary,
            'overall_missing_score': overall_missing_score,
        }
        self.quality_reports.setdefault(version, {})['missing'] = result
        self._log_audit(
            'missing_analysis',
            {'version': version, 'total_missing_cells': int(df.isnull().sum().sum())},
        )
        return result

    def analyze_duplicates_exact(self, version: str):
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")

        # --- FIX STARTS HERE ---
        # 1. Create a copy to modify without changing the source DataFrame
        df = self.data_versions[version].copy()

        # 2. Convert any datetime columns back to a simple string format for robust comparison
        # This solves the issue where microsecond differences in datetime objects caused mismatches.
        for col in df.columns:
            if is_datetime64_any_dtype(df[col]):
                # Convert to YYYY-MM-DD string to ensure identical representation
                df[col] = df[col].dt.strftime('%Y-%m-%d')
        # --- FIX ENDS HERE ---

        if len(df) == 0:
            out = {'duplicate_rows_count': 0, 'duplicate_rows_percent': 0.0, 'subset_used': 'ALL'}
        else:
            # Check for ID columns to ignore them in duplication analysis
            all_cols = df.columns.tolist()
            # Identify columns to ignore: snake_cased names containing 'id', 'key', or 'uuid'
            id_cols = [c for c in all_cols if any(k in c.lower() for k in ('id', 'key', 'uuid', 'identifier'))]

            # Determine which columns to check for duplication
            subset_cols = all_cols
            if id_cols and len(id_cols) < len(all_cols):
                # If there are ID columns and they don't cover all columns, exclude them
                subset_cols = [c for c in all_cols if c not in id_cols]
                subset_used = ', '.join(subset_cols)
            else:
                subset_used = 'ALL'

            # Use only the relevant subset of columns for the duplication check
            df_to_check = df if subset_used == 'ALL' else df[subset_cols]

            # Perform duplicate check
            # Keep=False marks ALL instances of a duplicate set (the first AND subsequent ones)
            dup_mask = df_to_check.duplicated(keep=False)
            dup_count = int(dup_mask.sum())

            out = {
                'duplicate_rows_count': dup_count,
                'duplicate_rows_percent': round(dup_count / len(df) * 100, 2),
                'subset_used': subset_used
            }

        self.quality_reports.setdefault(version, {})['duplicates_exact'] = out
        self._log_audit('duplicate_exact_analysis', {'version': version, **out})
        return out

    def analyze_duplicate_columns(self, version: str):
        """Detects duplicate column names and columns with identical values."""
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version]

        # 1. Duplicate Column Names
        col_names = df.columns.tolist()
        seen = set()
        duplicate_name_columns = []
        for name in col_names:
            if name in seen and name not in duplicate_name_columns:
                duplicate_name_columns.append(name)
            seen.add(name)

        # 2. Duplicate Column Values (Exact content duplicates)
        duplicate_value_columns = []

        # Use a string cat signature comparison for efficiency on large DFs.
        col_list = list(df.columns)
        for i in range(len(col_list)):
            col1_name = col_list[i]
            # Use string representation of the column for comparison
            col1_signature = df[col1_name].astype(str).str.cat(sep='|')

            for j in range(i + 1, len(col_list)):
                col2_name = col_list[j]
                col2_signature = df[col2_name].astype(str).str.cat(sep='|')

                if col1_signature == col2_signature:
                    # Store as a tuple of names
                    duplicate_value_columns.append((col1_name, col2_name))

        result = {
            'duplicate_name_columns': duplicate_name_columns,
            'duplicate_value_columns': duplicate_value_columns
        }

        self.quality_reports.setdefault(version, {})['duplicate_columns'] = result
        self._log_audit(
            'duplicate_columns_analysis',
            {'version': version, 'name_dups': len(duplicate_name_columns), 'value_dups': len(duplicate_value_columns)},
        )
        return result

    def analyze_duplicates_fuzzy(
            self,
            version: str,
            cols: Optional[List[str]] = None,
            threshold: float = 0.85,
    ):
        import difflib

        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version].copy()
        n = len(df)
        if n == 0:
            return {'fuzzy_pairs': [], 'fuzzy_pairs_count': 0}

        if cols:
            use_cols = [c for c in cols if c in df.columns]
            if not use_cols:
                raise ValueError("No matching columns")
        else:
            use_cols = df.select_dtypes(include=['object']).columns.tolist()
            if not use_cols:
                use_cols = list(df.columns[: min(2, len(df.columns))])

        def norm_row_signature(row):
            parts = []
            for c in use_cols:
                val = row.get(c)
                if pd.isna(val):
                    parts.append('')
                else:
                    s = str(val).lower().strip()
                    s = __import__('re').sub(r'[^a-z0-9]', '', s)
                    parts.append(s)
            return '_'.join(parts)

        df['_signature'] = df.apply(lambda r: norm_row_signature(r), axis=1)
        sigs = df['_signature'].tolist()
        buckets = defaultdict(list)
        for i, s in enumerate(sigs):
            key = s[:6] if len(s) >= 6 else s
            buckets[key].append((i, s))
        fuzzy_pairs = []
        comparisons = 0
        max_pairs = 200000

        for bucket_items in buckets.values():
            L = len(bucket_items)
            if L <= 1:
                continue
            for i in range(L):
                for j in range(i + 1, L):
                    comparisons += 1
                    if comparisons > max_pairs:
                        break
                    idx_i, sig_i = bucket_items[i]
                    idx_j, sig_j = bucket_items[j]
                    if not sig_i or not sig_j:
                        continue
                    ratio = difflib.SequenceMatcher(a=sig_i, b=sig_j).ratio()
                    if ratio >= threshold:
                        fuzzy_pairs.append(
                            {
                                'idx_left': int(idx_i),
                                'idx_right': int(idx_j),
                                'similarity': round(ratio, 3),
                            }
                        )
                if comparisons > max_pairs:
                    break
            if comparisons > max_pairs:
                break

        result = {'fuzzy_pairs': fuzzy_pairs, 'fuzzy_pairs_count': len(fuzzy_pairs)}
        self.quality_reports.setdefault(version, {})['duplicates_fuzzy'] = {
            'count': len(fuzzy_pairs)
        }
        self._log_audit(
            'duplicate_fuzzy_analysis',
            {'version': version, 'pairs_found': len(fuzzy_pairs)},
        )
        return result

    def analyze_consistency(self, version: str):
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version]
        issues = []
        for col in df.columns:
            if 'email' in col.lower():
                col_series = df[col].astype(str).fillna('')
                invalid_mask = ~col_series.apply(
                    lambda x: bool(EMAIL_REGEX.match(x)) if x else True
                )
                cnt_invalid = int(invalid_mask.sum())
                if cnt_invalid > 0:
                    issues.append(
                        {
                            'column': col,
                            'check': 'email_format',
                            'invalid_count': cnt_invalid,
                        }
                    )
            if 'phone' in col.lower() or 'mobile' in col.lower():
                col_series = df[col].astype(str).fillna('')
                invalid_mask = ~col_series.apply(
                    lambda x: bool(PHONE_DIGIT_RE.search(x)) if x else True
                )
                cnt_invalid = int(invalid_mask.sum())
                if cnt_invalid > 0:
                    issues.append(
                        {
                            'column': col,
                            'check': 'phone_format',
                            'invalid_count': cnt_invalid,
                        }
                    )
        # Store consistency report as a dictionary with an 'issues' key for clean access
        self.quality_reports.setdefault(version, {})['consistency'] = {'issues': issues}
        self._log_audit(
            'consistency_checks',
            {'version': version, 'issues_found': len(issues)},
        )
        return {'issues': issues}

    # ---------- NEW: Validity, Uniqueness, Profiling ----------

    def evaluate_validity(self, version: str):
        """
        Derive a 'validity' score per column from consistency issues.
        If we have no specific rules for a column, we assume validity = 100.
        """
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version]
        total_rows = len(df) if len(df) > 0 else 1

        # Use the internal report stored by analyze_consistency
        cons_result = self.quality_reports.get(version, {}).get('consistency', {})
        issues = cons_result.get('issues', [])

        # Map: column -> total invalid
        invalid_by_col = defaultdict(int)
        for issue in issues:
            c = issue.get('column')
            cnt = int(issue.get('invalid_count', 0))
            invalid_by_col[c] += cnt

        validity_rows = []
        for col in df.columns:
            invalid = invalid_by_col.get(col, 0)
            valid = max(0, total_rows - invalid)
            validity_pct = 100.0 if total_rows == 0 else round(valid / total_rows * 100, 2)
            validity_rows.append(
                {
                    'column': col,
                    'invalid_count': invalid,
                    'valid_count': valid,
                    'validity_percent': validity_pct,
                }
            )

        # Aggregate validity score = average over columns
        if validity_rows:
            avg_validity = round(
                sum(r['validity_percent'] for r in validity_rows) / len(validity_rows),
                2,
            )
        else:
            avg_validity = 100.0

        validity_report = {
            'per_column': validity_rows,
            'avg_validity_percent': avg_validity,
        }
        self.quality_reports.setdefault(version, {})['validity'] = validity_report
        self._log_audit(
            'validity_evaluation',
            {'version': version, 'avg_validity_percent': avg_validity},
        )
        return validity_report

    def evaluate_uniqueness(self, version: str):
        """
        For each column, compute distinct count + uniqueness ratio.
        Also mark potential candidate keys.
        """
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version]
        total_rows = len(df) if len(df) > 0 else 1

        rows = []
        candidate_keys = []

        for col in df.columns:
            non_null = df[col].notna().sum()
            distinct_vals = df[col].nunique(dropna=True)
            uniqueness_ratio = 0.0 if total_rows == 0 else round(distinct_vals / total_rows * 100, 2)

            is_candidate_key = (
                    total_rows > 0
                    and distinct_vals == non_null
                    and uniqueness_ratio >= 95.0
            )

            if is_candidate_key:
                candidate_keys.append(col)

            rows.append(
                {
                    'column': col,
                    'non_null_count': int(non_null),
                    'distinct_count': int(distinct_vals),
                    'uniqueness_percent': uniqueness_ratio,
                    'is_candidate_key': bool(is_candidate_key),
                }
            )

        # aggregate uniqueness = max uniqueness (best ID) or avg if nothing stands out
        if rows:
            best = max(r['uniqueness_percent'] for r in rows)
            avg = sum(r['uniqueness_percent'] for r in rows) / len(rows)
            agg_uniqueness = round(best if best >= 90 else avg, 2)
        else:
            agg_uniqueness = 100.0

        report = {
            'per_column': rows,
            'agg_uniqueness_percent': agg_uniqueness,
            'candidate_keys': candidate_keys,
        }
        self.quality_reports.setdefault(version, {})['uniqueness'] = report
        self._log_audit(
            'uniqueness_evaluation',
            {'version': version, 'agg_uniqueness_percent': agg_uniqueness, 'candidate_keys': candidate_keys},
        )
        return report

    def profile_numeric(self, version: str):
        """
        Numeric data profiling: min, max, mean, median, std, nulls, zeros, negatives, positives.
        """
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")

        df = self.data_versions[version]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        rows = []
        for col in numeric_cols:
            series = df[col]
            non_null = series.dropna()
            if non_null.empty:
                continue
            total = len(series)
            zeros = int((series == 0).sum())
            negatives = int((series < 0).sum())
            positives = int((series > 0).sum())
            nulls = int(series.isnull().sum())
            rows.append(
                {
                    'column': col,
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()) if len(non_null) > 1 else 0.0,
                    'null_count': nulls,
                    'zero_count': zeros,
                    'negative_count': negatives,
                    'positive_count': positives,
                    'zero_percent': round(zeros / total * 100, 2) if total > 0 else 0.0,
                    'null_percent': round(nulls / total * 100, 2) if total > 0 else 0.0,
                }
            )

        self.quality_reports.setdefault(version, {})['numeric_profile'] = rows
        self._log_audit(
            'numeric_profile',
            {'version': version, 'columns_profiled': len(rows)},
        )
        return rows

    def profile_text(self, version: str):
        """
        Text data profiling: average length, nulls, blanks, distinct, etc.
        """
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")

        df = self.data_versions[version]
        obj_cols = df.select_dtypes(include=['object']).columns.tolist()

        rows = []
        for col in obj_cols:
            series = df[col]
            lengths = series.dropna().astype(str).map(len)
            if lengths.empty:
                continue

            total = len(series)
            nulls = int(series.isnull().sum())
            blanks = int(series.astype(str).map(lambda x: x.strip() == '').sum())
            distinct_vals = series.nunique(dropna=True)

            rows.append(
                {
                    'column': col,
                    'avg_length': float(lengths.mean()),
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'null_count': nulls,
                    'blank_count': blanks,
                    'blank_percent': round(blanks / total * 100, 2) if total > 0 else 0.0,
                    'null_percent': round(nulls / total * 100, 2) if total > 0 else 0.0,
                    'distinct_count': int(distinct_vals),
                }
            )

        self.quality_reports.setdefault(version, {})['text_profile'] = rows
        self._log_audit(
            'text_profile',
            {'version': version, 'columns_profiled': len(rows)},
        )
        return rows

    # ----------------- MODULE 2: Schema change detection -----------------

    def detect_schema_changes(
            self,
            old_version: str,
            new_version: str,
            rename_similarity_threshold: float = 0.4,
    ):
        if old_version not in self.data_versions or new_version not in self.data_versions:
            raise KeyError("versions missing")
        df_old = self.data_versions[old_version]
        df_new = self.data_versions[new_version]
        old_cols = set(df_old.columns)
        new_cols = set(df_new.columns)
        added = sorted(list(new_cols - old_cols))
        removed = sorted(list(old_cols - new_cols))
        common = sorted(list(old_cols & new_cols))
        type_changes = []
        for c in common:
            old_dtype = str(df_old[c].dtype)
            new_dtype = str(df_new[c].dtype)
            if old_dtype != new_dtype:
                type_changes.append(
                    {'column': c, 'old_dtype': old_dtype, 'new_dtype': new_dtype}
                )
        old_rows = len(df_old)
        new_rows = len(df_new)
        row_change = new_rows - old_rows
        row_change_pct = round(100 * row_change / old_rows, 2) if old_rows > 0 else None

        # rename detection
        sample_limit = 200

        def sample_set(df, col):
            if col not in df.columns:
                return set()
            vals = (
                df[col]
                .dropna()
                .astype(str)
                .head(sample_limit)
                .str.lower()
                .astype(str)
                .tolist()
            )
            return set(vals)

        old_samples = {c: sample_set(df_old, c) for c in removed}
        new_samples = {c: sample_set(df_new, c) for c in added}
        import difflib

        rename_candidates = []
        for r in removed:
            best = None
            best_score = 0.0
            for a in added:
                s_old = old_samples.get(r, set())
                s_new = new_samples.get(a, set())
                overlap = len(s_old & s_new)
                denom = max(1, min(len(s_old), len(s_new)))
                overlap_score = overlap / denom if denom > 0 else 0.0
                name_sim = difflib.SequenceMatcher(None, r, a).ratio()
                combined = 0.6 * overlap_score + 0.4 * name_sim
                if combined > best_score:
                    best_score = combined
                    best = a
            if best and best_score >= rename_similarity_threshold:
                rename_candidates.append(
                    {'old_column': r, 'new_column': best, 'score': round(best_score, 3)}
                )

        schema_report = {
            'comparison_id': f"{old_version}_to_{new_version}",
            'old_version': old_version,
            'new_version': new_version,
            'columns_added': added,
            'columns_removed': removed,
            'type_changes': type_changes,
            'row_old': old_rows,
            'row_new': new_rows,
            'row_change': row_change,
            'row_change_percent': row_change_pct,
            'rename_suggestions': rename_candidates,
            'total_changes': len(added) + len(removed) + len(type_changes),
        }
        self.schema_reports.append(schema_report)
        self._log_audit(
            'schema_change_detection',
            {'comparison': schema_report['comparison_id'], 'total_changes': schema_report['total_changes']},
        )
        return schema_report

    # ----------------- MODULE 3: Transformations -----------------

    def drop_duplicates(
            self,
            version: str,
            subset: Optional[List[str]] = None,
            keep: str = 'first',
    ):
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version].copy()
        before = len(df)
        dropped_mask = df.duplicated(subset=subset, keep=keep)
        dropped = int(dropped_mask.sum())
        if dropped > 0:
            df = df.drop_duplicates(subset=subset, keep=keep)
            self.data_versions[version] = df
            self._log_audit(
                'transformation_drop_duplicates',
                {
                    'version': version,
                    'subset': subset or 'ALL',
                    'dropped': dropped,
                    'before': before,
                    'after': len(df),
                },
            )
        return df, dropped

    def impute_missing(
            self,
            version: str,
            column: str,
            strategy: str = 'mode',
            value: Optional[any] = None,
    ):
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version]
        if column not in df.columns:
            raise KeyError(f"{column} not in {version}")
        missing_before = int(df[column].isnull().sum())
        if missing_before == 0:
            return 0
        fill = None
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df[column]):
                fill = df[column].mean()
            else:
                raise TypeError("mean requires numeric")
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df[column]):
                fill = df[column].median()
            else:
                raise TypeError("median requires numeric")
        elif strategy == 'mode':
            m = df[column].mode()
            fill = m.iloc[0] if not m.empty else None
        elif strategy == 'constant':
            if value is None:
                raise ValueError("constant needs value")
            fill = value
        else:
            raise ValueError("unknown strategy")
        if fill is None:
            return 0
        df[column].fillna(fill, inplace=True)
        filled = missing_before - int(df[column].isnull().sum())
        self.data_versions[version] = df
        self._log_audit(
            'transformation_impute',
            {'version': version, 'column': column, 'strategy': strategy, 'filled': filled},
        )
        return filled

    # ----------------- ADDED: Computed Column and Health Score -----------------

    def add_computed_column(self, version: str, new_col: str, expression_fn):
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")
        df = self.data_versions[version]
        df[new_col] = expression_fn(df)
        self.data_versions[version] = df
        self._log_audit(
            'transformation_add_computed',
            {'version': version, 'new_column': new_col},
        )

    # ----------------- Health score & dimensions -----------------

    def compute_data_quality_dimensions(self, version: str):
        """
        Compute a richer breakdown of quality dimensions and aggregate score.
        """
        if version not in self.data_versions:
            raise KeyError(f"{version} not found")

        # Get existing reports (populated by run_full_scan)
        missing_report = self.quality_reports.get(version, {}).get('missing', {})
        dup_report = self.quality_reports.get(version, {}).get('duplicates_exact', {})
        validity_report = self.quality_reports.get(version, {}).get('validity', {})
        uniqueness_report = self.quality_reports.get(version, {}).get('uniqueness', {})
        cons_report = self.quality_reports.get(version, {}).get('consistency', {})

        # Calculate scores from reports
        completeness_score = missing_report.get('overall_missing_score', 100.0)
        duplicate_score = 100 - dup_report.get('duplicate_rows_percent', 0.0)
        validity_score = validity_report.get('avg_validity_percent', 100.0)
        uniqueness_score = uniqueness_report.get('agg_uniqueness_percent', 100.0)

        cons_issues = len(cons_report.get('issues', []))
        consistency_score = max(0.0, 100.0 - cons_issues * 5.0)

        # simple schema stability: for single-version session assume 100
        schema_stability_score = 100.0

        # Weighted overall quality
        overall_score = (
                0.40 * completeness_score +
                0.20 * validity_score +
                0.15 * duplicate_score +
                0.15 * uniqueness_score +
                0.10 * consistency_score
        )

        overall_score = round(max(0.0, min(100.0, overall_score)), 2)

        dims = {
            'completeness_score': round(completeness_score, 2),
            'duplicate_score': round(duplicate_score, 2),
            'validity_score': round(validity_score, 2),
            'uniqueness_score': round(uniqueness_score, 2),
            'consistency_score': round(consistency_score, 2),
            'schema_stability_score': round(schema_stability_score, 2),
            'overall_quality_score': overall_score,
        }

        self.quality_reports.setdefault(version, {})['dimensions'] = dims
        self.quality_reports.setdefault(version, {})['health_score'] = overall_score

        self._log_audit(
            'compute_quality_dimensions',
            {'version': version, **dims},
        )
        return dims

    def compute_data_health_score(self, version: str):
        """
        Kept for backward-compat with older callers; now delegates to compute_data_quality_dimensions.
        """
        dims = self.compute_data_quality_dimensions(version)
        return dims['overall_quality_score']

    # ----------------- Export helpers -----------------

    def _export_df(self, df: pd.DataFrame, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        self._log_audit('export_csv', {'path': path})

    def export_all_reports(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        exported = {}
        # cleaned datasets
        for v, df in self.data_versions.items():
            p = os.path.join(output_dir, f"{safe_filename(v)}_cleaned.csv")
            df.to_csv(p, index=False)
            exported[f"cleaned_{v}"] = p

        # quality summary
        qs_rows = []
        for v in self.data_versions.keys():
            miss = self.quality_reports.get(v, {}).get('missing', {})
            dup = self.quality_reports.get(v, {}).get('duplicates_exact', {})
            dims = self.quality_reports.get(v, {}).get('dimensions', {})
            health = self.quality_reports.get(v, {}).get('health_score', None)
            qs_rows.append(
                {
                    'version': v,
                    'total_rows': miss.get('total_rows', len(self.data_versions[v])),
                    'total_columns': miss.get(
                        'total_columns', len(self.data_versions[v].columns)
                    ),
                    'overall_missing_score': miss.get('overall_missing_score', None),
                    'duplicate_rows_percent': dup.get('duplicate_rows_percent', None),
                    'data_health_score': health,
                    'completeness_score': dims.get('completeness_score'),
                    'validity_score': dims.get('validity_score'),
                    'uniqueness_score': dims.get('uniqueness_score'),
                    'consistency_score': dims.get('consistency_score'),
                }
            )
        df_qs = pd.DataFrame(qs_rows)
        p = os.path.join(output_dir, "quality_summary.csv")
        df_qs.to_csv(p, index=False)
        exported['quality_summary'] = p

        # per-version details
        for v in self.data_versions.keys():
            miss = self.quality_reports.get(v, {}).get('missing', {})
            if miss and miss.get('missing_summary'):
                p = os.path.join(output_dir, f"missing_{safe_filename(v)}.csv")
                pd.DataFrame(miss['missing_summary']).to_csv(p, index=False)
                exported[f"missing_{v}"] = p

            fuzzy = self.quality_reports.get(v, {}).get('duplicates_fuzzy', {})
            if fuzzy:
                p = os.path.join(
                    output_dir, f"duplicates_fuzzy_{safe_filename(v)}.json"
                )
                with open(p, 'w', encoding='utf-8') as fh:
                    json.dump(fuzzy, fh, indent=2)
                exported[f"duplicates_fuzzy_{v}"] = p

            cons_report = self.quality_reports.get(v, {}).get('consistency', {})
            cons_issues = cons_report.get('issues', [])
            if cons_issues:
                p = os.path.join(output_dir, f"consistency_{safe_filename(v)}.csv")
                pd.DataFrame(cons_issues).to_csv(p, index=False)
                exported[f"consistency_{v}"] = p

            validity = self.quality_reports.get(v, {}).get('validity', {})
            if validity and validity.get('per_column'):
                p = os.path.join(output_dir, f"validity_{safe_filename(v)}.csv")
                pd.DataFrame(validity['per_column']).to_csv(p, index=False)
                exported[f"validity_{v}"] = p

            uniq = self.quality_reports.get(v, {}).get('uniqueness', {})
            if uniq and uniq.get('per_column'):
                p = os.path.join(output_dir, f"uniqueness_{safe_filename(v)}.csv")
                pd.DataFrame(uniq['per_column']).to_csv(p, index=False)
                exported[f"uniqueness_{v}"] = p

            num_prof = self.quality_reports.get(v, {}).get('numeric_profile', [])
            if num_prof:
                p = os.path.join(output_dir, f"numeric_profile_{safe_filename(v)}.csv")
                pd.DataFrame(num_prof).to_csv(p, index=False)
                exported[f"numeric_profile_{v}"] = p

            text_prof = self.quality_reports.get(v, {}).get('text_profile', [])
            if text_prof:
                p = os.path.join(output_dir, f"text_profile_{safe_filename(v)}.csv")
                pd.DataFrame(text_prof).to_csv(p, index=False)
                exported[f"text_profile_{v}"] = p

            # Duplicate Columns Report
            dup_cols = self.quality_reports.get(v, {}).get('duplicate_columns', {})
            if dup_cols and (dup_cols.get('duplicate_name_columns') or dup_cols.get('duplicate_value_columns')):
                # Create a simple summary DataFrame for easy export/view
                dup_names = [('Duplicate Column Name', name) for name in dup_cols.get('duplicate_name_columns', [])]
                dup_values = [('Duplicate Column Values', f"{n1} vs {n2}") for n1, n2 in
                              dup_cols.get('duplicate_value_columns', [])]
                dup_df = pd.DataFrame(dup_names + dup_values, columns=['Duplicate Type', 'Column(s)'])
                if not dup_df.empty:
                    p = os.path.join(output_dir, f"duplicate_columns_{safe_filename(v)}.csv")
                    dup_df.to_csv(p, index=False)
                    exported[f"duplicate_columns_{v}"] = p

        # schema diffs
        if self.schema_reports:
            p = os.path.join(output_dir, "schema_diffs.json")
            with open(p, 'w', encoding='utf-8') as fh:
                json.dump(self.schema_reports, fh, indent=2)
            exported['schema_diffs'] = p

        # transform log
        if self.transform_audit_log:
            p = os.path.join(output_dir, "transformation_log.csv")
            pd.DataFrame(self.transform_audit_log).to_csv(p, index=False)
            exported['transformation_log'] = p

        # health scores
        hs_rows = [
            {
                'version': v,
                'health_score': self.quality_reports.get(v, {}).get(
                    'health_score', None
                ),
            }
            for v in self.data_versions.keys()
        ]
        p = os.path.join(output_dir, "health_scores.csv")
        pd.DataFrame(hs_rows).to_csv(p, index=False)
        exported['health_scores'] = p

        self._log_audit(
            'export_all_reports',
            {'output_dir': output_dir, 'files': list(exported.values())},
        )
        return exported

    # ----------------- Full scan -----------------

    def run_full_scan(
            self,
            output_dir: Optional[str] = None,
            fuzzy_threshold: float = 0.85,
            performance_mode: bool = False,
    ):
        """
        Run:
        - Missing, duplicates, consistency, validity, uniqueness, numeric/text profiling
        - Quality dimensions + health score
        - Optional fuzzy duplicates (skipped in performance mode)
        """
        results = {'per_version': {}, 'schema_diffs': []}
        versions = (
            list(self._last_ingested_order)
            if self._last_ingested_order
            else list(self.data_versions.keys())
        )

        for v in versions:
            try:
                df = self.data_versions[v]
                is_big = len(df) > PERF_THRESHOLD_ROWS
                use_perf = performance_mode or is_big

                # Run analysis methods (they store results internally in self.quality_reports[v])
                self.analyze_missing_data(v)
                self.analyze_duplicates_exact(v)
                self.analyze_duplicate_columns(v)
                self.analyze_consistency(v)
                self.evaluate_validity(v)
                self.evaluate_uniqueness(v)

                # Retrieve reports
                miss = self.quality_reports.get(v, {}).get('missing', {})
                dup_e = self.quality_reports.get(v, {}).get('duplicates_exact', {})
                dup_c = self.quality_reports.get(v, {}).get('duplicate_columns', {})
                cons = self.quality_reports.get(v, {}).get('consistency', {})
                validity = self.quality_reports.get(v, {}).get('validity', {})
                uniqueness = self.quality_reports.get(v, {}).get('uniqueness', {})

                fuzzy = {}
                if not use_perf and len(df) <= 20000:
                    fuzzy = self.analyze_duplicates_fuzzy(v, threshold=fuzzy_threshold)

                num_prof = []
                text_prof = []
                if not use_perf:
                    num_prof = self.profile_numeric(v)
                    text_prof = self.profile_text(v)

                dims = self.compute_data_quality_dimensions(v)  # Calculates health score and stores dimensions

                health = dims['overall_quality_score']

                results['per_version'][v] = {
                    'missing': miss,
                    'duplicates_exact': dup_e,
                    'duplicate_columns': dup_c,
                    'duplicates_fuzzy': {'count': fuzzy.get('fuzzy_pairs_count', 0)},
                    'validity': validity,
                    'uniqueness': uniqueness,
                    'consistency': cons,
                    'numeric_profile': num_prof,
                    'text_profile': text_prof,
                    'dimensions': dims,
                    'health_score': health,
                    'performance_mode_used': use_perf,
                }

                # persist to SQLite history
                self._save_history_entry(v)

            except Exception as e:
                self._log_audit('scan_error', {'version': v, 'error': str(e)})
                results['per_version'][v] = {'error': str(e)}

        for i in range(1, len(versions)):
            oldv = versions[i - 1]
            newv = versions[i]
            diff = self.detect_schema_changes(oldv, newv)
            results['schema_diffs'].append(diff)

        if output_dir:
            exported = self.export_all_reports(output_dir)
            results['exported'] = exported

        self._log_audit('run_full_scan', {'versions': versions})
        return results


# ---------------------- Helper for CLI ----------------------


def collect_files_from_folder(folder: str) -> List[str]:
    supported = ('.csv',)
    file_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(supported):
                file_paths.append(os.path.join(root, f))
    return sorted(file_paths)


def extract_zip_to_temp(zip_path: str) -> str:
    tmp = tempfile.mkdtemp(prefix='dq_zip_')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp)
    return tmp


# ---------------------- CLI entrypoint ----------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified Data Quality Analyzer - CLI (CSV only)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--files', nargs='+',
                       help='List of CSV file paths to ingest')
    group.add_argument('--folder', help='Folder path containing CSV dataset versions')
    group.add_argument('--zip', help='Zip file containing CSV datasets')
    parser.add_argument('--file_names', nargs='+',
                        help='Optional friendly names for files (in same order)')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to write exported CSV/JSON outputs')
    parser.add_argument('--fuzzy_threshold', type=float, default=0.85,
                        help='Fuzzy duplicate similarity threshold (0-1)')
    parser.add_argument('--no_fuzzy_for_big', action='store_true',
                        help='(kept for backward compat; perf mode handled internally)')
    args = parser.parse_args(argv)

    inputs = []
    input_names = None
    tmp = None

    if args.files:
        inputs = args.files
        input_names = args.file_names if args.file_names else None
    elif args.folder:
        inputs = collect_files_from_folder(args.folder)
    elif args.zip:
        tmp = extract_zip_to_temp(args.zip)
        print(f"[info] extracted zip to temporary folder: {tmp}")
        inputs = collect_files_from_folder(tmp)

    if not inputs:
        print("[error] no supported files found")
        sys.exit(1)

    analyzer = UnifiedDataQualityAnalyzer(user_id='cli_user')
    print(f"[info] ingesting {len(inputs)} files...")
    ingested = analyzer.ingest_files(inputs, file_names=input_names,
                                     output_dir=args.output_dir)
    print(f"[info] ingested versions: {ingested}")

    print("[info] running full scan...")
    # Pass 'performance_mode=True' to run_full_scan if needed, but Flet controls this via the switch
    summary = analyzer.run_full_scan(output_dir=args.output_dir,
                                     fuzzy_threshold=args.fuzzy_threshold,
                                     performance_mode=False)

    print("[done] scan complete. Exported files:")
    for k, v in summary.get('exported', {}).items():
        print(f"  - {k}: {v}")

    if tmp is not None:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass


if __name__ == '__main__':
    main()
