#!/usr/bin/env python
# coding: utf-8
"""
portfolio/allocator.py — Minimal Allocator (Phase 3, Step 5).

Orchestrates a list of Book instances: runs each active book independently
and combines their PnL streams by simple addition.

This is the FIRST VERSION of the Allocator.  It contains NO regime logic.
Regime-aware allocation (book activation, alpha multipliers, weight tilts)
will be added in a subsequent step via the Allocator.

Public API
----------
    Allocator(books)
        books : list[Book]

    allocator.run(returns_df) -> dict
        Returns {"pnl": combined_pnl, "book_results": {name: result_dict}}

Design constraints (invariants)
--------------------------------
- Pure orchestration: no alpha modification, no position scaling, no weighting.
- Books are run independently; Allocator does not mutate Book state.
- Inactive books (book.is_active == False) are skipped silently.
- PnL combination uses pandas .add(fill_value=0.0) to handle index mismatches
  between books that may have different date coverage.
- With a single active book, combined_pnl is identical to book.run()["pnl"].
"""

import pandas as pd


class Allocator:
    """
    Minimal portfolio allocator.

    Runs each active Book independently and sums their PnL streams.
    No regime conditioning, no weight tilts, no alpha modification.

    Parameters
    ----------
    books : list[Book]
        List of Book instances to orchestrate.
    """

    def __init__(self, books):
        self.books = books

    def run(self, returns_df: pd.DataFrame) -> dict:
        """
        Run all active books and combine their PnL by simple addition.

        Parameters
        ----------
        returns_df : pd.DataFrame (T, N) — daily returns, same format as Book.run() expects.

        Returns
        -------
        dict with:
            "pnl"          : pd.Series — combined PnL (sum across active books)
            "book_results" : dict[str, dict] — {book.name: result} for each active book
        """
        book_results = {}

        # ── Step 1: run each active book independently ────────────────────────
        for book in self.books:
            if not book.is_active:
                continue
            res = book.run(returns_df)
            book_results[book.name] = res

        # ── Step 2: combine PnL by addition ───────────────────────────────────
        combined_pnl = None
        for res in book_results.values():
            pnl = res["pnl"]
            if combined_pnl is None:
                combined_pnl = pnl.copy()
            else:
                combined_pnl = combined_pnl.add(pnl, fill_value=0.0)

        # ── Step 3: return ────────────────────────────────────────────────────
        return {
            "pnl":          combined_pnl,
            "book_results": book_results,
        }
