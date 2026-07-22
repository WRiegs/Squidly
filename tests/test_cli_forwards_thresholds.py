"""
Regression tests for the bug where ``squidly run --mean-prob / --mean-var``
were silently ignored because the outer typer CLI in ``squidly/__main__.py``
did not forward those flags to the ``squidly/squidly.py`` worker subprocess.

The worker filters residues at its argparse defaults (``mean_prob=0.6``,
``mean_var=0.225``) whenever the flags are absent, so the bug produced empty
``Squidly_CR_Position`` outputs for anyone using non-default thresholds on
divergent enzyme families.

These tests use AST inspection rather than end-to-end execution because
``run`` does considerable setup (fasta parsing, BLAST, pickle loading) before
it reaches the subprocess call, and mocking all of that would be heavier and
more fragile than checking the source-level invariant directly.
"""
from __future__ import annotations

import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN_PATH = REPO_ROOT / "squidly" / "__main__.py"


def _cmd_list_literals(tree: ast.AST) -> list[ast.List]:
    """Return every list literal in the source that looks like a subprocess
    command targeting ``squidly.py`` (the ensemble worker).

    A cmd list starts with ``'python'`` and mentions ``'squidly.py'`` (not
    ``'squidly_single_model.py'``) in its elements. We match on the joined
    path element that ends with ``squidly.py``.
    """
    matches: list[ast.List] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        if not node.elts:
            continue
        first = node.elts[0]
        if not (isinstance(first, ast.Constant) and first.value == "python"):
            continue
        # Look for an os.path.join(..., 'squidly.py') element. The AST for
        # ``os.path.join(pckage_dir, 'squidly.py')`` is a Call whose last arg
        # is a Constant "squidly.py".
        targets_ensemble_worker = False
        for elt in node.elts:
            if (
                isinstance(elt, ast.Call)
                and elt.args
                and isinstance(elt.args[-1], ast.Constant)
                and elt.args[-1].value == "squidly.py"
            ):
                targets_ensemble_worker = True
                break
        if targets_ensemble_worker:
            matches.append(node)
    return matches


def _literal_string_elements(list_node: ast.List) -> list[str]:
    """Return the string-valued elements of a list literal, in order."""
    out: list[str] = []
    for elt in list_node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            out.append(elt.value)
    return out


def test_ensemble_worker_cmds_forward_mean_prob_and_mean_var() -> None:
    """Every subprocess cmd list that invokes ``squidly.py`` (the ensemble
    worker) must forward ``--mean_prob`` and ``--mean_var`` so the outer
    CLI's tunable thresholds actually reach the filter.

    This test would fail on squidly<=0.1.0, where the six ensemble cmd lists
    (chunked/non-chunked x cpu/iterative/default) all omitted these flags.
    """
    source = MAIN_PATH.read_text()
    tree = ast.parse(source)

    cmd_lists = _cmd_list_literals(tree)

    # Sanity check that we found the expected six ensemble cmd lists. If this
    # count changes, the test needs to be reviewed alongside the change.
    assert len(cmd_lists) == 6, (
        f"Expected 6 ensemble worker cmd lists in {MAIN_PATH.name}, "
        f"found {len(cmd_lists)}. If the CLI structure changed, update this "
        f"test alongside the change."
    )

    # For each cmd list, at least ONE of two things must be true:
    #   (a) the literal itself contains '--mean_prob' and '--mean_var', or
    #   (b) the assignment appends a list containing them (BinOp Add).
    # We check the surrounding assignment's RHS as a whole for robustness.
    source_lines = source.splitlines()
    for cmd_list in cmd_lists:
        # Grab a window of source around the cmd list so we can inspect the
        # full assignment (including any `+ threshold_args` suffix).
        start = cmd_list.lineno - 1
        end = (cmd_list.end_lineno or cmd_list.lineno)
        snippet = "\n".join(source_lines[start:end + 2])

        # The literal itself may not contain the flags if they're concatenated
        # via a threshold_args variable. In that case, the assignment must
        # reference `threshold_args` (or explicitly contain the flags).
        literal_strings = _literal_string_elements(cmd_list)
        has_flags_in_literal = (
            "--mean_prob" in literal_strings and "--mean_var" in literal_strings
        )
        has_threshold_args_ref = "threshold_args" in snippet
        assert has_flags_in_literal or has_threshold_args_ref, (
            f"Ensemble worker cmd at line {cmd_list.lineno} does not forward "
            f"--mean_prob / --mean_var to the subprocess. This reintroduces "
            f"the bug where the outer CLI's threshold flags were silently "
            f"ignored. Snippet:\n{snippet}"
        )


def test_threshold_args_variable_uses_cli_arguments() -> None:
    """If the fix uses a ``threshold_args`` helper variable, that variable
    must be built from the outer CLI's ``mean_prob`` and ``mean_var``
    arguments (not from hardcoded numbers). This prevents a regression where
    someone might "fix" the missing-flags bug by hardcoding defaults.
    """
    source = MAIN_PATH.read_text()
    tree = ast.parse(source)

    threshold_assignments: list[ast.Assign] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "threshold_args":
                    threshold_assignments.append(node)

    # If the current implementation does not use `threshold_args`, that's OK
    # as long as the other test above still passes. We only enforce this
    # invariant when the helper variable is actually present.
    if not threshold_assignments:
        return

    for assign in threshold_assignments:
        # The RHS should reference the argument names `mean_prob` and
        # `mean_var`. We search for Name nodes with those ids inside the RHS.
        referenced_names = {
            n.id for n in ast.walk(assign.value) if isinstance(n, ast.Name)
        }
        assert "mean_prob" in referenced_names, (
            f"threshold_args assignment at line {assign.lineno} does not "
            f"reference the `mean_prob` CLI argument. Did someone hardcode "
            f"the threshold instead of forwarding the user's value?"
        )
        assert "mean_var" in referenced_names, (
            f"threshold_args assignment at line {assign.lineno} does not "
            f"reference the `mean_var` CLI argument. Did someone hardcode "
            f"the threshold instead of forwarding the user's value?"
        )
