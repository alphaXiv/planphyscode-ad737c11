import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # PlanPhys released-checkpoint reproduction

    Long-horizon planning asks an agent to track an evolving inventory and compose many small actions into one successful result. PlanPhys reports that explicit state-transition reasoning and limited exposure to long tasks both improve this ability. This notebook is a self-contained account of what our Kubernetes reproduction did—and did not—establish.

    **Verdict: behavioral reproduction blocked before inference.** No observed model accuracies are available, so neither target claim is confirmed or contradicted. A separate verifier audit completed successfully.
    """)
    return


@app.cell
def _():
    evidence = [
        ("World-model behavior", "Blocked", "No checkpoint outcomes"),
        ("Long-horizon exposure behavior", "Blocked", "No checkpoint outcomes"),
        ("Graph A verifier", "Complete", "1,440 tasks × 3 checks"),
    ]
    paper_targets = {
        "World model · middle avg@8": (46.6, 85.9),
        "World model · long avg@8": (22.7, 68.5),
        "Data exposure · middle pass@8": (0.83, 47.29),
        "Data exposure · long pass@8": (0.0, 11.46),
    }
    verifier = {
        "Gold terminal success": (1440, 1440),
        "Stored step-count agreement": (1440, 1440),
        "Corrupted-terminal rejection": (1440, 1440),
    }
    return evidence, paper_targets, verifier


@app.cell
def _(evidence, mo):
    rows = []
    for label, status, detail in evidence:
        color = "#16a34a" if status == "Complete" else "#64748b"
        rows.append(
            f"""
            <div style="display:grid;grid-template-columns:2fr 1fr 2fr;gap:12px;
                        padding:14px 0;border-bottom:1px solid #cbd5e1;">
              <strong>{label}</strong>
              <span style="color:{color};font-weight:700">{status}</span>
              <span>{detail}</span>
            </div>
            """
        )
    mo.Html(
        '<div aria-label="Evidence coverage">'
        '<div style="display:grid;grid-template-columns:2fr 1fr 2fr;gap:12px;'
        'padding:8px 0;color:#64748b">'
        "<span>Target</span><span>Status</span><span>Evidence</span></div>"
        + "".join(rows)
        + "</div>"
    )
    return


@app.cell
def _(mo, paper_targets):
    paper_rows = "\n".join(
        f"| {name} | {baseline:.2f}% | {treatment:.2f}% | "
        f"{treatment - baseline:+.2f} points |"
        for name, (baseline, treatment) in paper_targets.items()
    )
    mo.md(
        f"""
        ## The claims we intended to test

        The paper evaluates eight samples per task at temperature 0.4 with a 20-turn limit. The released final-checkpoint targets are:

        | Paper comparison | Baseline | Treatment | Reported change |
        |---|---:|---:|---:|
        {paper_rows}

        These are **paper values**, not reproduction measurements. We configured the same final comparison plus checkpoint 5400 to check whether the ordering was more than a terminal-checkpoint effect.
        """
    )
    return


@app.cell
def _(mo, verifier):
    audit_rows = "\n".join(
        f"| {name} | {passed}/{total} | {passed / total:.1%} |"
        for name, (passed, total) in verifier.items()
    )
    mo.md(
        f"""
        ## What did run: the Graph A verifier

        The public JSON gives each target, initial inventory, concrete item map, Graph A recipes, gold actions, and stored plan length. We reconstructed each action as a one- or two-input recipe transition, added its concrete output to inventory, and required the named target for terminal success.

        | Audit | Result | Rate |
        |---|---:|---:|
        {audit_rows}

        Selection was deterministic: 160 records per domain and difficulty, balanced over exact gold length, for 1,440 total tasks. Corruption replaced one terminal-action material with an impossible item. This validates the measurement rule, not model quality.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the model comparison is missing

    Four 4-GPU Kubernetes jobs covered the two claims at checkpoints 5400 and 9000. Attempt one exposed shell parsing in the injected Kubernetes command. Attempt two downloaded the selective public artifacts and passed the full verifier audit, then Transformers rejected tokenizer-produced `token_type_ids` before the first generation call. The two-attempt repair cap prevented tuning or silently moving the setup to new claim branches.

    The public `main` implementation contains the diagnosed compatibility fix, `return_token_type_ids=False`, but it is intentionally labeled unmeasured.

    ## Compute and provenance

    - Backend: Kubernetes
    - Cluster GPU: NVIDIA RTX PRO 6000 Blackwell
    - Peak concurrent GPU allocation: 16
    - Actual elapsed wall time: 337 seconds (0.094 hours)
    - Successful evidence run: CPU-only verifier audit, 36 seconds
    - Public repository: [alphaXiv/planphyscode-ad737c11](https://github.com/alphaXiv/planphyscode-ad737c11)

    ## Bottom line

    Both behavioral claims remain **inconclusive under this setup**. The strongest valid result is narrower: the independently reconstructed Graph A verifier exactly fits the released gold records and rejects a decisive terminal corruption. A full follow-up should run the four already-defined conditions once, without tuning to the paper’s numbers, and then report paired bootstrap intervals plus strict/lenient parser sensitivity.
    """)
    return


if __name__ == "__main__":
    app.run()
