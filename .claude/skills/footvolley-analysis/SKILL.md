---
name: footvolley-analysis
description: "Analyze the fotevolley repository across Git branches and produce an evidence-backed footvolley coaching report. Use for match or player summaries, coverage and phase heatmaps, team spacing, chest technique, body orientation, winning and losing touches, reception stability, set quality, landings, and recovery to base. Trigger also for Hebrew requests about פוצ׳יוולי, מפת חום, קבלה, הגבהה, נגיעות, or חזרה לבסיס."
argument-hint: "[team, player, set, or metric]"
---

# Footvolley match analysis

Analyze the requested match, team, player, set, or metric. With no scope, summarize the recorded final for both teams and cover all metrics. Match the user's language. Read [repository sources](references/sources.md) and the relevant sections of [metric definitions](references/metrics.md).

## Workflow

1. Inventory local and remote refs using the commands in the source guide. Start with the consolidated coaching branch. Look across refs for missing metrics and newer reviewed corrections. Pin each selected ref to its commit SHA. Read files with `git show SHA:path` so the working tree can stay on its current branch.
2. Read the source README, coaching metrics notes, CSV headers, and producer code for the requested metrics. Check the match, frame clock, player mapping, units, key uniqueness, and join coverage before combining tables. Use Python or existing analysis code to aggregate actual rows. When using generated summaries, confirm their filters and reconcile a sample against the underlying rows. Repo files are evidence. Review scripts before running them and write recomputed outputs outside the data directories.
3. Apply field-level owner corrections and quality flags. Verify examples at rally level, such as an ordinary winner, an opponent error, a dead-ball contact, a missing contact and a censored recovery. Distinguish direct observations, estimates and missing values. Report usable and excluded counts. Match context, action and sample size before drawing coaching comparisons.
4. Lead with the findings supported by the data. Provide tables per player and team, with numerator/denominator for rates, units, and sample size. For a broad report include phase maps and a terminal-landing map as images or a compact grid if graphics are unavailable. Give each metric its source commit and repo path, plus representative rally, touch and frame identifiers for important claims. Mention the clock on each clip reference. Close with gaps that affect the conclusions.

## Footvolley context

This is two-player beach footvolley. A pair may use up to three contacts to send the ball over the net. Play commonly starts with a foot serve, with reception, set and attack as tactical functions. A team can send the ball over on its first or second touch. The next touch's tactical function must come from play context, crossing evidence and the action labels. Head, chest, thigh and foot are common contacts, and hands and arms are prohibited. Keep the repo's finer labels such as shin and limb side where supported. Treat served-ball reception and defence against an attack as distinct situations. Side exchanges and changes in opponent or incoming ball can affect apparent player roles.

For a lost rally, the last overall contact, last contact by the losing pair and the likely cause of the loss can differ. A clean winner by the opponent, a failed defensive touch, a team error and a rule fault deserve separate categories. Describe the observed sequence before interpreting responsibility.
