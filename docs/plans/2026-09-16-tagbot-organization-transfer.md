# TagBot Organization Transfer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore automatic tags, releases, and tagged documentation builds while the organization policy disables deploy keys.

**Architecture:** TagBot writes tags and releases with the repository-scoped `GITHUB_TOKEN`. The workflow records tag refs before and after TagBot runs, then sends a `repository_dispatch` event for every new tag so the documentation workflow can check out that tag explicitly.

**Tech Stack:** GitHub Actions, GitHub CLI, JuliaRegistries/TagBot, Documenter.jl

### Task 1: Replace the disabled deploy-key path

**Files:**
- Modify: `.github/workflows/TagBot.yml`

1. Preserve `contents: write` and change `issues` to `write` so TagBot can publish releases and report failures.
2. Record existing tag refs before invoking TagBot.
3. Remove the `ssh: ${{ secrets.DOCUMENTER_KEY }}` input so TagBot uses `GITHUB_TOKEN`.
4. Compare tag refs after TagBot exits and send a `tagbot-release` repository dispatch for each new tag.

### Task 2: Accept tagged documentation dispatches

**Files:**
- Modify: `.github/workflows/Documentation.yml`

1. Add a `repository_dispatch` trigger restricted to `tagbot-release`.
2. For that event, check out the tag supplied in `client_payload.tag`; preserve the existing ref for push and pull-request runs.

### Task 3: Verify and publish

1. Run `git diff --check` and parse both workflow files as YAML.
2. Push the branch and confirm GitHub accepts both workflows.
3. Run TagBot manually and verify it succeeds without an SSH permission error.
4. Verify each new tag produces a `repository_dispatch` documentation run at that tag.

The already-registered `v1.0.76` release is a one-time exception: its commit changes workflow files, which GitHub does not allow `GITHUB_TOKEN` to tag. Create that tag and release manually before validating the permanent workflow.
