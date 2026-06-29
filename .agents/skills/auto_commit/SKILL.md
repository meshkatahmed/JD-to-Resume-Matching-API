---
name: auto_commit
description: >
  Generates a conventional commit message by inspecting the current git diff,
  stages all changes, commits with the generated message, and pushes to the
  current remote branch. Triggered by phrases like "commit and push",
  "push the code", "create a commit", "auto commit", or "generate commit message".
---

# Auto Commit Skill

## Goal
Inspect the working tree, generate a well-formed **Conventional Commits** message,
then stage → commit → push — all in one flow.

---

## Commit Message Format

Follow the [Conventional Commits v1.0](https://www.conventionalcommits.org/) spec:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

### Types
| Type | When to use |
|---|---|
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `refactor` | Code restructuring with no behaviour change |
| `docs` | Documentation / comments only |
| `style` | Formatting, whitespace — no logic change |
| `test` | Adding or updating tests |
| `chore` | Build config, dependencies, tooling |
| `perf` | Performance improvement |

### Scope
Use the **primary module or filename** that was changed (without extension),
e.g. `matching_service`, `tfidf_vectorizer`, `main`, `index_2`, `utils`.
If changes span multiple unrelated areas, omit the scope.

### Summary line rules
- Lowercase, imperative mood ("add", "fix", "remove" — not "added" or "fixes")
- No trailing period
- Max 72 characters

### Body (include when useful)
- Explain **why**, not what (the diff already shows what)
- Wrap at 72 characters
- Separate from summary with a blank line

---

## Step-by-Step Execution

### 1. Inspect the working tree

Run these commands and study the output carefully before writing anything:

```powershell
git status
git diff HEAD
```

If nothing is staged or modified, inform the user there is nothing to commit and stop.

### 2. Determine type and scope

Read the diff and classify:
- Which **files** changed? → infer scope (use the most significant one)
- What kind of change? → infer type from the table above
- What is the net effect for a user or developer? → write the summary

### 3. Draft the commit message

Compose the full message string. Examples for this codebase:

```
feat(tfidf_vectorizer): add pure-python TF-IDF vectorizer with smooth IDF

Implements TFIDFVectorizer with fit/transform/fit_transform methods.
Uses sklearn-style smooth IDF formula; no external ML dependencies.
```

```
feat(matching_service): add compute_similarity_scores_tfidf function

Fits a single TFIDFVectorizer over the full corpus (JD + all CVs) so
IDF weights reflect term rarity across the whole document collection.
```

```
feat(main): add vectorization_method query param to /match_cv endpoint

Defaults to "cbow" so existing API consumers are unaffected.
Passing "tfidf" routes to the new TF-IDF matching function.
```

```
feat(index_2): add CBOW / TF-IDF pill toggle to match section

Active method is appended as ?vectorization_method= query param on fetch.
Hint text beneath the toggle describes each method briefly.
```

### 4. Stage, commit, and push

Run the following commands **in order**, waiting for each to succeed:

```powershell
# Stage everything
git add .

# Commit with the generated message
# Use -m for single-line; use a here-string for multi-line body
git commit -m "<type>(<scope>): <summary>" -m "<body paragraph>"

# Push to the current tracking branch
git push
```

If there is no upstream set yet:
```powershell
git push --set-upstream origin <current-branch>
```

To get the current branch name if needed:
```powershell
git rev-parse --abbrev-ref HEAD
```

### 5. Confirm

After a successful push, report back:
- The exact commit message used
- The branch pushed to
- The short commit hash (`git rev-parse --short HEAD`)

---

## Error Handling

| Situation | Action |
|---|---|
| Nothing to commit | Inform user, stop |
| Merge conflict markers in diff | Warn user, do not commit, ask them to resolve first |
| `git push` rejected (non-fast-forward) | Run `git pull --rebase` then retry push; if still failing, report the error |
| Untracked sensitive files (`.env`, secrets) | Skip staging those files, warn the user |

---

## Notes specific to this repository

- The project uses **Poetry** — never commit changes to `.venv/`
- `job_descriptions/` and `cvs/` hold uploaded user files — do not commit their contents
- `__pycache__/` is already in `.gitignore` — no action needed
