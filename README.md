# Forgewright Manufacturing — CNC Tool-Wear Analysis

## Background

Forgewright Manufacturing runs a CNC machining cell. They've sent us three data exports
covering a production shift across several machines in the cell and asked us to help
them catch **tool wear** before it causes scrap parts or machine damage.

The physics they care about: when a cutting tool gets dull or chipped, it
**chatters** — it vibrates more than a healthy tool would for the same amount of
cutting work. Cutting work shows up as electrical power draw. So a worn tool looks
like *high vibration relative to power*. A healthy tool's vibration scales
predictably with its power draw.

## Background for non-specialists

You don't need any manufacturing experience for this. Here's everything you need
to know about the setup:

- **CNC machining cell.** An automated machine that cuts metal parts. It works
  through a sequence of **jobs** — each job produces a batch of one part type
  (e.g., an aluminum bracket, a titanium fitting). Between jobs the machine sits
  **idle** while an operator loads material or sets up the next run.

- **Power meter.** An electrical meter on the machine, reporting power draw in
  **kilowatts (kW)**. When the machine is idle it draws a small baseline; when it's
  actively cutting it draws much more. Harder materials and heavier cuts draw more
  power. It logs roughly once per second.

- **Vibration sensor.** An accelerometer mounted on the spindle, reporting vibration
  amplitude in **g** (units of gravitational acceleration). It reports two values per
  reading: **RMS** (overall amplitude) and **peak** (the largest spike). When idle,
  vibration is near zero; when cutting, it rises. It logs a few times per second.

- **MES / production log.** The factory's manufacturing execution system — the
  software that tracks production. It records one row per job: the part type,
  operator, quantity, and the start/stop times the system logged.

- **The link between them.** Cutting a part takes mechanical work. The motor draws
  electrical power to do that work, and the cutting makes the machine vibrate. So
  during a job, power *and* vibration both go up together; during idle, both fall.
  Heavier cuts raise both. A healthy tool's vibration tracks its power draw in a
  predictable way; a worn tool vibrates more than its power draw would predict
  (that extra vibration is the "chatter"). Note that a high-power job is not the
  same as a worn-tool job — a big titanium cut is legitimately high-vibration
  without any wear.

## The data

Three files in `/data`, covering a production shift across several machines in the cell:

| File | What it is | Source |
|------|-----------|--------|
| `power.csv` | Electrical power draw, sampled by an inline power meter | Power meter |
| `vibration.csv` | Vibration amplitude (RMS and peak), sampled by an accelerometer on the spindle | Vibration sensor |
| `production_log.csv` | One row per production job: which part, which operator, quantity, start/stop | MES (manufacturing execution system) |

### Data dictionary

**power.csv** — power meter, logs ~1 reading/second

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | When the reading was taken |
| `machine_id` | string | Which machine the reading came from (the export covers several) |
| `power_kw` | float | Power draw in kilowatts. Low when idle, higher when cutting. |

**vibration.csv** — spindle accelerometer, logs ~2 readings/second

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | When the reading was taken |
| `machine_id` | string | Which machine the reading came from (the export covers several) |
| `vibration_rms_g` | float | RMS vibration amplitude in g. Near zero when idle. |
| `vibration_peak_g` | float | Peak vibration amplitude in g within the reading interval. |

**production_log.csv** — MES, one row per job

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | string | Job identifier (unique across the whole export) |
| `machine_id` | string | Which machine ran the job |
| `part_type` | string | The part being machined (material + part) |
| `operator` | string | Operator who ran the job |
| `quantity` | int | Number of parts in the batch |
| `start_time` | datetime | Job start |
| `end_time` | datetime | Job end |

## Your task

1. **For each production job**, determine the average power draw and the peak
   vibration during that job.
2. **Identify which jobs show signs of tool wear** — elevated vibration relative
   to the power draw. Produce a ranked list with your reasoning.
3. **State how confident you are** in your findings and what would make you more
   confident.

There is no answer key. We're interested in how you reason about the data, the
methods you choose, and how you validate your own conclusions.

**Build it to generalize.** We'll evaluate your pipeline by running it unchanged on a
**held-out shift from a different set of machines**. Treat the machine IDs, job IDs,
and dates in this sample as just one example — your code should read whatever machines
are present in the data and produce results without manual, per-machine edits. Anything
hardcoded to the specific names or values here won't transfer. You can preview how well
your pipeline generalizes with the optional [self-check](#self-check-optional) below.

## Deliverables

1. **Code** that runs end-to-end and produces your results from the three input files.
2. **A results file** (`output/job_summary.csv` or similar) with your per-job
   metrics and tool-wear assessment.
3. **A write-up (max ~2 pages)**, written entirely by you with **no LLM/AI help**
   (see Constraints & notes), covering:
   - Your approach and the methods you chose
   - Any issues you ran into with the data and how you handled them
   - How you checked your results
   - Your tool-wear findings and your confidence in them
   - What you'd do differently with more time or more data
4. **(Encouraged) Your EDA notebook** (e.g. `notebook.ipynb`) — the exploratory work you
   did to understand the data before computing anything. See the note below.

## A note on one-shot AI solutions

Be aware: we've tried solving this by handing the whole task to frontier models in a
single shot — **GPT-5.5 (extra-high)** and **Claude Opus 4.8 (high)** — and those solutions
**fail our internal validation set**. They produce plausible-looking output but miss what's
actually going on in the data. So a quick "load the files, ask the model for the worn jobs,
paste the answer" will not pass here, no matter how capable the model is.

What works is doing the legwork yourself. We strongly suggest you **start with an
exploratory data analysis (EDA)** — actually look at the data, plot it, profile it, sanity-
check it — before you write any analysis. That's usually where the real insight lives.
**Please include your EDA notebook** with your submission; it's one of the most useful
artifacts for us to see how you reasoned about the problem. (Use AI freely for the code and
the notebook — just remember the `WRITEUP.md` is yours alone; see below.)

## Output format

Your results file (`output/job_summary.csv`) should have one row per job with at
least the following columns. You may add columns or restructure as you see fit, but
the file should be machine-readable and contain the required fields.

| Column | Required | Description |
|--------|----------|-------------|
| `job_id` | Yes | Job identifier, from `production_log.csv` |
| `machine_id` | No | Which machine ran the job (encouraged when the export spans several) |
| `mean_power_kw` | Yes | Average power draw during the job |
| `peak_vibration_g` | Yes | Peak vibration during the job |
| `wear_score` | Yes | Your tool-wear score/metric for the job |
| `flagged` | Yes | Whether you flag this job as showing tool wear (`true`/`false`) |
| `rank` | No | Rank within your worn-tool list (encouraged) |
| `notes` | No | Anything notable about the job |

## Constraints & notes

- Use any language/libraries you like (Python + pandas is fine).
- **AI / coding agents are welcome for the code.** Use Copilot, Cursor, Claude, ChatGPT,
  or whatever you like — we expect you to. How you use them, and where you apply your own
  judgment, is part of what we evaluate.
- **The write-up (`WRITEUP.md`) is the one exception: write it entirely yourself, with no
  LLM/AI help.** No AI generation, drafting, editing, rephrasing, touch-ups, or fixes —
  your own words, start to finish. We want to read *your* reasoning and how you actually
  think about the problem, not a model's. A short, plainly-written write-up in your own
  voice is exactly what we're after.
- Time budget: ~4-6 hours. Scope accordingly; a well-reasoned partial solution
  beats a rushed complete one.
- Commit as you go (git log is a useful artifact for us).
- The code should run from a single command. Document how in your README.

## Project structure

```
/data
  power.csv             # power meter readings
  vibration.csv         # spindle accelerometer readings
  production_log.csv    # MES job log (one row per job)
/output
  job_summary.csv       # your results (per-job metrics + tool-wear assessment)
/src
  analyze.py            # entry point — runs end-to-end
/bin
  latent-cli-*          # submission CLI (do not edit)
run.sh / run.ps1        # submission wrappers (macOS·Linux / Windows)
WRITEUP.md
README.md
requirements.txt
```

You may restructure however you like, but the analysis must run end-to-end from a
single command.

## Setup

```bash
pip install -r requirements.txt
python src/analyze.py
```

## Self-check (optional)

You can preview how your pipeline scores on a **practice held-out shift** — a
*different* set of machines than this sample — using the same CLI:

```bash
./run.sh validate                 # scores src/analyze.py by default
./run.sh validate src/analyze.py  # or point at your entry file
.\run.ps1 validate                # Windows (PowerShell)
```

It runs your pipeline in an isolated sandbox on our side and returns an **F1 score**, a
**per-machine recall** breakdown, and **how much of the data your pipeline scored
without error**. You get **3 attempts**.

This is a practice check to see whether your method *generalizes* — it is **not** your
grade, and this practice shift is **not** the shift we grade on. A pipeline that hardcodes
values specific to this sample, rather than deriving them from whatever data is present,
will score poorly here — that's the signal it won't transfer.

### How we run your pipeline

The self-check — and our final evaluation — run your code the same way: from your repo
root, on a fresh copy of the three input files. For that to work:

- **Read the inputs relative to the working directory**: `data/power.csv`,
  `data/vibration.csv`, `data/production_log.csv`. We run from the repo root, so read
  `data/…` relative to the current directory — **don't** hardcode an absolute path or
  derive one from the script's own location (e.g. `Path(__file__).parent…`).
- **Write your results to `output/job_summary.csv`** with at least `job_id` and
  `flagged` (see [Output format](#output-format)) — that's the file we read to score you.
- **Declare any extra dependencies in `requirements.txt`.** The sandbox has `pandas`,
  `numpy`, `scipy`, `scikit-learn`, and `plotly` preinstalled; anything else (e.g.
  `statsmodels`) must be listed there so we can install it before running.
- Your pipeline must run end-to-end from a single command, with **no manual,
  per-machine edits**.

## Submission

When you're ready to submit, use the provided CLI to bundle and upload your
repository. From the root of your assignment repo:

```bash
./run.sh publish        # macOS / Linux
.\run.ps1 publish       # Windows (PowerShell)
```

This bundles your git repository — including the `.git` history we review, and
respecting `.gitignore` (so `.env`, `.venv`, `__pycache__`, etc. are excluded) — and
uploads it to our evaluation system.

### Providing Your Email

The CLI needs your email address to notify you that the upload succeeded. It will:

1. First try to read your email from `git config user.email`
2. If it is not set, prompt you to enter it interactively
3. Alternatively, pass it directly: `./run.sh publish --email you@example.com`

### Confirmation

After the upload completes, you will be asked to reply to the email you received for this assignment with your name and the repository name. Please send that reply so we can match your submission to your application.

**Troubleshooting**

- *"Not inside a git repository"* — initialize git first: `git init && git add -A && git commit -m "initial"`
- *"Repository must have at least one commit after initialization"* — the initial setup commit is not enough. Make at least one additional commit with your work before submitting.
- *Binary not found* — run it directly: `bin/latent-cli-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m) publish`
- *Upload fails* — check your internet connection
