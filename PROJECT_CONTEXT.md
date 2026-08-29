# PROJECT_CONTEXT.md

# Object-Centric GitHub Development Pattern Mining

## Project Context, Current State, Plan, and Continuation Instructions

------------------------------------------------------------------------

## 1. Project Overview

This is a **2-week Data Mining short-semester project**.

The project should be technically ambitious, interesting to students,
and impressive to the professor, while remaining realistic enough to
complete in approximately two weeks.

The current project direction is:

> **Mining Hidden Development Patterns from Object-Centric GitHub Commit
> Data**

The core idea is to use **Object-Centric Event Log (OCEL)** data from a
real GitHub software project and combine:

-   Exploratory Data Analysis (EDA)
-   Object-centric process mining
-   Temporal pattern mining
-   Feature engineering
-   Unsupervised machine learning
-   Anomaly detection
-   Graph mining / network analysis
-   Potentially a small interactive visualization/dashboard

The project should NOT use machine learning merely for decoration. Each
technique should answer a meaningful data-mining question.

------------------------------------------------------------------------

# 2. Main Research Question

Current working research question:

> **Can object-centric event data and graph-based features reveal
> unusual and recurring patterns in software development activity?**

Possible supporting questions:

1.  What does a typical software-development commit look like?
2.  Do different commit activity types affect different numbers of
    files?
3.  How does development behavior change over time?
4.  Which files and branches are strongly connected through development
    activity?
5.  Are there distinct clusters of commit behavior?
6.  Which commits are behaviorally unusual?
7.  Can we explain WHY a commit is considered unusual?
8.  Can object-centric and graph-based features reveal patterns that
    ordinary tabular analysis would miss?

Do not prematurely claim that unusual = bad. The project is primarily
about **behavioral/anomalous patterns**, not automatically detecting
bugs or malicious activity.

------------------------------------------------------------------------

# 3. Dataset

Current dataset:

**Angular GitHub commits OCEL log**

Files currently in the project:

``` text
data/
├── angular_github_commits_ocel.csv
└── angular_github_commits_ocel.jsonocel
```

Notebook:

``` text
notebooks/
└── data_explore.ipynb
```

Project structure currently resembles:

``` text
process-mining-project/
├── data/
├── models/
├── notebooks/
│   └── data_explore.ipynb
├── results/
├── src/
└── README.md
```

------------------------------------------------------------------------

# 4. Environment

Conda environment:

``` text
processmining
```

Python:

``` text
Python 3.12.13
```

Python executable:

``` text
/home/mish/anaconda3/envs/processmining/bin/python
```

PM4Py:

``` text
2.7.23.6
```

PM4Py is installed and working.

Imports currently used:

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pm4py as ppy
```

------------------------------------------------------------------------

# 5. Important OCEL Format Discovery

Initially the file was attempted with:

``` python
ppy.read_ocel2_json(...)
```

This produced:

``` text
KeyError: 'events'
```

The raw JSON was inspected:

``` python
import json

with open("../data/angular_github_commits_ocel.jsonocel", "r") as f:
    data = json.load(f)

print(data.keys())
```

Output:

``` text
dict_keys([
    'ocel:global-event',
    'ocel:global-object',
    'ocel:global-log',
    'ocel:events',
    'ocel:objects'
])
```

This showed that the file is an **OCEL 1.0-style JSON**, not the OCEL
2.0 JSON structure expected by `read_ocel2_json()`.

Correct loading method:

``` python
ocel = ppy.read_ocel_json(
    "../data/angular_github_commits_ocel.jsonocel"
)
```

This successfully loaded the dataset.

------------------------------------------------------------------------

# 6. What the Dataset Represents

The event log represents GitHub software-development activity.

In this dataset:

## Events

Events correspond to GitHub commits.

There are:

``` text
27,842 events
```

The event table contains fields including:

``` text
ocel:eid
ocel:timestamp
ocel:activity
commit_message
author_name
merge
...
```

Examples observed:

``` text
activity = initial
activity = build
activity = chore
```

Example event:

``` text
ocel:activity = initial
ocel:timestamp = 2014-09-18 16:12:01+00:00
commit_message = Initial commit
author_name = Miško Hevery
```

------------------------------------------------------------------------

# 7. Object Types

The dataset has exactly two object types:

``` text
['branches', 'files']
```

Total objects:

``` text
28,317
```

Breakdown:

``` text
files       28,178
branches       139
```

Examples observed:

Branch object:

``` text
8.2.x
```

File objects included paths such as:

``` text
modules/angular2/src/core/compiler/...
modules/angular2/src/core/compiler/shadow_dom...
projects/ng-devtools/src/lib/devtools-tabs/...
```

------------------------------------------------------------------------

# 8. Event-Object Relationships

The OCEL relationship table is:

``` python
ocel.relations
```

Columns:

``` text
[
    'ocel:eid',
    'ocel:activity',
    'ocel:timestamp',
    'ocel:oid',
    'ocel:type',
    'ocel:qualifier'
]
```

Total event-object relationships:

``` text
2,451,100
```

Breakdown:

``` text
branches    2,271,272
files         179,828
```

Important interpretation:

A relationship means that an event/commit is connected to an object.

Conceptually:

``` text
Commit
 ├── Branch
 ├── Branch
 ├── File
 ├── File
 └── File
```

A large number of branch relationships does NOT necessarily mean a
commit directly modified all those branches. It can reflect the branches
in which the commit occurs/is represented.

This distinction must be preserved in the final report.

------------------------------------------------------------------------

# 9. First File-Level Analysis

File relationships were isolated:

``` python
file_relations = ocel.relations[
    ocel.relations["ocel:type"] == "files"
]

print("File relationships:", len(file_relations))
```

Result:

``` text
179,828
```

Files per commit were calculated:

``` python
files_per_commit = (
    file_relations
    .groupby("ocel:eid")["ocel:oid"]
    .nunique()
)
```

Statistics:

``` text
count    27828.000000
mean         6.462124
std         30.877579
min          1.000000
25%          1.000000
50%          2.000000
75%          5.000000
max       1637.000000
```

Important discoveries:

-   Median commit affects only **2 files**.
-   Mean is **6.46 files**.
-   75% of commits affect **5 or fewer files**.
-   Maximum observed is **1,637 files**.
-   Distribution is strongly right-skewed.
-   There are 27,842 events but only 27,828 commits with recorded file
    relationships, meaning **14 events have no file relationship**.

This long-tailed distribution is a promising basis for anomaly
detection, but large does not automatically mean bad.

------------------------------------------------------------------------

# 10. Visualization Already Created

Histogram:

``` python
plt.figure(figsize=(10, 5))

plt.hist(files_per_commit, bins=50)

plt.xlabel("Number of files affected")
plt.ylabel("Number of commits")
plt.title("Distribution of Files per Commit")

plt.show()
```

Because the maximum is 1,637, a second focused histogram was created:

``` python
plt.figure(figsize=(10, 5))

plt.hist(files_per_commit, bins=50)

plt.xlim(0, 100)

plt.xlabel("Number of files affected")
plt.ylabel("Number of commits")
plt.title("Files Affected per Commit (0–100)")

plt.show()
```

The visualization clearly shows a heavily right-skewed distribution with
most commits concentrated at low file counts and a small long tail.

------------------------------------------------------------------------

# 11. Commit Activity Discovery

The dataset contains:

``` text
67 activity categories
```

Most frequent observed categories:

``` text
fix             6093
docs            5625
refactor        4122
build           3992
feat            2880
```

There are many rare activity labels as well, including examples such as:

``` text
rendererv2
code
filetree
table
consolidated
```

The existence of 67 activity types creates a useful research direction:

> Do different development activities have different file-change
> patterns?

For example, we can investigate whether `fix`, `docs`, `refactor`,
`build`, and `feat` have different distributions of files changed.

------------------------------------------------------------------------

# 12. Commit Feature Table --- Started

The next major table is intended to combine event information with
file-count information.

Code already planned:

``` python
files_per_commit_df = files_per_commit.rename(
    "files_changed"
).reset_index()

commit_features = ocel.events.merge(
    files_per_commit_df,
    on="ocel:eid",
    how="left"
)

commit_features["files_changed"] = (
    commit_features["files_changed"]
    .fillna(0)
    .astype(int)
)

print(commit_features.shape)
```

Conceptual structure:

``` text
commit_features
│
├── event ID
├── timestamp
├── activity
├── author
├── commit message
├── merge
└── files_changed
```

This table will eventually become the foundation for ML feature
engineering.

------------------------------------------------------------------------

# 13. Activity vs Commit Size Analysis

The next analysis is:

``` python
activity_file_stats = (
    commit_features
    .groupby("ocel:activity")["files_changed"]
    .agg(["count", "mean", "median", "max"])
    .sort_values("median", ascending=False)
)

activity_file_stats
```

Purpose:

Determine whether different activity types have different typical commit
sizes.

Median is especially important because the distribution is highly
skewed.

A planned visualization is a boxplot for the top 10 activities.

Because the installed Matplotlib version rejected:

``` python
labels=
```

the correct parameter is:

``` python
tick_labels=
```

Current plotting code:

``` python
top_activities = (
    commit_features["ocel:activity"]
    .value_counts()
    .head(10)
    .index
)

plot_data = [
    commit_features.loc[
        commit_features["ocel:activity"] == activity,
        "files_changed"
    ]
    for activity in top_activities
]

plt.figure(figsize=(12, 6))

plt.boxplot(
    plot_data,
    tick_labels=top_activities,
    showfliers=False
)

plt.xlabel("Commit activity")
plt.ylabel("Files changed")
plt.title("Commit Size by Development Activity")

plt.xticks(rotation=45)

plt.show()
```

`showfliers=False` is only for visualization; extreme observations are
NOT removed from the underlying data.

------------------------------------------------------------------------

# 14. Overall Technical Architecture

Target architecture:

``` text
                         GITHUB OCEL
                              │
                              ▼
                    ┌───────────────────┐
                    │ Data Understanding│
                    └─────────┬─────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Event Analysis  Object Analysis  Relations
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                       Pattern Discovery
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
           Temporal       Activity        File/Branch
           Patterns       Patterns        Patterns
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                    Feature Engineering
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Commit Features      Graph Features
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     Data Mining / ML
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
             Clustering            Anomaly Detection
                  │                       │
                  └───────────┬───────────┘
                              ▼
                        Graph Mining
                              │
                              ▼
                     Hidden Patterns
                              │
                              ▼
                    Explainable Results
                              │
                              ▼
                  Dashboard / Presentation
```

------------------------------------------------------------------------

# 15. Planned Project Components

## Component A --- Exploratory Data Mining

Analyze:

-   number of commits
-   number of files
-   number of branches
-   activity frequencies
-   commit sizes
-   author activity
-   temporal distribution
-   file popularity
-   branch popularity

------------------------------------------------------------------------

## Component B --- Temporal Mining

Investigate:

-   commits over time
-   development intensity
-   activity changes over time
-   active periods
-   bursts of development
-   gaps in development
-   commit frequency
-   time between commits

Potential visualizations:

-   commits per month
-   activity over time
-   rolling commit frequency
-   heatmaps by time period
-   development bursts

------------------------------------------------------------------------

## Component C --- Process Mining

Use PM4Py where useful.

Potential questions:

-   What activity patterns occur?
-   Which activities commonly follow one another?
-   Are there recurring development sequences?
-   Do different activity categories form recognizable process patterns?

Potential process representation:

``` text
feat → fix → refactor
```

or:

``` text
docs → docs → fix
```

or:

``` text
build → build → fix → refactor
```

Important: process-mining conclusions must respect the limitations of
the event log and its ordering.

------------------------------------------------------------------------

## Component D --- Developer Behavior

Possible features:

-   commits per author
-   average files per commit
-   median files per commit
-   activity diversity
-   active development periods
-   number of files touched
-   number of branches associated
-   temporal activity
-   unusual behavior compared with the developer's own baseline

Potential question:

> Do different developers exhibit distinct development behavior
> patterns?

Avoid making psychological claims. This is software-development behavior
recorded in the dataset.

------------------------------------------------------------------------

## Component E --- File Interaction Mining

Build relationships based on files appearing in the same commit.

Example:

``` text
Commit A → File A + File B
Commit B → File A + File C
```

This creates a file co-change relationship:

``` text
File A ─── File B
   │
   └──── File C
```

Potential questions:

-   Which files are frequently changed together?
-   Which files are central?
-   Which file groups form communities?
-   Which files connect otherwise separate communities?

------------------------------------------------------------------------

# 16. Graph Mining Plan

Potential graph:

### Bipartite commit-file graph

``` text
Commit 1 ─── File A
         └── File B

Commit 2 ─── File B
         └── File C
```

Potential projected file graph:

``` text
File A ───── File B ───── File C
```

Edge weight:

> Number of commits in which two files occurred together.

Potential graph metrics:

-   degree
-   weighted degree
-   betweenness centrality
-   connected components
-   community detection
-   clustering coefficient

Potential advanced analysis:

> Identify files that connect otherwise separate development
> communities.

------------------------------------------------------------------------

# 17. Machine Learning Plan

ML should be introduced only after feature engineering and EDA.

## Unsupervised clustering

Possible algorithms:

-   K-Means
-   DBSCAN
-   hierarchical clustering

Potential goal:

> Discover natural types of commits based on behavior.

Possible commit features:

``` text
files_changed
branches_count
activity_encoded
author_commit_count
time_since_previous_commit
commit_frequency
activity_frequency
file_reuse
...
```

Categorical variables should be encoded appropriately rather than
blindly converting labels to numbers.

------------------------------------------------------------------------

## Anomaly Detection

Potential algorithms:

-   Isolation Forest
-   Local Outlier Factor
-   One-Class SVM

Recommended first choice:

> **Isolation Forest**

Reason:

-   suitable for unsupervised anomaly detection
-   works well as a baseline
-   relatively easy to explain
-   suitable for mixed engineered numerical features after preprocessing
-   computationally practical for \~28k events

Potential anomaly features:

``` text
files_changed
branches_count
time_since_previous_commit
author_activity
commit_frequency
activity_frequency
file_reuse
graph metrics
```

The output should be:

``` text
commit
    ↓
feature vector
    ↓
anomaly model
    ↓
anomaly score
    ↓
explanation
```

The model should not simply call a commit "bad."

Instead:

> "This commit is statistically/behaviorally unusual relative to the
> observed development history."

------------------------------------------------------------------------

# 18. Advanced "WOW" Component

The final system should ideally allow the user to inspect an unusual
commit.

Conceptual output:

``` text
┌──────────────────────────────────────┐
│          COMMIT ANALYSIS             │
├──────────────────────────────────────┤
│ Activity:       refactor             │
│ Author:         ...                  │
│ Files changed:  127                  │
│ Branches:       ...                  │
│ Timestamp:      ...                  │
│                                      │
│ Anomaly score:  0.94                 │
└──────────────────────────────────────┘
```

Then explain:

``` text
Why unusual?

✓ Much larger than typical commits
✓ Rare activity type
✓ Unusual temporal behavior
✓ Unusual file-community involvement
✓ Unusual graph connectivity
```

The exact explanations must be generated from actual computed features,
not fabricated.

------------------------------------------------------------------------

# 19. Important Scientific Principle

The project should distinguish:

### Detection

"This event is unusual."

from:

### Interpretation

"These measurable properties make it unusual."

from:

### Causal claims

"This caused a bug."

The project should generally avoid unsupported causal claims.

The safest research framing is:

> **behavioral anomaly / unusual development pattern detection**

rather than:

> bug detection

unless we later obtain an appropriate ground-truth label.

------------------------------------------------------------------------

# 20. Two-Week Execution Plan

Because this is a short semester, prioritize a coherent system over
excessive algorithms.

## Day 1 --- Dataset + OCEL

Completed / mostly completed:

-   environment
-   PM4Py
-   dataset loading
-   OCEL format discovery
-   events
-   objects
-   relationships

Status: **DONE**

------------------------------------------------------------------------

## Day 2 --- EDA + activity patterns

Current stage:

-   commit size distribution
-   activity distribution
-   activity vs file count

Then:

-   temporal distribution
-   author activity
-   branch statistics
-   file statistics

Status: **CURRENT**

------------------------------------------------------------------------

## Day 3 --- Temporal and process mining

Build:

-   commits over time
-   activity over time
-   activity transitions
-   commit bursts
-   time gaps

------------------------------------------------------------------------

## Day 4 --- Feature engineering

Create commit-level feature table.

Possible features:

``` text
files_changed
branches_count
author_commit_count
author_unique_files
time_since_previous_commit
activity_frequency
files_changed_rolling_average
commit_hour
commit_day
commit_month
...
```

------------------------------------------------------------------------

## Day 5 --- Statistical anomaly baseline

Before ML:

-   IQR
-   z-score where appropriate
-   percentile-based thresholds
-   robust statistics

This gives a simple baseline to compare ML against.

------------------------------------------------------------------------

## Day 6 --- Clustering

Try:

-   K-Means
-   possibly DBSCAN

Evaluate clusters with:

-   silhouette score
-   cluster size
-   feature interpretation

------------------------------------------------------------------------

## Day 7 --- Isolation Forest

Train anomaly detector.

Investigate:

-   top anomalies
-   anomaly score distribution
-   feature differences

------------------------------------------------------------------------

## Day 8 --- Graph construction

Build:

``` text
commit ↔ file
```

and/or:

``` text
file ↔ file
```

co-change graph.

------------------------------------------------------------------------

## Day 9 --- Graph mining

Calculate:

-   degree
-   weighted degree
-   centrality
-   communities

Connect graph features back to commits.

------------------------------------------------------------------------

## Day 10 --- Combine models

Combine:

``` text
temporal features
+
process features
+
behavioral features
+
graph features
```

Build a stronger anomaly-analysis pipeline.

------------------------------------------------------------------------

## Day 11 --- Explainable results

For selected anomalies:

-   show feature values
-   compare against normal commits
-   show affected files
-   show activity
-   show graph neighborhood
-   show timeline context

------------------------------------------------------------------------

## Day 12 --- Visualization / dashboard

Potential tools:

-   Streamlit
-   Plotly
-   NetworkX + Matplotlib

If time is limited, prioritize a polished notebook rather than spending
too much time on a web app.

------------------------------------------------------------------------

## Day 13 --- Report + presentation

Prepare:

-   problem
-   dataset
-   methodology
-   EDA
-   mining results
-   ML
-   graph analysis
-   anomaly examples
-   limitations
-   conclusion

------------------------------------------------------------------------

## Day 14 --- Final polish

-   clean code
-   reproducibility
-   figures
-   README
-   presentation
-   demo
-   backup

------------------------------------------------------------------------

# 21. Priority Order if Time Runs Out

The project should have a minimum viable version.

### Must-have

1.  Dataset understanding
2.  EDA
3.  Feature engineering
4.  Activity/temporal mining
5.  Isolation Forest anomaly detection
6.  Explainable anomaly examples

### Strongly recommended

7.  Process mining
8.  File co-change graph
9.  Graph features

### Nice-to-have

10. Clustering
11. Interactive dashboard
12. Advanced community detection

Do NOT sacrifice the core analysis just to add more algorithms.

------------------------------------------------------------------------

# 22. What NOT to Do

Avoid:

-   using deep learning just because it sounds advanced
-   adding random algorithms without a research question
-   claiming anomalies are bugs without ground truth
-   deleting outliers merely because they make plots ugly
-   treating branch relationships as direct file modifications
-   treating activity labels as perfect ground truth
-   using every available PM4Py function without understanding it
-   building a dashboard before the analysis works
-   spending several days on UI

The strongest project is one with a clear chain:

``` text
Question
   ↓
Data
   ↓
Mining method
   ↓
Evidence
   ↓
Interpretation
```

------------------------------------------------------------------------

# 23. Current Notebook State

Known working objects:

``` python
ocel
file_relations
files_per_commit
files_per_commit_df
commit_features
```

Known useful imports:

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pm4py as ppy
```

The current notebook is at the transition between:

``` text
PHASE 1 — Data Understanding
```

and:

``` text
PHASE 2 — Pattern Discovery
```

The next immediate task is to finish the activity-vs-commit-size
analysis and then begin **temporal mining**.

------------------------------------------------------------------------

# 24. Immediate Next Steps

Do NOT jump directly into ML.

Next sequence:

### Step A

Run:

``` python
activity_file_stats = (
    commit_features
    .groupby("ocel:activity")["files_changed"]
    .agg(["count", "mean", "median", "max"])
    .sort_values("median", ascending=False)
)

activity_file_stats
```

Use it to inspect which activity types tend to involve more files.

### Step B

Create the top-10 activity boxplot using `tick_labels=`.

### Step C

Begin temporal analysis:

``` text
How does development activity change over time?
```

### Step D

Create temporal features.

### Step E

Move toward the ML feature matrix.

------------------------------------------------------------------------

# 25. Continuation Instructions for Another AI / Kilo Code

If this project is continued in another AI coding assistant, it should:

1.  Read this file first.
2.  Inspect the existing repository before changing anything.
3.  Preserve the current conda environment assumptions.
4.  Do not replace the dataset without a strong reason.
5.  Do not assume OCEL 2.0; the current JSON is OCEL 1.0-style.
6.  Use:

``` python
ppy.read_ocel_json(...)
```

for the current `.jsonocel` file. 7. Treat `ocel.events`,
`ocel.objects`, and `ocel.relations` as the primary OCEL structures. 8.
Preserve the distinction between branch relationships and file
relationships. 9. Continue step-by-step because the project owner has
limited Data Mining experience. 10. Explain important code before
introducing it. 11. Prefer simple, reproducible implementations. 12. Do
not ask the project owner to repeatedly paste notebook outputs unless
absolutely necessary; inspect files/code directly when possible. 13.
Avoid unnecessary package installation. 14. Keep the two-week deadline
in mind. 15. Prioritize a technically coherent final result over feature
bloat.

------------------------------------------------------------------------

# 26. Final Project Vision

The final project should tell this story:

> GitHub software development generates a complex object-centric event
> history. Instead of analyzing commits as isolated rows, we model their
> relationships with files and branches, examine how development evolves
> over time, discover recurring behavioral patterns, construct a
> development graph, and use machine learning to identify unusually
> structured development events.

The final result should answer:

> **What hidden patterns exist in the development process, and which
> events behave unusually when viewed through temporal, process, and
> graph-based perspectives?**

That is the central project.
