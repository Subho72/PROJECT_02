# Student Performance Analysis
### Pluto Academy — Data Analytics Internship Program | Project 02

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![EDA](https://img.shields.io/badge/EDA-%26%20Reporting-orange)
![Status](https://img.shields.io/badge/Status-Complete%20by%20Week%204-green)

---

## Problem Statement

> You are a Data Analyst hired by a school. The principal has asked you to analyse student performance data to understand which factors affect grades, identify at-risk students, and produce a report with 3 actionable recommendations that can improve outcomes in the next academic year.

---

## Dataset

**Source:** [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) — Kaggle (free)

- 1,000+ student records
- Features: gender, race/ethnicity, parental level of education, lunch type, test preparation course, math score, reading score, writing score

**Download the CSV**, rename it `StudentsPerformance.csv`, and place it in the project root.

---

## Project Structure

```
student_performance_analysis/
│
├── analysis.py                        # Full pipeline as a Python script
├── Student_Performance_Analysis.ipynb # Interactive Jupyter notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
└── output/                            # Auto-created on first run
    ├── chart1_boxplot_parental_education.png
    ├── chart2_bar_test_prep.png
    ├── chart3_correlation_heatmap.png
    ├── chart4_grouped_bar_gender.png
    ├── chart5_histogram_total_score.png
    ├── chart6_scatter_reading_vs_math.png
    ├── at_risk_students.csv
    └── principals_report.txt
```

---

## What the Code Builds

### Step 1 — Data Exploration & Cleaning
- Loads the dataset and inspects shape, data types, and null values
- Renames columns for convenience
- Adds derived columns: `total_score` and `avg_score`
- Prints a **5-line human-readable summary**

### Step 2 — Factor Analysis (5 Questions)
| # | Question |
|---|----------|
| 1 | Does parental education level affect scores? |
| 2 | Do students who complete test prep score higher? |
| 3 | What is the correlation between reading, writing, and math scores? |
| 4 | Which gender performs better in which subject? |
| 5 | What is the distribution of total scores? |

### Step 3 — Visualisations (6 Charts)
| Chart | Type | Description |
|-------|------|-------------|
| 1 | Box Plot | Score spread by parental education level |
| 2 | Bar Chart | Average scores — test prep vs no test prep |
| 3 | Heatmap | Correlation matrix for math, reading, writing |
| 4 | Grouped Bar | Gender vs performance per subject |
| 5 | Histogram | Total score distribution with mean/median markers |
| 6 | Scatter Plot | Reading vs math scores, coloured by gender |

### Step 4 — At-Risk Student Segmentation
- **Definition:** Student scoring **below 50** in any single subject
- Counts at-risk students and calculates the percentage
- Breaks down at-risk rates by parental education, test prep status, gender, and ethnicity
- Exports the full at-risk list to `output/at_risk_students.csv`

### Step 5 — Principal's Report
A concise 1-page report saved to `output/principals_report.txt` containing:
- **Executive Summary** (3 sentences)
- **5 Key Findings** (data-backed)
- **3 Actionable Recommendations** the school can implement

---

## Quick Start

### Option A — Jupyter Notebook (recommended)

```bash
# 1. Clone / download this project folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Student_Performance_Analysis.ipynb
```

Run cells top-to-bottom. Charts display inline; outputs are also saved to `output/`.

### Option B — Python Script

```bash
pip install -r requirements.txt

# Place StudentsPerformance.csv in the same folder, then:
python analysis.py

# Or pass a custom path:
python analysis.py path/to/StudentsPerformance.csv
```

---

## Key Findings (Preview)

| Finding | Detail |
|---------|--------|
| **At-risk students** | ~18% score below 50 in at least one subject |
| **Test prep boost** | Completing the course adds ~5–10 points per subject on average |
| **Parental education gap** | Master's-degree households outscore "some high school" by 15–20 pts |
| **Gender split** | Males lead in maths; females lead in reading & writing |
| **Score correlation** | r ≈ 0.80–0.95 across subjects — literacy strongly predicts maths |

---

## Recommendations

1. **Expand test preparation enrolment** — make it mandatory for at-risk students and subsidise access for lower-income families. Target ≥85% participation.
2. **Implement a tiered reading & writing support programme** — because high reading/writing correlates with better maths, a school-wide literacy initiative lifts all three subjects simultaneously.
3. **Launch a parental engagement initiative** — partner sessions with practical homework-support guides and assigned mentors for students flagged as at-risk.

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, cleaning, grouping |
| `numpy` | Numerical operations & trend-line fitting |
| `matplotlib` | Chart construction and export |
| `seaborn` | Statistical visualisations and theming |
| `Jupyter` | Interactive notebook environment |

---

## License

This project is produced for educational purposes as part of the **Pluto Academy Data Analytics Internship Program**.
