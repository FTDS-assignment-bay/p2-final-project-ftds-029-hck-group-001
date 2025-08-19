# Milestone 3

## Repository Outline
```
p2-ftds029-hck-m3-IanKuzuma
    |
    ├── .env --- dotenv file to set up all the environment variables for the Docker stack
    ├── description.md --- markdown file for this project's repository documentation
    ├── P2M3_rd_ladityarsa_ilyankusuma_conceptual.txt --- text file for answering the six conceptual questions
    ├── P2M3_rd_ladityarsa_ilyankusuma_config.yaml --- yaml file to set up a data engineering stack using Docker
    ├── P2M3_rd_ladityarsa_ilyankusuma_ddl.txt --- text file for the postgesql query documentation
    ├── P2M3_rd_ladityarsa_ilyankusuma_data_raw.csv --- comma seperated value file of our raw dataset
    ├── P2M3_rd_ladityarsa_ilyankusuma_data_clean.csv --- comma seperated value file of our cleaned dataset
    ├── P2M3_rd_ladityarsa_ilyankusuma_DAG.py --- python script for running our entire ETL data pipeline with DAG
    ├── P2M3_rd_ladityarsa_ilyankusuma_DAG_graph.jpg --- jpeg screenshot file of the DAG's successful graph sequence
    ├── P2M3_rd_ladityarsa_ilyankusuma_GX.ipynb --- jupyter notebook of our GX data validation project
    ├── P2M3_rd_ladityarsa_ilyankusuma_GX_docs.jpg --- jpeg screenshot file of the GX's local Data Docs
    ├── README.md --- markdown file for the problems and matrix of this project given by hacktiv8
    ├── /dags --- folder for Docker to read our DAG python script off of
    ├── /data --- folder for PostgreSQL and Docker to read and write our dataset files off of
    ├── /gx --- auto-generated folder for saving our GX's Data Context
    ├── /postgres_data --- auto-generated folder for our PostgreSQL's container
    ├── /images
        ├── introduction & objective.png --- png screenshot file of our Kibana dashboard's introduction
        └── plot & insight 01.png --- png screenshot file of our Kibana dashboard's first plot & insight
        └── plot & insight 02.png --- png screenshot file of our Kibana dashboard's second plot & insight
        └── plot & insight 03.png --- png screenshot file of our Kibana dashboard's third plot & insight
        └── plot & insight 04.png --- png screenshot file of our Kibana dashboard's fourth plot & insight
        └── plot & insight 05.png --- png screenshot file of our Kibana dashboard's fifth plot & insight
        └── plot & insight 06.png --- png screenshot file of our Kibana dashboard's sixth plot & insight
        └── conclusion.png --- png screenshot file of our Kibana dashboard's conclusion
```

## Problem Background

In any company, whether it's a hot tech startup or a legacy manufacturing giant, losing a good employee always stings. You're not just losing a headcount, you're losing experience, project momentum, team synergy, and probably spending a ton of time and money replacing them. And yet, most companies still only react *after* someone resigns or underperforms, instead of predicting *who* might be at risk and *why* it's happening in the first place.

This project aims to change that.
By analyzing structured HR data from a real organization, the goal is to explore what really drives attrition and performance disparities. Are certain job roles more prone to burnout? Does overtime hurt performance? Are promotions aligned with actual capability? These are the kinds of questions HR teams should be asking *before* attrition happens, not after.
The project isn't just about churn, it's about giving businesses a data-driven way to manage, support, and retain their talents better.

## Project Output

The final output of this project is a complete data analysis pipeline that turns raw HR data into business-ready insights. It includes:

* Exploratory Data Analysis (EDA) focused on attrition, performance ratings, promotions, and job roles.
* A series of visualizations (pie chart, bar plots, heatmap, gauge, etc.) that explore key trends across departments, genders, job levels, and work experience.
* A Kibana dashboard for interactive data storytelling and operational use.
* Business insights and recommendations based on analytical patterns.

The core value lies in surfacing *early indicators* of disengagement, mismatch, or structural bottlenecks in HR planning.

## Data

Dataset URL: [https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

The dataset contains HR records for 1,470 employees, including:

* Demographics: gender, education, age, marital status
* Work-related info: job role, department, overtime, years at company, performance rating
* Compensation: monthly income, stock options, percent salary hike
* Target variable: **Attrition** (Yes/No)

The dataset is clean, consistent, and feature-rich. It has a balanced mix of categorical and numerical data, which makes it perfect for slicing across segments like department vs performance, gender vs promotion, or job level vs attrition.

## Method

This is a **descriptive analytics project** to uncover patterns and trends in employee behavior, performance, and attrition. Through a combination of statistical summaries, visual breakdowns, and segmentation analysis, it extracts actionable insights from historical HR data.

Here’s what’s done:

* Data Cleaning: Standardized column names, parsed values, removed irrelevant fields (like employee number).
* Feature Engineering: Derived tenure brackets, grouped job levels, and normalized rating scales.
* Visualization Suite:

  * **Pie Chart** for high-level education distributuon.
  * **Bar Plots** for counting attrition by department and job role.
  * **Line Chart** for showing the trend of avg. training times by years at company.
  * **Heatmap** showing distribution of job levels across gender.
  * **Gauge Chart** for avg. years since last promotion across the company.

The visualizations were then translated into interactive components using **Kibana** on top of **ElasticSearch**, with all the pipeline managed using **Airflow** and validated using **Great Expectations**.

## Stacks

* **Languages**: Python, YAML
* **Libraries**: pandas, seaborn, matplotlib, plotly, numpy
* **Tools**: Apache Airflow, Docker Compose, PostgreSQL
* **Visualization**: Kibana, ElasticSearch
* **Data Validation**: Great Expectations
* **Notebook Environment**: JupyterLab

## Reference

* [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* [McKinsey: The Great Attrition](https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/great-attrition-or-great-attraction-why-people-are-leaving-jobs)
* [Gartner Workforce Analytics](https://www.gartner.com/en/human-resources/insights/workforce-analytics)
* [SHRM: Predictive Analytics in HR](https://www.shrm.org/hr-today/news/hr-magazine/summer2022/pages/how-predictive-analytics-can-help-hr.aspx)
* [Airflow Documentation](https://airflow.apache.org/docs/)
* [Kibana Documentation](https://www.elastic.co/guide/en/kibana/current/index.html)
* [Great Expectations Documentation](https://docs.greatexpectations.io/)

---

**Extra Reference:**
- [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [readme example](https://github.com/fahmimnalfrzki/Swift-XRT-Automation)
- [Another example](https://github.com/sanggusti/final_bangkit) (**Must read**)
- [Additional reference](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)