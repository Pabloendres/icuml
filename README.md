# Development of an Interpretable Machine Learning Model for Clinical Decision Support on Mortality Prediction of People Admitted to Intensive Care Units

{cite}

This repository is intented as a replication tool of our work, including instalation of data sources, cohort construction, and analysis process.

If you have difficulties to install or analyse data, please create an issue.

Also, the repository was written originally in Spanish and later translated into English. There may be files that have been omitted from the translation, so we ask you to raise an issue if you find untranslated files to fix it ASAP.

# Instalation steps

1. Navigate to /eicu/1-instalation.ipynb and follow the instructions to install the eICU-CRD database. Other notebooks in the folder are not required for the instalation.
2. Navigate to /mimic/1-instalation.ipynb and follow the instructions to install the MIMIC-III database. Other notebooks in the folder are not required for the instalation.
3. Navigate to /anaylysis/ and run the notebooks inside.

# Directory tree

```
.
├── analysis/                       // Principal project analysis
│   ├── data/                       // Data input/output of the analysis
│   ├── figures/                    // Plotting outputs
│   ├── saves/                      // Pickle saves of different analysis steps
│   └── 1-feature-analysis.ipynb    // Feature analysis of the cohorts
│   └── 2-strategies.ipynb          // Feature selection strategies
│   └── 3-interpretability.ipynb    // Construction of interpretability layer
│   └── aidxmods.py                 // Analysis functions and classes
│
├── eicu/                           // eICU-CRD instalation utilities
│   ├── data/                       // Folder reserved for eICU-CRD CSV files
│   ├── eicu-code/                  // eICU-CRD Code Repository used (https://github.com/MIT-LCP/eicu-code)
│   ├── sql/                        // SQL files for the eICU-CRD cohort construction
│   ├── 1-instalation.ipynb         // eICU-CRD Instalation guide
│   └── ...
│
└── mimic/                          // MIMIC-III instalation utilities
    ├── data/                       // Folder reserved for MIMIC-III CSV files
    ├── mimic-code/                 // MIMIC Code Repository used (https://github.com/MIT-LCP/mimic-code)
    ├── sql/                        // SQL files for the MIMIC-III cohort construction
    ├── 1-instalation.ipynb         // MIMIC-III Instalation guide
    └── ...

```
