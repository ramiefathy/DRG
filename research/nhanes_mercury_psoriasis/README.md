# NHANES mercury–psoriasis replication and robustness audit

This isolated research workspace evaluates the 2024 PLOS ONE analysis of blood total mercury and self-reported psoriasis using the 2005–2006 and 2013–2014 NHANES cycles.

The first workflow inventories the required public NHANES source tables, verifies cycle availability and variable names, and records source-file checksums. A subsequent locked analysis will reproduce the reported ordinary logistic models and compare them with survey-design-correct estimates and prespecified robustness analyses.

Data source: CDC NHANES public-use data, accessed through the `nhanesdata` harmonized public mirror at `https://nhanes.kylegrealis.com/`. No participant-level data are committed to the repository; transient source files and derived results are retained only as workflow artifacts.
