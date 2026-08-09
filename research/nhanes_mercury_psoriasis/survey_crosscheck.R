#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: survey_crosscheck.R <enriched_analysis_data.tsv> <output.csv>")
}
input_path <- args[[1]]
output_path <- args[[2]]

suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "fail")

d <- read.delim(input_path, check.names = FALSE, stringsAsFactors = FALSE)
names(d)[names(d) == "GENDER"] <- "sex_code"
names(d)[names(d) == "AGE"] <- "age"
names(d)[names(d) == "RACE"] <- "race_code"
names(d)[names(d) == "EDUCATION"] <- "education_code"
names(d)[names(d) == "PIR"] <- "pir"
names(d)[names(d) == "BMI"] <- "bmi"
names(d)[names(d) == "HAD.AT.LEAST.12.ALCOHOL.DRINKS.1.YR."] <- "alcohol_code"
names(d)[names(d) == "BLOOD.CADMIUM..UG.L."] <- "cadmium"
names(d)[names(d) == "BLOOD.LEAD..UG.DL."] <- "lead"
names(d)[names(d) == "BLOOD.MERCURY..TOTAL..UG.L."] <- "mercury"
names(d)[names(d) == "PSORIASIS"] <- "psoriasis"
names(d)[names(d) == "SMOKED.AT.LEAST.100.CIGARETTES.IN.LIFE"] <- "smoking_code"

d$sex <- factor(d$sex_code, levels = c(1, 2), labels = c("Male", "Female"))
d$race <- factor(
  d$race_code,
  levels = c(1, 2, 3, 4, 5),
  labels = c("Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Other/multiracial")
)
d$education <- d$education_code
d$education[!(d$education %in% 1:5)] <- NA
d$education <- factor(d$education)
d$smoking <- d$smoking_code
d$smoking[!(d$smoking %in% c(1, 2))] <- NA
d$smoking <- factor(d$smoking)
d$alcohol <- d$alcohol_code
d$alcohol[!(d$alcohol %in% c(1, 2))] <- NA
d$alcohol <- factor(d$alcohol)
d$cycle <- factor(d$cycle)
d$log2_mercury <- log2(d$mercury)

design <- svydesign(
  ids = ~combined_psu,
  strata = ~combined_stratum,
  weights = ~correct_component_weight_4yr,
  nest = TRUE,
  data = d
)

fits <- list(
  S1_crude_survey_raw = svyglm(psoriasis ~ mercury, design = design, family = quasibinomial()),
  S2_model2_survey_raw = svyglm(psoriasis ~ mercury + age + sex + race + cycle, design = design, family = quasibinomial()),
  S3_model2_survey_log2 = svyglm(psoriasis ~ log2_mercury + age + sex + race + cycle, design = design, family = quasibinomial())
)

primary_design <- subset(
  design,
  !is.na(log2_mercury) & !is.na(age) & !is.na(sex) & !is.na(race) &
  !is.na(education) & !is.na(pir) & !is.na(bmi) & !is.na(smoking) &
  !is.na(alcohol) & !is.na(cadmium) & !is.na(lead) & !is.na(cycle)
)
fits$S4_parsimonious_cc_survey_log2 <- svyglm(
  psoriasis ~ log2_mercury + age + sex + race + education + pir + bmi + smoking + alcohol + cadmium + lead + cycle,
  design = primary_design,
  family = quasibinomial()
)

rows <- list()
for (model_id in names(fits)) {
  fit <- fits[[model_id]]
  co <- coef(fit)
  se <- sqrt(diag(vcov(fit)))
  rows[[model_id]] <- data.frame(
    model_id = model_id,
    term = names(co),
    beta_r_survey = as.numeric(co),
    se_r_survey = as.numeric(se),
    n = nrow(model.frame(fit)),
    design_df = degf(fit$survey.design),
    stringsAsFactors = FALSE
  )
}
out <- do.call(rbind, rows)
write.csv(out, output_path, row.names = FALSE)
