###### Script to run all the examples in the paper
###### Run from the repository root:
###### Rscript run_all.R

#########################

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("this.path", quietly = TRUE)) install.packages("this.path", repos = "https://cloud.r-project.org")
examples_path <- this.path::this.dir()
setwd(examples_path)

if (!("deepspat" %in% rownames(installed.packages()))) {
  install.packages("deepspat", type = "source")
} else if (packageVersion("deepspat") < package_version("0.3.5")) {
  message("The installed deepspat version is outdated; updating to version 0.3.5 or later.")
  install.packages("deepspat", type = "source")
}

envname <- file.path(examples_path, "deepspat_venv")
# Remark: On macOS, Python installation may fail because of the local Homebrew
# OpenSSL configuration. Run `brew update` and `brew upgrade openssl@3` (or
# `brew install openssl@3` if needed). This is a local system configuration
# issue, not part of the package implementation, so it is not handled here.
if (!dir.exists(envname)) source(file.path(examples_path, "reproduce_prepare.R"))

####### Use the Python environment created by reproduce_prepare.R #########
library(reticulate)
reticulate::use_virtualenv(file.path(examples_path, "deepspat_venv"), required = TRUE)
reticulate::py_config()

####### Check availability of required packages #########
source("Examples/check_packages.R")

####### Simulation in Section 3.3

#### Fit the models for simulated data from deepspat
source("Examples/sims_1.R")
#### Results for simulated data from deepspat
source("Examples/results_sims_1.R")

#### Fit the models for simulated data from cocons
source("Examples/sims_2.R")
#### Results for simulated data from cocons
source("Examples/results_sims_2.R")

#### Print Table 3 for direct comparison with the paper
sim_deepspat <- new.env()
sim_cocons <- new.env()

load("Examples/Data/sim_results_from_deepspat.RData",
     envir = sim_deepspat)
load("Examples/Data/sim_results_from_cocons.RData",
     envir = sim_cocons)

table_3 <- data.frame(
  Model = c(
    "Stationary GP model in gstat",
    "Nonstationary GP model in cocons",
    "Nonstationary NNGP model in deepspat",
    "Nonstationary FRK model in deepspat",
    "Nonstationary GP model in deepspat"
  ),
  `deepspat RMSPE` = c(
    sim_deepspat$rmspe_gstat,
    sim_deepspat$rmspe_cocons,
    sim_deepspat$rmspe_nngp,
    sim_deepspat$rmspe_frk,
    sim_deepspat$rmspe_gp
  ),
  `deepspat CRPS` = c(
    sim_deepspat$crps_gstat,
    sim_deepspat$crps_cocons,
    sim_deepspat$crps_nngp,
    sim_deepspat$crps_frk,
    sim_deepspat$crps_gp
  ),
  `cocons RMSPE` = c(
    sim_cocons$rmspe_gstat,
    sim_cocons$rmspe_cocons,
    sim_cocons$rmspe_nngp,
    sim_cocons$rmspe_frk,
    sim_cocons$rmspe_gp
  ),
  `cocons CRPS` = c(
    sim_cocons$crps_gstat,
    sim_cocons$crps_cocons,
    sim_cocons$crps_nngp,
    sim_cocons$crps_frk,
    sim_cocons$crps_gp
  ),
  check.names = FALSE
)

table_3[, -1] <- round(table_3[, -1], 3)
print(table_3, row.names = FALSE)



####### Application with mean temperature in Section 4.1

#### Fit the models for mean temperature
source("Examples/app_model_GP_ST.R")

#### Print Table 4 for direct comparison with the paper
table_4 <- data.frame(
  Model = c(
    "Stationary, intercept-only",
    "Stationary, with elevation",
    "Nonstationary, intercept-only",
    "Nonstationary, with elevation"
  ),
  RMSPE = c(RMSPE_d2, RMSPE_d1, RMSPE_d4, RMSPE_d3),
  CRPS = c(CRPS_d2, CRPS_d1, CRPS_d4, CRPS_d3)
)

table_4[, -1] <- round(table_4[, -1], 4)
print(table_4, row.names = FALSE)

#### Results for mean temperature
source("Examples/app_results_GP_ST.R")



####### Application with maximum temperature in Section 4.2

#### Fit the models for maximum temperature
source("Examples/app_model_MSP.R")
#### Results for maximum temperature
source("Examples/app_results_MSP.R")
