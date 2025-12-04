###### Script to run all the examples in the paper

#########################

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



####### Application with mean temperature in Section 4.1

#### Fit the models for mean temperature
source("Examples/app_model_GP_ST.R")
#### Results for mean temperature
source("Examples/app_results_GP_ST.R")



####### Application with maximum temperature in Section 4.2

#### Fit the models for maximum temperature
source("Examples/app_model_MSP.R")
#### Results for maximum temperature
source("Examples/app_results_MSP.R")