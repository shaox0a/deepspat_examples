################################################################################
# Create and configure the project-local Python environment.
# `deepspat_venv` is created alongside `reproduce_prepare_pyenv.R`.
# If prompted to install packages from source, select "no" to use binary packages.
options(repos = c(CRAN = "https://cloud.r-project.org"))

library(reticulate)

project_path <- this.path::this.dir()
py_version <- "3.12:latest"
envname <- file.path(project_path, "deepspat_venv")
path_to_python <- reticulate::virtualenv_starter("3.12")
if (is.null(path_to_python)) {
  path_to_python <- reticulate::install_python(version = py_version)
}

reticulate::virtualenv_create(
  envname = envname,
  python = path_to_python,
  version = py_version
)

reticulate::use_virtualenv(envname, required = TRUE)

reticulate::virtualenv_install(
  envname = envname,
  packages = c(
    "tensorflow==2.18.0",
    "tensorflow-probability==0.25.0",
    "tf-keras==2.18.0",
    "scipy"
  )
)

# The Python path shown below should point to `deepspat_venv`.
reticulate::py_config()
