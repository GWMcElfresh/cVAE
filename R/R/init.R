#' Initialize Python Environment for cVAE-BioBART
#'
#' Sets up the Python environment and imports the required Python package.
#' This function must be called before using other cVAE functions.
#'
#' @param pythonPath Path to Python executable. If NULL, uses reticulate defaults.
#' @param condaEnv Name of conda environment to use. If NULL, doesn't use conda.
#' @param installPackage Whether to install the cvae_biobart Python package if not found.
#'
#' @return Invisibly returns the Python module object.
#' @export
#'
#' @examples
#' \dontrun{
#' InitPythonEnv()
#' }
InitPythonEnv <- function(pythonPath = NULL,
                          condaEnv = NULL,
                          installPackage = TRUE) {
  # Set Python path if provided
  if (!is.null(pythonPath)) {
    reticulate::use_python(pythonPath, required = TRUE)
  } else if (!is.null(condaEnv)) {
    reticulate::use_condaenv(condaEnv, required = TRUE)
  }

  # Try to import the package
  tryCatch({
    cvaeModule <- reticulate::import("cvae_biobart")
    message("cvae_biobart Python package loaded successfully.")
    assign("cvaeModule", cvaeModule, envir = .pkgEnv)
    invisible(cvaeModule)
  }, error = function(e) {
    if (installPackage) {
      message("cvae_biobart not found. Attempting to install...")
      pkgPath <- system.file("python", package = "cvaeBiobart")
      if (pkgPath == "") {
        stop("Could not find Python package path. Please install manually.")
      }
      reticulate::py_install(pkgPath, pip = TRUE)
      cvaeModule <- reticulate::import("cvae_biobart")
      assign("cvaeModule", cvaeModule, envir = .pkgEnv)
      invisible(cvaeModule)
    } else {
      stop("cvae_biobart Python package not found. ",
           "Please install it or set installPackage = TRUE")
    }
  })
}

# Package environment for storing Python module reference
.pkgEnv <- new.env(parent = emptyenv())
