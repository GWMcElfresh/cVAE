#' Run cVAE-BioBART on Seurat Object
#'
#' Runs the conditional VAE with BioBART conditioning on a Seurat object.
#' This function extracts gene expression and metadata, trains (or uses
#' a pre-trained) cVAE model, and adds the results back to the Seurat object.
#'
#' @param seuratObj A Seurat object to process.
#' @param assay Name of the assay to use. Default is "RNA".
#' @param slot Slot of the assay to use. Default is "scale.data" to use Seurat's
#'   normalized and scaled data. Can also be "data" or "counts".
#' @param metadataColumns Character vector of metadata column names for
#'   BioBART conditioning. If NULL, uses all character/factor columns.
#' @param geneList Character vector of genes to use. If NULL, uses variable
#'   features or all genes.
#' @param latentDim Dimension of the VAE latent space. Default is 32.
#' @param hiddenDims Numeric vector of hidden layer dimensions.
#' @param epochs Number of training epochs. Default is 100.
#' @param batchSize Training batch size. Default is 32.
#' @param learningRate Learning rate. Default is 0.001.
#' @param reductionName Name for the reduction in the Seurat object.
#'   Default is "cvae".
#' @param device Device for computation ("cuda", "cpu", or NULL for auto).
#' @param verbose Whether to show training progress. Default is TRUE.
#' @param returnModel Whether to return the trained model. Default is FALSE.
#'
#' @return The Seurat object with added cVAE reduction and metadata.
#'   If returnModel is TRUE, returns a list with both the Seurat object
#'   and the trained model.
#' @export
#'
#' @details
#' By default, this function uses Seurat's normalized layer (scale.data) which
#' is already normalized and scaled. This avoids redundant normalization and
#' respects Seurat's preprocessing pipeline. If using a different slot like
#' "data" or "counts", the data will be passed as-is to the Python model.
#'
#' @examples
#' \dontrun{
#' seuratObj <- RunCvaeBiobert(seuratObj,
#'                              metadataColumns = c("celltype", "condition"),
#'                              latentDim = 32,
#'                              epochs = 100)
#' }
RunCvaeBiobert <- function(seuratObj,
                            assay = "RNA",
                            slot = "scale.data",
                            metadataColumns = NULL,
                            geneList = NULL,
                            latentDim = 32L,
                            hiddenDims = c(512L, 256L, 128L),
                            epochs = 100L,
                            batchSize = 32L,
                            learningRate = 0.001,
                            reductionName = "cvae",
                            device = NULL,
                            verbose = TRUE,
                            returnModel = FALSE) {
  # Check Python environment
  if (!exists("cvaeModule", envir = .pkgEnv)) {
    message("Python environment not initialized. Calling InitPythonEnv()...")
    InitPythonEnv()
  }

  cvaeModule <- get("cvaeModule", envir = .pkgEnv)

  # Prepare data
  preparedData <- PrepareSeuratForCvae(
    seuratObj = seuratObj,
    assay = assay,
    slot = slot,
    metadataColumns = metadataColumns,
    geneList = geneList
  )

  # Convert to numpy arrays
  geneMatrix <- reticulate::np_array(preparedData$geneMatrix)
  geneCount <- ncol(preparedData$geneMatrix)

  # Create model
  model <- cvaeModule$CvaeBiobert(
    geneCount = as.integer(geneCount),
    latentDim = as.integer(latentDim),
    hiddenDims = as.integer(hiddenDims),
    device = device
  )

  # Train model
  trainingResult <- model$Fit(
    geneMatrix = geneMatrix,
    metadataTexts = preparedData$metadataTexts,
    epochs = as.integer(epochs),
    batchSize = as.integer(batchSize),
    learningRate = learningRate,
    verbose = verbose
  )

  # Get embeddings
  embeddings <- model$Transform(
    geneMatrix = geneMatrix,
    metadataTexts = preparedData$metadataTexts
  )

  # Convert embeddings to R matrix
  embeddings <- as.matrix(reticulate::py_to_r(embeddings))
  rownames(embeddings) <- preparedData$cellNames
  colnames(embeddings) <- paste0(reductionName, "_", seq_len(ncol(embeddings)))

  # Create dimension reduction object
  cvaeReduction <- Seurat::CreateDimReducObject(
    embeddings = embeddings,
    key = paste0(reductionName, "_"),
    assay = assay
  )

  # Add to Seurat object
  seuratObj[[reductionName]] <- cvaeReduction

  # Add training metadata
  trainingHistory <- reticulate::py_to_r(trainingResult)
  seuratObj@misc[[paste0(reductionName, "_training")]] <- list(
    finalTrainLoss = trainingHistory$finalTrainLoss,
    finalValLoss = trainingHistory$finalValLoss,
    epochs = epochs,
    latentDim = latentDim,
    geneCount = geneCount,
    nSamples = nrow(preparedData$geneMatrix),
    metadataColumns = metadataColumns
  )

  if (returnModel) {
    return(list(
      seuratObj = seuratObj,
      model = model
    ))
  }

  return(seuratObj)
}

#' Get cVAE Embeddings from Seurat Object
#'
#' Retrieves the cVAE latent space embeddings from a Seurat object.
#'
#' @param seuratObj A Seurat object with cVAE reduction.
#' @param reductionName Name of the cVAE reduction. Default is "cvae".
#'
#' @return Matrix of cVAE embeddings (cells x latent dimensions).
#' @export
#'
#' @examples
#' \dontrun{
#' embeddings <- GetCvaeEmbeddings(seuratObj)
#' }
GetCvaeEmbeddings <- function(seuratObj, reductionName = "cvae") {
  if (!reductionName %in% names(seuratObj@reductions)) {
    stop(paste("Reduction", reductionName, "not found. Run RunCvaeBiobart first."))
  }

  return(Seurat::Embeddings(seuratObj, reduction = reductionName))
}

#' Get cVAE Training Metadata
#'
#' Retrieves training metadata from a Seurat object that has been
#' processed with RunCvaeBiobart.
#'
#' @param seuratObj A Seurat object with cVAE reduction.
#' @param reductionName Name of the cVAE reduction. Default is "cvae".
#'
#' @return List containing training metadata.
#' @export
#'
#' @examples
#' \dontrun{
#' metadata <- GetCvaeMetadata(seuratObj)
#' }
GetCvaeMetadata <- function(seuratObj, reductionName = "cvae") {
  metaKey <- paste0(reductionName, "_training")
  if (!metaKey %in% names(seuratObj@misc)) {
    stop(paste("cVAE training metadata not found. Run RunCvaeBiobart first."))
  }

  return(seuratObj@misc[[metaKey]])
}
