#' Prepare Seurat Object for cVAE
#'
#' Extracts gene expression data and metadata from a Seurat object
#' for use with the cVAE-BioBERT model.
#'
#' @param seuratObj A Seurat object with gene expression data.
#' @param assay Name of the assay to use. Default is "RNA".
#' @param slot Name of the slot to use. Default is "scale.data" to use Seurat's
#'   normalized and scaled data. Can also be "data" or "counts".
#' @param metadataColumns Character vector of metadata column names to use for
#'   conditioning. If NULL, uses all character/factor columns.
#' @param geneList Character vector of gene names to include. If NULL, uses
#'   variable features or all genes if no variable features are set.
#'
#' @return A list containing:
#'   \item{geneMatrix}{Gene expression matrix (cells x genes)}
#'   \item{metadataTexts}{Character vector of combined metadata strings}
#'   \item{geneNames}{Names of genes in the matrix}
#'   \item{cellNames}{Names of cells/samples}
#'   \item{isNormalized}{Logical indicating if data is already normalized (TRUE for scale.data)}
#' @export
#'
#' @details
#' When using slot = "scale.data" (default), the data is already normalized and
#' scaled by Seurat, so no additional normalization is applied by the Python model.
#' This avoids redundant normalization and preserves Seurat's preprocessing.
#'
#' @examples
#' \dontrun{
#' data <- PrepareSeuratForCvae(seuratObj, metadataColumns = c("celltype", "condition"))
#' }
PrepareSeuratForCvae <- function(seuratObj,
                                  assay = "RNA",
                                  slot = "scale.data",
                                  metadataColumns = NULL,
                                  geneList = NULL) {
  # Validate input
  if (!inherits(seuratObj, "Seurat")) {
    stop("seuratObj must be a Seurat object")
  }

  # Get gene expression matrix
  if (!assay %in% names(seuratObj@assays)) {
    stop(paste("Assay", assay, "not found in Seurat object"))
  }

  # Determine genes to use
  if (is.null(geneList)) {
    # Try to use variable features
    varFeatures <- Seurat::VariableFeatures(seuratObj)
    if (length(varFeatures) > 0) {
      geneList <- varFeatures
    } else {
      # Use all genes
      geneList <- rownames(seuratObj[[assay]])
    }
  }

  # Get expression data
  exprData <- SeuratObject::GetAssayData(seuratObj, assay = assay, slot = slot)
  exprData <- exprData[geneList, , drop = FALSE]

  # Transpose to samples x genes
  geneMatrix <- as.matrix(t(exprData))

  # Get metadata
  metadata <- seuratObj@meta.data

  # Determine metadata columns to use
  if (is.null(metadataColumns)) {
    # Use character and factor columns
    metadataColumns <- names(metadata)[sapply(metadata, function(x) {
      is.character(x) || is.factor(x)
    })]
  }

  # Validate columns exist
  missingCols <- setdiff(metadataColumns, names(metadata))
  if (length(missingCols) > 0) {
    stop(paste("Metadata columns not found:", paste(missingCols, collapse = ", ")))
  }

  # Build metadata text strings
  metadataTexts <- sapply(seq_len(nrow(metadata)), function(i) {
    parts <- lapply(metadataColumns, function(col) {
      val <- as.character(metadata[i, col])
      if (!is.na(val) && nchar(trimws(val)) > 0) {
        paste0(col, ": ", val)
      } else {
        NULL
      }
    })
    parts <- Filter(Negate(is.null), parts)
    if (length(parts) > 0) {
      paste(parts, collapse = " | ")
    } else {
      "unknown sample"
    }
  })

  list(
    geneMatrix = geneMatrix,
    metadataTexts = metadataTexts,
    geneNames = geneList,
    cellNames = colnames(seuratObj),
    isNormalized = (slot == "scale.data")  # Flag for Python to skip normalization
  )
}
