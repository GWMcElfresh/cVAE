# Test PrepareSeuratForCvae function
test_that("PrepareSeuratForCvae validates input", {
  # Test with non-Seurat object
  expect_error(
    PrepareSeuratForCvae("not a seurat object"),
    "must be a Seurat object"
  )
})

# Note: Full integration tests require Seurat and Python environment
# These are placeholder tests that validate the function structure
test_that("GetCvaeEmbeddings requires reduction", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")

  # Create minimal Seurat object for testing
  mat <- matrix(rnorm(200), nrow = 10, ncol = 20)
  rownames(mat) <- paste0("Gene", 1:10)
  colnames(mat) <- paste0("Cell", 1:20)

  seuratObj <- SeuratObject::CreateSeuratObject(counts = mat)

  # Should error because no cvae reduction exists
  expect_error(
    GetCvaeEmbeddings(seuratObj),
    "not found"
  )
})

test_that("GetCvaeMetadata requires training metadata", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")

  mat <- matrix(rnorm(200), nrow = 10, ncol = 20)
  rownames(mat) <- paste0("Gene", 1:10)
  colnames(mat) <- paste0("Cell", 1:20)

  seuratObj <- SeuratObject::CreateSeuratObject(counts = mat)

  expect_error(
    GetCvaeMetadata(seuratObj),
    "not found"
  )
})
