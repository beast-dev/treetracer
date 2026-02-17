#!/usr/bin/env Rscript
# Compute RF distances on a .trees file and output the upper-triangle values.
#
# Usage:
#   Rscript bench_rf_R.R <trees_file> <output_file>
#
# Reads ALL trees from the file, computes RF.dist, and writes:
#   - Line 1: elapsed time in seconds
#   - Lines 2+: upper-triangle RF distances, one per line, row-major order
#     (i.e. d[1,2], d[1,3], ..., d[1,n], d[2,3], ..., d[n-1,n])
#
# Progress is printed to stderr.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  cat("Usage: Rscript bench_rf_R.R <trees_file> <output_file>\n",
      file = stderr())
  quit(status = 1)
}

trees_file <- args[1]
output_file <- args[2]

suppressPackageStartupMessages({
  library(ape)
  library(phangorn)
})

# --- Load ---
cat(sprintf("Reading %s ...\n", trees_file), file = stderr())
t0 <- proc.time()
trees <- read.nexus(trees_file)
load_time <- (proc.time() - t0)[3]
n <- length(trees)
n_tips <- length(trees[[1]]$tip.label)
cat(sprintf("Read %d trees (%d tips) in %.2fs\n", n, n_tips, load_time),
    file = stderr())

# --- Compute RF ---
cat(sprintf("Computing RF.dist on %d trees ...\n", n), file = stderr())
t0 <- proc.time()
d <- RF.dist(trees)
elapsed <- (proc.time() - t0)[3]
cat(sprintf("Done in %.3fs\n", elapsed), file = stderr())

# --- Write output ---
# Upper triangle in row-major order: (1,2), (1,3), ..., (1,n), (2,3), ...
m <- as.matrix(d)
idx <- which(upper.tri(m), arr.ind = TRUE)
idx <- idx[order(idx[, 1], idx[, 2]), ]
vals <- m[idx]

con <- file(output_file, "w")
writeLines(sprintf("%.6f", elapsed), con)
writeLines(as.character(as.integer(vals)), con)
close(con)

cat(sprintf("Wrote %d distances to %s\n", length(vals), output_file),
    file = stderr())
