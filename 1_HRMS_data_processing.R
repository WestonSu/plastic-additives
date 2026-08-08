# Script generated on Tue Dec 12 21:34:26 2025
# Edited by Wenyuan Su

library(patRoon)

# -------------------------
# initialization
# -------------------------
workPath <- "E:/patRoon/PA"
setwd(workPath)

# Set paths for external software
options(patRoon.path.pwiz = "C:/ProteoWizard")                  # ProteoWizard installation folder
options(patRoon.path.SIRIUS = "D:/Program Files/sirius")       # Optional; only required for SIRIUS
patRoon::verifyDependencies()

# Generate analysis information for raw MS files
anaInfo <- generateAnalysisInfo(paths = "E:/patRoon/PA/analyses/raw")

# Convert Thermo raw files to centroided mzML files; set FALSE to skip this step
doDataPretreatment <- TRUE
if (doDataPretreatment) {
  convertMSFiles(anaInfo = anaInfo, outPath = "analyses/mzml", from = "thermo",
                 to = "mzML", algorithm = "pwiz", centroid = "vendor")
}

# Generate analysis information for the demo mzML dataset
anaInfo <- generateAnalysisInfo(paths = "E:/patRoon/PA/analyses/mzml",
                                groups = c("Pooled_sample", "BK-STD"),
                                blanks = c("BK-STD", "BK-STD"),
                                norm_concs = c(NA, NA))

# -------------------------
# feature detection
# -------------------------
# Detect chromatographic features using OpenMS
fList <- findFeatures(anaInfo, "openms", noiseThrInt = 10000, chromSNR = 3,
                      chromFWHM = 5, mzPPM = 5, minFWHM = 1, maxFWHM = 30)

# Group and align features across analyses
fGroups <- groupFeatures(fList, "openms", rtalign = TRUE)

# Rule-based filtering of low-intensity, poorly reproducible and blank-associated features
fGroupsfil <- filter(fGroups, preAbsMinIntensity = 100, absMinIntensity = 10000,
                     relMinReplicateAbundance = 1, maxReplicateIntRSD = 0.75,
                     blankThreshold = 3, removeBlanks = TRUE, retentionRange = NULL,
                     mzRange = c(100, 800))

# -------------------------
# componentization
# -------------------------
# Group isotopes and adducts belonging to the same compound
components <- generateComponents(fGroupsfil, "openms", ionization = "positive")

# Select preferred molecular ions and monoisotopic peaks
# prefAdduct can be changed according to ionization mode, e.g., [M]+, [M+H]+ or [M-H]-
fGroupsSel <- selectIons(fGroupsfil, components, prefAdduct = "[M+H]+", onlyMonoIso = TRUE)

# -------------------------
# suspect screening
# -------------------------
# This section can be skipped for a fully nontarget workflow
suspList <- read.csv("C:/Users/SWY/Desktop/suspectlist.csv", stringsAsFactors = FALSE)
fGroupsSusp <- screenSuspects(fGroupsSel, suspList, onlyHits = TRUE)

# -------------------------
# transformation products
# -------------------------
# Examples of different approaches for TP generation; select according to the study purpose
TPs_CTS <- generateTPsCTS(suspList, "combined_photolysis_abiotic_hydrolysis", generations = 2)
TPs_library <- generateTPs("library", parents = suspList)
TPs_biotransformer <- generateTPs("biotransformer", type = "env", generations = 2, parents = suspList)

# Example: export transformation products generated from the internal library
tp_edges <- as.data.table(TPs_library)
data.table::fwrite(tp_edges, file = "library.tsv", sep = "\t")

# -------------------------
# MS/MS peak lists
# -------------------------
# Generate averaged MS peak lists; precursorMzWindow = NULL can be used for DIA data
avgMSListParams <- getDefAvgPListParams(clusterMzWindow = 0.005)
mslists <- generateMSPeakLists(fGroupsSusp, "mzr", maxMSRtWindow = 12,
                               precursorMzWindow = NULL, avgFeatParams = avgMSListParams,
                               avgFGroupParams = avgMSListParams)

# Remove weak MS/MS peaks and retain the 30 most intense fragments
mslists <- filter(mslists, absMSIntThr = NULL, absMSMSIntThr = NULL,
                  relMSIntThr = NULL, withMSMS = TRUE, relMSMSIntThr = 0.02,
                  topMSPeaks = NULL, topMSMSPeaks = 30)

# -------------------------
# diagnostic MS/MS filtering
# -------------------------
# Example for organophosphorus compounds: filter MS/MS spectra using characteristic
# fragment ions and neutral losses. The lists can be replaced by class-specific values
# derived from literature or reference standards.

mz_tolerance <- 0.005

# Example characteristic fragment ions for organophosphorus compounds
characteristic_fragments <- c(98.9842, 251.0468, 265.0624, 327.0781)

# Example neutral losses: H2O, C2H4, C3H6 and C4H8
neutral_losses <- c(18.0106, 28.0313, 42.0470, 56.0626)

# Check whether a characteristic fragment is present within the specified mass tolerance
equalMZ <- function(target_mz, observed_mz) any(abs(observed_mz - target_mz) < mz_tolerance)

# Identify different types of neutral losses present between pairs of MS/MS ions
findNeutralLosses <- function(observed_mz) {
  if (length(observed_mz) < 2) return(rep(FALSE, length(neutral_losses)))
  mass_diffs <- abs(combn(observed_mz, 2, FUN = function(x) x[1] - x[2]))
  sapply(neutral_losses, function(loss) any(abs(mass_diffs - loss) < mz_tolerance))
}

# Retain MS/MS spectra containing at least two diagnostic features
# Diagnostic features include characteristic fragment ions and characteristic neutral losses
min_diagnostic_features <- 2

mslistsF <- delete(mslists, j = function(pl, grp, ana, type) {
  if (type != "MSMS") return(integer(0))
  num_characteristic_fragments <- sum(sapply(characteristic_fragments, equalMZ, observed_mz = pl$mz))
  num_neutral_losses <- sum(findNeutralLosses(pl$mz))
  num_diagnostic_features <- num_characteristic_fragments + num_neutral_losses
  
  # delete the complete MS/MS peak list if fewer than the required diagnostic features are detected
  if (num_diagnostic_features < min_diagnostic_features) return(seq_len(nrow(pl)))
  return(integer(0))
})

# -------------------------
# formula and structural annotation
# -------------------------
# Generate molecular formula candidates using GenForm
formulas <- generateFormulas(fGroupsSusp, mslistsF, "genform", relMzDev = 10,
                             MSMode = "msms", batchSize = 64, maxCandidates = 1000,
                             elements = "C[100]H[200]O[10]N[10]P[3]S[3]F[10]Cl[10]Br[10]",
                             oc = TRUE, calculateFeatures = FALSE, timeout = 60)

# Generate structural candidates using MetFrag and PubChemLite
compounds <- generateCompounds(fGroupsSusp, mslistsF, "metfrag", method = "CL",
                               dbRelMzDev = 10, fragRelMzDev = 10, fragAbsMzDev = 0.002,
                               database = "pubchemlite", maxCandidatesToStop = 100)

# Optional SIRIUS annotation
# compounds <- generateCompounds(fGroupsSusp, mslistsF, "sirius", relMzDev = 5,
#                                fingerIDDatabase = "pubchem", elements = "CHNOP",
#                                profile = "orbitrap", token = tokenFID)

# Combine suspect matches, formula candidates and structural annotations
fGroupsa <- annotateSuspects(fGroupsSusp, formulas = formulas, compounds = compounds,
                             MSPeakLists = mslistsF)

# Incorporate molecular formula scores into compound candidate ranking
compounds <- addFormulaScoring(compounds, formulas, updateScore = TRUE)

# -------------------------
# reporting
# -------------------------
# Advanced report settings can be modified in report.yml
report(fGroupsa, MSPeakLists = mslistsF, formulas = formulas, compounds = compounds,
       components = NULL, TPs = NULL, parallel = TRUE, openReport = FALSE)