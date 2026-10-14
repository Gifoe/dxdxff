# Fixed label-free patient-location offsets

Twobases×four globaloffsetcoefficients×fivefolds:40actual calibrationcells. Patientmedianlogit usesonlyunlabelledscores; sourcecutoffsfitownvalidationgroupswithFITcoverage. No trueEZcount or per-patientlabel/oracle cutoff. Inverse-spaceprediction equality andranking invariance are tested/audited. Allmapsremainprivate.

BestfittedDevelopmentMacro-F1: **0.693045386132**, `M18_TIME_CONTROL__SOURCE_OFFSET_0.25`, below0.70. Independentverification covers40map cells/520patient metrics and10000IDbootstrap. Allconfigs are retained andnoOuter isreadinthisphase. These are repeatedly fittedvalidationresults; see rootfinalreport foractualOuterfailure.
