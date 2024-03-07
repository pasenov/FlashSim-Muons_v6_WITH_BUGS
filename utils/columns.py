muon_cond = [
    "MGenMuon_eta",
    "MGenMuon_phi",
    "MGenMuon_pt",
    "MGenMuon_charge",
    "MGenMuon_statusFlag0",
    "MGenMuon_statusFlag1",
    "MGenMuon_statusFlag2",
    "MGenMuon_statusFlag3",
    "MGenMuon_statusFlag4",
    "MGenMuon_statusFlag5",
    "MGenMuon_statusFlag6",
    "MGenMuon_statusFlag7",
    "MGenMuon_statusFlag8",
    "MGenMuon_statusFlag9",
    "MGenMuon_statusFlag10",
    "MGenMuon_statusFlag11",
    "MGenMuon_statusFlag12",
    "MGenMuon_statusFlag13",
    "MGenMuon_statusFlag14",
    "ClosestJet_dr",
    "ClosestJet_dphi",
    "ClosestJet_deta",
    "ClosestJet_pt",
    "ClosestJet_mass",
    "ClosestJet_EncodedPartonFlavour_light",
    "ClosestJet_EncodedPartonFlavour_gluon",
    "ClosestJet_EncodedPartonFlavour_c",
    "ClosestJet_EncodedPartonFlavour_b",
    "ClosestJet_EncodedPartonFlavour_undefined",
    "ClosestJet_EncodedHadronFlavour_b",
    "ClosestJet_EncodedHadronFlavour_c",
    "ClosestJet_EncodedHadronFlavour_light"
]

eff_muon = [var.replace("C", "GC", 1) for var in muon_cond] + [
    "GenMuon_isReco"
]  # for efficiency

gen_muon = [var.replace("G", "MG", 1) for var in muon_cond]  # for flow



muon_names = [
    "convVeto",
    "deltaEtaSC",
    "dr03EcalRecHitSumEt",
    "dr03HcalDepth1TowerSumEt",
    "dr03TkSumPt",
    "dr03TkSumPtHEEP",
    "dxy",
    "dxyErr",
    "dz",
    "dzErr",
    "eInvMinusPInv",
    "energyErr",
    "eta",
    "hoe",
    "ip3d",
    "isPFcand",
    "jetPtRelv2",
    "jetRelIso",
    "lostHits",
    "miniPFRelIso_all",
    "miniPFRelIso_chg",
    "mvaFall17V2Iso",
    "mvaFall17V2Iso_WP80",
    "mvaFall17V2Iso_WP90",
    "mvaFall17V2Iso_WPL",
    "mvaFall17V2noIso",
    "mvaFall17V2noIso_WP80",
    "mvaFall17V2noIso_WP90",
    "mvaFall17V2noIso_WPL",
    "mvaTTH",
    "pfRelIso03_all",
    "pfRelIso03_chg",
    "phi",
    "pt",
    "r9",
    "seedGain",
    "sieie",
    "sip3d",
    "tightCharge",
    "charge"
]
