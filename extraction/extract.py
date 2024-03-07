import os
import json
import ROOT

ROOT.gInterpreter.ProcessLine('#include "extraction.h"')


fileName1 = "0307C1DA-E49C-AB4B-9179-C70BE232321E.root"
fileName2 = "048D5B7E-A63F-554D-8386-4E8BFA761E62.root"
fileName3 = "23F225E4-7763-4146-A6AC-315AEC94DD6C.root"
fileName4 = "339CC18D-8C9E-E04F-8EC3-13D2F6CAB8D0.root"
fileName5 = "3DD8D5A5-F4DC-8042-BC91-28464A4B7047.root"
fileName6 = "50FC7DEB-7973-BC49-8750-2C6EC1DAD0FD.root"


treeName = "Events" 

d = ROOT.RDataFrame(treeName, {fileName1, fileName2, fileName3, fileName4, fileName5, fileName6})

# jet cleaning
d_cleaned = d.Define("TMPGenElectronMask", "abs(GenPart_pdgId) == 11") \
    .Define("TMPGenElectron_pt", "GenPart_pt[TMPGenElectronMask]") \
    .Define("TMPGenElectron_eta", "GenPart_eta[TMPGenElectronMask]") \
    .Define("TMPGenElectron_phi", "GenPart_phi[TMPGenElectronMask]") \
    .Define("TMPGenMuonMask", "abs(GenPart_pdgId) == 13") \
    .Define("TMPGenMuon_pt", "GenPart_pt[TMPGenMuonMask]") \
    .Define("TMPGenMuon_eta", "GenPart_eta[TMPGenMuonMask]") \
    .Define("TMPGenMuon_phi", "GenPart_phi[TMPGenMuonMask]") \
    .Define("CleanGenJet_mask_ele", "muon_clean_genjet_mask(GenJet_pt, GenJet_eta, GenJet_phi, TMPGenElectron_pt, TMPGenElectron_eta, TMPGenElectron_phi)") \
    .Define("CleanGenJet_mask_muon", "muon_clean_genjet_mask(GenJet_pt, GenJet_eta, GenJet_phi, TMPGenMuon_pt, TMPGenMuon_eta, TMPGenMuon_phi)") \
    .Define("CleanGenJetMask", "CleanGenJet_mask_ele && CleanGenJet_mask_muon") \
    .Define("CleanGenJet_pt", "GenJet_pt[CleanGenJetMask]") \
    .Define("CleanGenJet_eta", "GenJet_eta[CleanGenJetMask]") \
    .Define("CleanGenJet_phi", "GenJet_phi[CleanGenJetMask]") \
    .Define("CleanGenJet_mass", "GenJet_mass[CleanGenJetMask]") \
    .Define("CleanGenJet_hadronFlavour_uchar", "GenJet_hadronFlavour[CleanGenJetMask]") \
    .Define("CleanGenJet_hadronFlavour", "static_cast<ROOT::VecOps::RVec<int>>(CleanGenJet_hadronFlavour_uchar)") \
    .Define("CleanGenJet_partonFlavour", "GenJet_partonFlavour[CleanGenJetMask]")

# check if Muon_genPartIdx >= 0
d_check = d_cleaned.Define("GenPart_MuonIdx_check", "GenPart_MuonIdx(GenPart_pt, Muon_genPartIdx)")

    # .Define("GenPart_MuonIdx_totNum_check", "GenPart_MuonIdx_totNum(GenPart_pt, Muon_genPartIdx)")

d_extracted = d_check.Define("GenMuonMask", "(GenPart_pdgId == 13 | GenPart_pdgId == -13) && ((GenPart_statusFlags & 8192) > 0)") \
    .Define("GenMuon_isReco", "GenPart_MuonIdx_check[GenMuonMask]") \
    .Define("MGenMuon_mass", "GenPart_mass[GenMuonMask]") \
    .Define("MGenMuon_eta", "GenPart_eta[GenMuonMask]") \
    .Define("MGenMuon_pdgId", "GenPart_pdgId[GenMuonMask]") \
    .Define("MGenMuon_charge", "Mcharge(MGenMuon_pdgId)") \
    .Define("MGenMuon_phi", "GenPart_phi[GenMuonMask]") \
    .Define("MGenMuon_pt", "GenPart_pt[GenMuonMask]") \
    .Define("MGenMuon_statusFlags", "GenPart_statusFlags[GenMuonMask]") \
    .Define("MGenMuon_statusFlag0", "MBitwiseDecoder(MGenMuon_statusFlags, 0)") \
    .Define("MGenMuon_statusFlag1", "MBitwiseDecoder(MGenMuon_statusFlags, 1)") \
    .Define("MGenMuon_statusFlag2", "MBitwiseDecoder(MGenMuon_statusFlags, 2)") \
    .Define("MGenMuon_statusFlag3", "MBitwiseDecoder(MGenMuon_statusFlags, 3)") \
    .Define("MGenMuon_statusFlag4", "MBitwiseDecoder(MGenMuon_statusFlags, 4)") \
    .Define("MGenMuon_statusFlag5", "MBitwiseDecoder(MGenMuon_statusFlags, 5)") \
    .Define("MGenMuon_statusFlag6", "MBitwiseDecoder(MGenMuon_statusFlags, 6)") \
    .Define("MGenMuon_statusFlag7", "MBitwiseDecoder(MGenMuon_statusFlags, 7)") \
    .Define("MGenMuon_statusFlag8", "MBitwiseDecoder(MGenMuon_statusFlags, 8)") \
    .Define("MGenMuon_statusFlag9", "MBitwiseDecoder(MGenMuon_statusFlags, 9)") \
    .Define("MGenMuon_statusFlag10", "MBitwiseDecoder(MGenMuon_statusFlags, 10)") \
    .Define("MGenMuon_statusFlag11", "MBitwiseDecoder(MGenMuon_statusFlags, 11)") \
    .Define("MGenMuon_statusFlag12", "MBitwiseDecoder(MGenMuon_statusFlags, 12)") \
    .Define("MGenMuon_statusFlag13", "MBitwiseDecoder(MGenMuon_statusFlags, 13)") \
    .Define("MGenMuon_statusFlag14", "MBitwiseDecoder(MGenMuon_statusFlags, 14)") \
    .Define("GClosestJet_dr", "Mclosest_jet_dr(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi)") \
    .Define("GClosestJet_deta", "Mclosest_jet_deta(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi)") \
    .Define("GClosestJet_dphi", "Mclosest_jet_dphi(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi)") \
    .Define("GClosestJet_pt", "Mclosest_jet_pt(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_pt)") \
    .Define("GClosestJet_mass", "Mclosest_jet_mass(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_mass)") \
    .Define("GClosestJet_EncodedPartonFlavour_light", \
        "muon_closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{1,2,3})") \
    .Define("GClosestJet_EncodedPartonFlavour_gluon", \
        "muon_closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{21})") \
    .Define("GClosestJet_EncodedPartonFlavour_c", \
        "muon_closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{4})") \
    .Define("GClosestJet_EncodedPartonFlavour_b", \
        "muon_closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{5})") \
    .Define("GClosestJet_EncodedPartonFlavour_undefined", \
        "muon_closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{0})") \
    .Define("GClosestJet_EncodedHadronFlavour_b", \
        "muon_closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{5})") \
    .Define("GClosestJet_EncodedHadronFlavour_c", \
        "muon_closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{4})") \
    .Define("GClosestJet_EncodedHadronFlavour_light", \
        "muon_closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenMuon_eta, MGenMuon_phi, CleanGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{0})")
    # .Define("GenMuon_isReco_totNum_perEvent", "GenPart_MuonIdx_totNum_check[GenMuonMask]")

# Specify targets
col_to_save = ["GenMuon_isReco", "MGenMuon_mass", "MGenMuon_eta", "MGenMuon_pdgId", "MGenMuon_charge", "MGenMuon_phi", "MGenMuon_pt", "MGenMuon_statusFlags", "MGenMuon_statusFlag0", \
    "MGenMuon_statusFlag1", "MGenMuon_statusFlag2", "MGenMuon_statusFlag3", "MGenMuon_statusFlag4", "MGenMuon_statusFlag5", "MGenMuon_statusFlag6", "MGenMuon_statusFlag7", \
    "MGenMuon_statusFlag8", "MGenMuon_statusFlag9", "MGenMuon_statusFlag10", "MGenMuon_statusFlag11", "MGenMuon_statusFlag12", "MGenMuon_statusFlag13", "MGenMuon_statusFlag14", \
    "GClosestJet_dr", "GClosestJet_deta", "GClosestJet_dphi", "GClosestJet_pt", "GClosestJet_mass", "GClosestJet_EncodedPartonFlavour_light", "GClosestJet_EncodedPartonFlavour_gluon", \
    "GClosestJet_EncodedPartonFlavour_c", "GClosestJet_EncodedPartonFlavour_b", "GClosestJet_EncodedPartonFlavour_undefined", "GClosestJet_EncodedHadronFlavour_b",  \
    "GClosestJet_EncodedHadronFlavour_c", "GClosestJet_EncodedHadronFlavour_light"]

# Finally process columns and save to .root file
d_extracted.Snapshot("MMuons", "MMuonsA.root", col_to_save)