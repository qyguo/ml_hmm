combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg.txt datacard_cat2_ggh_bkg.txt datacard_cat3_ggh_bkg.txt datacard_cat4_ggh_bkg.txt >  datacard_comb_sig_cat0_ggh.txt
combineCards.py datacard_cat0_ggh_bkg.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg.txt datacard_cat3_ggh_bkg.txt datacard_cat4_ggh_bkg.txt >  datacard_comb_sig_cat1_ggh.txt
combineCards.py datacard_cat0_ggh_bkg.txt datacard_cat1_ggh_bkg.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg.txt datacard_cat4_ggh_bkg.txt >  datacard_comb_sig_cat2_ggh.txt
combineCards.py datacard_cat0_ggh_bkg.txt datacard_cat1_ggh_bkg.txt datacard_cat2_ggh_bkg.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg.txt >  datacard_comb_sig_cat3_ggh.txt
combineCards.py datacard_cat0_ggh_bkg.txt datacard_cat1_ggh_bkg.txt datacard_cat2_ggh_bkg.txt datacard_cat3_ggh_bkg.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt
combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh.txt datacard_cat2_ggh.txt datacard_cat3_ggh.txt datacard_cat4_ggh.txt >  datacard_comb_sig_all_ggh.txt

text2workspace.py -m 125 datacard_comb_sig_cat0_ggh.txt
text2workspace.py -m 125 datacard_comb_sig_cat1_ggh.txt
text2workspace.py -m 125 datacard_comb_sig_cat2_ggh.txt
text2workspace.py -m 125 datacard_comb_sig_cat3_ggh.txt
text2workspace.py -m 125 datacard_comb_sig_cat4_ggh.txt
text2workspace.py -m 125 datacard_comb_sig_all_ggh.txt

combineCards.py datacard_cat0_vbf.txt datacard_cat1_vbf.txt datacard_cat2_vbf.txt datacard_cat3_vbf.txt > datacard_comb_vbf.txt

text2workspace.py -m 125 datacard_cat0_vbf.txt
text2workspace.py -m 125 datacard_cat1_vbf.txt
text2workspace.py -m 125 datacard_cat2_vbf.txt
text2workspace.py -m 125 datacard_cat3_vbf.txt
text2workspace.py -m 125 datacard_comb_vbf.txt

combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh.txt datacard_cat2_ggh.txt datacard_cat3_ggh.txt datacard_cat4_ggh.txt datacard_cat0_vbf.txt datacard_cat1_vbf.txt datacard_cat2_vbf.txt datacard_cat3_vbf.txt > datacard_comb_sig_all_total.txt
text2workspace.py -m 125 datacard_comb_sig_all_total.txt





combine -M Significance -d datacard_comb_sig_cat0_ggh.root -m 125 -n _signif_cat0_ggh --cminDefaultMinimizerStrategy 1 -t -1 --toysFrequentist --expectSignal 1 --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --setParameters pdf_index_ggh=0 --cminRunAllDiscreteCombinations --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParams --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

combine -M Significance -d datacard_comb_sig_cat1_ggh.root -m 125 -n _signif_cat1_ggh  --cminDefaultMinimizerStrategy 1 -t -1 --toysFrequentist --expectSignal 1 --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --setParameters pdf_index_ggh=0 --cminRunAllDiscreteCombinations --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

combine -M Significance -d datacard_comb_sig_cat2_ggh.root -m 125 -n _signif_cat2_ggh  --cminDefaultMinimizerStrategy 1 -t -1 --toysFrequentist --expectSignal 1 --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --setParameters pdf_index_ggh=0 --cminRunAllDiscreteCombinations --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

combine -M Significance -d datacard_comb_sig_cat3_ggh.root -m 125 -n _signif_cat3_ggh  --cminDefaultMinimizerStrategy 1 -t -1 --toysFrequentist --expectSignal 1 --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --setParameters pdf_index_ggh=0 --cminRunAllDiscreteCombinations --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

combine -M Significance -d datacard_comb_sig_cat4_ggh.root -m 125 -n _signif_cat4_ggh  --cminDefaultMinimizerStrategy 1 -t -1 --toysFrequentist --expectSignal 1 --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --setParameters pdf_index_ggh=0 --cminRunAllDiscreteCombinations --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

combine -M Significance -d datacard_cat0_vbf.root -n _signif_cat0_vbf -m 125   --cminDefaultMinimizerStrategy 1  -t -1 --toysFrequentist --expectSignal 1  --X-rtd FITTER_NEVER_GIVE_UP --X-rtd FITTER_BOUND  --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH
combine -M Significance -d datacard_cat1_vbf.root -n _signif_cat1_vbf -m 125   --cminDefaultMinimizerStrategy 1  -t -1 --toysFrequentist --expectSignal 1  --X-rtd FITTER_NEVER_GIVE_UP --X-rtd FITTER_BOUND  --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH
combine -M Significance -d datacard_cat2_vbf.root -n _signif_cat2_vbf -m 125   --cminDefaultMinimizerStrategy 1  -t -1 --toysFrequentist --expectSignal 1  --X-rtd FITTER_NEVER_GIVE_UP --X-rtd FITTER_BOUND  --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH
combine -M Significance -d datacard_cat3_vbf.root -n _signif_cat3_vbf -m 125   --cminDefaultMinimizerStrategy 1  -t -1 --toysFrequentist --expectSignal 1  --X-rtd FITTER_NEVER_GIVE_UP --X-rtd FITTER_BOUND  --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

combine -M Significance -d datacard_comb_sig_all_ggh.root -m 125 -n _signif_all_ggh  --cminDefaultMinimizerStrategy 1   -t -1 --toysFrequentist --expectSignal 1 --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --setParameters pdf_index_ggh=0 --cminRunAllDiscreteCombinations --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

combine -M Significance -d datacard_comb_vbf.root -n _signif_comb_vbf -m 125   --cminDefaultMinimizerStrategy 1  -t -1 --toysFrequentist --expectSignal 1  --X-rtd FITTER_NEVER_GIVE_UP --X-rtd FITTER_BOUND --cminRunAllDiscreteCombinations --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

combine -M Significance -d datacard_comb_sig_all_total.root -m 125 -n _signif_comb  --cminDefaultMinimizerStrategy 1   -t -1  --toysFrequentist --expectSignal 1 --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_BOUND --cminRunAllDiscreteCombinations --setParameterRanges r=-10,10 --X-rtd MINIMIZER_freezeDisassociatedParam --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd FAST_VERTICAL_MORPH

