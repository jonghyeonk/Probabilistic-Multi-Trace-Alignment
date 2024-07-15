# Descriptions

### Paper under review (an extended paper from "Approximating Multi-Perspective Trace Alignment Using Trace Encodings (BPM 2023)")

This repository contains the scripts developed for a proposed probabilistic approximate approach of multi-perspective alignment.

We evaluated our approach (in probabilistic setting) by using 5 event logs (Synthetic logs: Credit, Pub / Real logs: Helpdesk, Road Fines, Hospital Billing).

1. 'encoding_multinoise.py' and 'encoding_partialnoise.py'

    The two script show the processes (i) to generate non-complying traces by modifying the original traces and (ii) to encode the traces for training kNN using 5 encoding methods ('aggregate', 'boolean', 'complexindex', 'laststate', 'p-gram+aggregate'). The division between 'multinose' and 'partialnoise' is used for detailed analysis. In table 3-4 of our paper, we used encoded logs with 'multinoise' for general performance comparision with baselines. We used encoded logs with 'partialnoise' only for deeper analysis on effects of each noise patterns (see Figure 8). The encoded logs will be saved in the folder 'data_trans'.

2. 'main.py'

    Using the encoded logs generated from step 1, we implement our developed approach. The summary of result ('precision', 'distance', 'computation time') will be saved in the folder 'result'. Note that if you activate the line 211 in the code, you can get results on 'partialnose'.

3. The others

    The two synthetic logs (Credit and Pub) are generated from the bpmn files ('credit_card.bpmn' and 'mccloud_sim.bpmn' respectively). The scripts for the 5 encoding functions used in our paper can be found in the folder 'tansformers'.
