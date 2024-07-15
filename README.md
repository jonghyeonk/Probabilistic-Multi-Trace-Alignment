# Descriptions

### Paper under review (an extended paper from "Approximating Multi-Perspective Trace Alignment Using Trace Encodings (BPM 2023)")

This repository contains the scripts developed for a proposed probabilistic approximate approach of multi-perspective alignment.

We evaluated our approach (in probabilistic setting) by using 5 event logs (Synthetic logs: Credit, Pub / Real logs: Helpdesk, Road Fines, Hospital Billing).

1. 'encoding_multinoise.py' and 'encoding_partial.py'

    This script shows the processes (i) to generate non-complying traces by modifying the original traces, (ii) to save ground truth of original traces before modification, and (iii) to encode the prepared experimental datasets.

2. 'main.py'

    This TBD

4. Folder 'non_schatistic' includes datasets in non-stochastic setting and experimental codes

    This script shows (i) the approximate approach of multi-perspective alignment and (ii) the results by implementing it (the summary of the results is seen in table 3~6 in our paper).
