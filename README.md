# Contributors
**Omar Mosleh**	[Email](omarmosleh@student.elte.hu); [LinkedIn](https://www.linkedin.com/in/omar-mosleh-029b91302/)
**Mhd Walid Al Jallad**	[Email](arnmwj@student.elte.hu); [LinkedIn](www.linkedin.com/in/mhd-walid-al-jallad-379498334)
Computational and Cognitive Neuroscience MSc.
Faculty of Education and Psychology
Eötvös Loránd University
## Supervised by
**Hunor István Lukács**
Department of Artificial Intelligence
Faculty of Informatics
Eötvös Loránd University

---

# Abstract
Electroencephalogram (EEG) is critical for diagnosing epilepsy, a chronic neurological disorder
affecting over 70 million people worldwide and characterized by abnormal neuronal discharges
causing recurrent seizures. Manual EEG analysis is labor-intensive and struggles with large data
volumes, necessitating efficient automated solutions. While existing seizure detection methods
predominantly use supervised learning, they face challenges due to rare annotated ictal events and
data imbalance. Emerging unsupervised approaches offer promise, such as transformer-based
models for label-free EEG activity identification [1].
By framing seizure identification as an anomaly detection problem, we aim to propose an
unsupervised anomaly detection framework using an xLSTM (Extended Long Short-Term
Memory) Autoencoder to detect epileptic seizures without expert-labeled data. To our knowledge,
this is the first application of xLSTM Autoencoders to unsupervised anomaly detection in time-
series EEG. This architecture leverages xLSTM’s enhanced memory mechanisms, improving
sequence modeling over traditional LSTMs [2].
	The model will be trained exclusively on normal EEG segments from healthy subjects.
Seizures will then be identified as anomalies via high reconstruction errors when input patterns
deviate from learned normal distributions. The encoder (residual mLSTM and sLSTM blocks)
compresses EEG segments into latent vectors, while the decoder (residual mLSTM and sLSTM
blocks) reconstructs the sequence. Anomaly scoring is usually done by measuring reconstruction
error and flagging segments above a threshold as seizures.
We use a curated subset of the Temple University Hospital TUH EEG Corpus, the TUH
EEG Epilepsy Corpus (TUEP), containing recordings from 100 epilepsy patients and 100 healthy
controls [3]. Two unipolar montages were used in the recordings: Average Reference (AR) and
Linked Ears Reference (LE), of which only the LE montage is selected due to its stability and
artifact robustness. Additionally, non-shared channels across recordings are excluded, retaining
only common channels.
	To allow the model to learn its own latent space from raw signals, we did not extract
features. Instead, we applied a minimal preprocessing pipeline: a [0.5–49 Hz] bandpass filter to
retain seizure-relevant frequencies and remove artifacts, z-score normalization and resampling to
250 Hz. We finally segmented the recordings into one-second windows with 50% overlap.
While the preprocessing pipeline is complete, model implementation and training on
normal EEG segments are underway. We plan to evaluate performance using accuracy, precision,
recall, F1-score, and AUC-ROC. Future steps include optimizing hyperparameters via
evolutionary algorithms, benchmarking against state-of-the-art methods (e.g., [1]) and validating
the model’s real-time detection capability.

## References
[1] İ. Y. Potter, G. Zerveas, C. Eickhoff, and D. Duncan, “Unsupervised Multivariate Time-Series
Transformers for Seizure Identification on EEG,” 2022 21st IEEE International Conference on
Machine Learning and Applications (ICMLA), Nassau, Bahamas, 2022, pp. 1304-1311.
doi: 10.1109/ICMLA55696.2022.00208.
[2] M. Beck et al., “xLSTM: Extended Long Short-Term Memory,” arXiv, 2024.
doi: 10.48550/ARXIV.2405.04517.
[3] S. I. Choi, S. Lopez, I. Obeid, M. Jacobson, and J. Picone, “The Temple University Hospital
EEG Corpus.” 2017. [Online]. Available: http://www.isip.piconepress.com/projects/tuh_eeg

# Conference Poster
- The initial steps of this work was published in [Proceedings of the MEi:CogSci Conference](https://journals.phl.univie.ac.at/meicogsci/index)[Link to abstract](https://journals.phl.univie.ac.at/meicogsci/article/view/978)
- The following poster was presented on MEi:CogSci conference 2025 in Ljubljana, Slovenia.

![Poster](misc\/img/ResearchPoster.png)
