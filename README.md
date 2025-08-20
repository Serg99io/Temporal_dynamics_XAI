# Temporal_dynamics_XAI
GitHub Repository of the paper "Unraveling Temporal Dynamics in Resting-State Data with an Interpretable Siamese Convolutional Neural Network" by Sergio Kazatzidis

To work with this repository you will need to download the subject data Set 1 of the prepocessed data from this site: https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html and put it all in a "subject_data" folder.

Further preprocessing is done within the file Preprocessing.py. In training_model.py you can train the model from scratch. And in analysis.py all the analysis is taking place. 

How time is exactly represented in the brain remains a fundamental and unresolved question. In this study, we employed the Relative Positioning task to train a model to learn temporal dynamics from resting-state EEG data. Specifically, the model was tasked with determining whether two EEG windows were temporally close or distant. We utlized a Siamese convolutional neural network architecture.![Alt text](images/model_siamese.png)



We achieved the highest accuracy of 77.22\% when training on data filtered above 30 Hz, outperforming the other frequency bands, with notable variability observed across subjects. To interpret the model’s decision-making, we applied Grad-CAM, an explainable AI technique, to visualize the features driving classification. Analysis of the power spectral density of the Grad-CAM weights revealed a focus on signals above 30 Hz, as well as a prominent peak around 2 Hz. Further examination showed that the most important Grad-CAM weights in the 1.5–4 Hz range reflected changes in power modulation and rhythmicity of gamma oscillations. Our findings suggest that this low-frequency peak in the Grad-CAM weights reflects slow fluctuations in the gamma envelope, which likely encode temporal structure and could be used to differentiate temporally close and distant segments in resting-state EEG. As shown in the figure ![Alt text](images/envelop_powerdiff.pdf)








For questions, feel free to contact Sergio Kazatzidis (s.kazatzidis@gmail.com).
