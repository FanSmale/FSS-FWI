# FSS-FWI: Frequency Scale-Separated FullWaveform Inversion under Cross Self-Paced Learning

**Abstract**  

Full waveform inversion (FWI), as a cutting-edge imaging technique in geophysics, is widely used to reconstruct various underground structures.
Recently, the deep learning full waveform inversion (DL-FWI) has attracted extensive research interest due to its low prediction overhead.
However, existing DL-FWI approaches struggle to trade off the accuracy of background velocities and geological details.
In this paper, we propose a low- and high-Frequency Scale-Separated Full Waveform Inversion approach under cross self-paced learning (FSS-FWI).
The key idea is to invert low-frequency velocities using a lightweight auxiliary network, and high-frequency details using the main network.
Consequently, the background and details represented by different frequencies are decoupled at the task level to avoid erroneous feature mapping.
Additionally, the idea of self-paced learning that cross-constrains l2 and perceptual weight is incorporated into the loss function.
It is convenient to deploy and facilitates the convergence of complex features in multi-constraint environments.
Experiments are undertaken on three datasets from OpenFWI and a slice dataset from Marmousi II.
Results show that our approach outperforms state-of-the-art DL-FWI approaches in terms of five metrics.

---


"fwi_dataset.py" encapsulates the FWI data and their set of related operations.  

"metrics" encapsulates our metrics calculation functions and some of the methods for logging metrics.  

"net_control" provides all operations related to network.  

"parse_args" gives the process by which the argparse object is initialized with a configuration file and a string entered by the user.  

"test" provides methods related to testing.  

"train" provides methods related to training.  

"utils.py" provides some basic methods.  
