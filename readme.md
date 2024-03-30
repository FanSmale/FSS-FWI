# Double-resolution full waveform inversion under self-paced learning

**Abstract**  

The advancement of image processing brings light to deep learning full waveform inversion (DL-FWI) of seismic data.
Existing DL-FWI approaches often suffer from the lack of interpretability and slow converge due to single network, and network parameter fluctuate due to static training control.
In this paper, we propose a double-resolution inversion network under self-paced learning to handle these issues.
Regarding the network design, the low-resolution part inverts low-wavenumber background velocity model using a lightweight U-Net-like architecture, while the high-resolution part inverts high-wavenumber details with the help of the background using dense modules and dual decoders.
Regarding the learning control, a new cross self-paced learning approach enhances the partial constraint minimization of the network.
This is implemented by applying dynamic parameter strategy to a joint loss function of the network.

---

"forward2openfwi.py" gives our forward simulation process.  
Our seismic records will be generated through this .py file.  

"fwi_dataset.py" encapsulates the FWI data and their set of related operations.  

"metrics" encapsulates our metrics calculation functions and some of the methods for logging metrics.  

"net_control" provides all operations related to network.  

"parse_args" gives the process by which the argparse object is initialized with a configuration file and a string entered by the user.  

"test" provides methods related to testing.  

"train" provides methods related to training.  

"utils.py" provides some basic methods.  
