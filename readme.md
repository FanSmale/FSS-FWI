# Double-resolution full waveform inversion under self-paced learning

The advancement of image processing brings light to deep learning full waveform inversion (DL-FWI) of seismic data.
Existing DL-FWI approaches often suffer from the lack of interpretability and slow converge due to single network, and network parameter fluctuate due to static training control.
In this paper, we propose a double-resolution inversion network under self-paced learning to handle these issues.
Regarding the network design, the low-resolution part inverts low-wavenumber background velocity model using a lightweight U-Net-like architecture, while the high-resolution part inverts high-wavenumber details with the help of the background using dense modules and dual decoders.
Regarding the learning control, a new cross self-paced learning approach enhances the partial constraint minimization of the network.
This is implemented by applying dynamic parameter strategy to a joint loss function of the network.
