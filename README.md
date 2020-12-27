# Ultrafast

These Python programs are used in our lab for data analysis for:
- Transient Absorption Spectroscopy (TAS)
- Pump-Degenerate Four-Wave Mixing (pDFWM)

The TAS programs are an interactive image profile map, similar to the image profile graphing option found in Origin, and a global fitting population program.

The image profile program is still in the works and works best in a text editor, rather than a full IDE like Spyder, as the user can use their mouse to interact with the plots, which then give updated transients/spectra. The scaling of the map still needs improvement.

The global fitting program can theoretically fit as many transients as the user wishes, but currently up to 5 is supported. New models are also easy to impliment.
The object-oriented nature of this program makes fitting multiple data sets, or the same dataset with various models, very easy.

The pDFWM program can simply be used for simple FFT's and for fitting Gaussians. Currently fitting 7 peaks is supported, but it is easy to add more peaks or to add different functions, such as Lorenzians.
For pDFWM, data is collected in the lab such that three sets of data are saved for each pump delay time step: pump on, pump off, and the subtracted signals. If your lab is like ours, this means you have many files to sort though.
The pDFWM program includes a class that sorts through these files, and the sorting is naturally based on how we name our files. If you name your files differently, feel free to adjust the Data class.
The pDFWM class fits all of the peaks that the user chooses and stores the results and their corresponding errors in one numpy array. This array is already sorted by time step. There is a function in the program to average the data if there are repeating time steps.
The Gaussian function used for fitting is based on the GaussAmp function from Origin. The pDFWM().transients() method automatically averages all of the baselines, to ensure that the transients of the amplitudes are more reliable.

