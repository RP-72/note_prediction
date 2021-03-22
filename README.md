# note_prediction
The whole idea behind the working of the project is to locate the time at which a particular 
note is played and then using that information and other techniques to predict which 
note was played. Moreover, the project is divided into two steps: 
1. Detect the location of the note in the audio file
2. Check and analyze the audio file at that point to detect what note was played at 
that point.

The analysis here depends a lot on the recording, whether there were any 
background noises or has some high frequency or low-frequency noise pollution in it. 
Measures are taken in the program to distinguish between the important frequencies 
and the frequencies generated due to the noise in the background. The major libraries 
of python used here are **PyDub**, **NumPy**, **SciPy**, **python-Levenshtein** and **Tkinter**. The 
whole process after being divided, spans over the two topics note location 
detection and not classification. This project helps in understanding PyDub as a 
library and also at the same time teaches how to take an audio sample and perform a 
practical Fast Fourier transform over it. 
