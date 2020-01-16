# VoiceEmotion
1．	Run run.py directly. In order to run fast on the CPU, the model parameters are slightly reduced.

2．	Delete iemocap / model.pth before running.

3．	The format of input data is a 2D array: frame number * feature dimension. Among them, the size of feature dimension is 40. You can modify the input dimension by modifying the input dim variable in the settings.

4．	The filterBank feature extracted by the author is used for convenience, and only a small part of the data set is taken.

5．	Errors may be reported during the data loading process, because the newline characters of Linux and windows are different, you can change the newline characters of the files under iemocap / lists.
