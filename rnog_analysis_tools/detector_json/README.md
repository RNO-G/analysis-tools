## Detector JSON
These are some scripts to build a JSON file containing information about RNO-G to be used with NuRadioMC.

### build_detector.py
A script that builds the detector description

### build_info.json
A JSON file containing the instructions on how to assemble the detector description

### draw_detector.py
A script to draw the resulting detector, to check if it worked.

### fiber_delays.py
Uses Kaeli's measurements to determine the signal delays from the fibers used for the deep channels. The measure
ments are stored at https://drive.google.com/drive/u/0/folders/1p2JDgfZcc8YXz6twgPYIJ1CAyt_5v2no (you may need to ask 
Kaeli for permission)

### fiber_delays.json
The fiber delays determined by fiber_delays.py
