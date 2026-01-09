# DDR-Accelero
Virtual Dance Pad using smartphone accelerometer.

Folder `raw_data` contains captures taken on android phone with Sensor Logger.
Folder `sm_files` contains stepmania .sm files database. (The captures match sm_files).

The idea is to progressivelly elaborate a dataset. The first thing is to match a raw_data capture to the correct sm file, at the correct offset. Correlation with minimal pre-processing of sm_files (to generate a comparable signal) should be sufficient, or else with a better feature detection approach.

We create a collection of minimal independant scripts to test the ideas.

After we are able to align files, we can think about how to create a dataset, by associating to each window of signal (-500ms + 500ms for instance), the main corresponding arrows to detect.

The final idea is to be able to predict pressed arrows on the DDR mat solely with sensor data.