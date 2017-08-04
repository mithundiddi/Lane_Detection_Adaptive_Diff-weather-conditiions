## Advanced Lane Finding by Naoki

Made for Robotics Sensing and Navigation at Northeastern University

Steps taken for processsing frames
---
Each frame was processed using these thresholds:
* Frames are saturated so that the yellow pops out more for better edge detecting
* Edges are detected using the Canny Edge detector; edge mask produced
* Yellow and white are detected using HSL values; color mask produced
* Edge mask is blurred by a 10x10 kernel to ensure overlap, then ANDed with the color mask
* Mask is used to isolate desired colors
* Perspective transform implemented using a trapezoid that avoids the hood of the car and includes as many lane marking as possible
* Polyfit function used to overlay best fit lines
* Perspective transform inversed and road is now colored between lanes
