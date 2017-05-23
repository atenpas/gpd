# This message describes a 2-finger grasp configuration by its 6-DOF pose, 
# consisting of a 3-DOF position and 3-DOF orientation, and the opening 
# width of the robot hand.

# Position
geometry_msgs/Point bottom # centered bottom/base of the robot hand)
geometry_msgs/Point top # centered top/fingertip of the robot hand)
geometry_msgs/Point surface # centered position on object surface

# Orientation represented as three axis (R = [approach binormal axis])
geometry_msgs/Vector3 approach # The grasp approach direction
geometry_msgs/Vector3 binormal # The binormal
geometry_msgs/Vector3 axis # The hand axis

geometry_msgs/Point sample # Point at which the grasp was found

std_msgs/Float32 width # Required aperture (opening width) of the robot hand 

std_msgs/Float32 score # Score assigned to the grasp by the classifier
