# Robot status information
string current_state  # idle, moving, grasping, error
geometry_msgs/Pose current_pose
bool is_gripper_open
string last_error
uint64 timestamp
