import rospy
import numpy as np
import matplotlib.pyplot as plt

from numpy import sin, cos
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped

from wheeledSim.util import quat_to_yaw

class PDTracker:
    """
    Simple PD controller that tracks to a lookahead.
    """
    def __init__(self, Kp=np.array([0.7, -4.]), Kd=np.array([0., 0.]), lookahead=0.5, done_threshold=0.1):
        self.Kp = Kp
        self.Kd = Kd
        self.lookahead = lookahead
        self.done_threshold = done_threshold

        self.position = np.zeros(2)
        self.lookahead_pt = np.zeros(2)
        self.yaw = np.zeros(1)
        self.path = np.zeros([1, 2])
        self.err_old = np.zeros(2) #Need this for PD
        self.is_done = True

    def handle_odom(self, msg):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.yaw = quat_to_yaw(msg.pose.pose.orientation)

    def handle_path(self, msg):
        new_path = []
        for pose in msg.poses:
            new_path.append(np.array([pose.pose.position.x, pose.pose.position.y]))
        new_path = np.stack(new_path, axis=0)
        self.path = new_path
        self.check_done()

    def check_done(self):
        self.is_done = np.linalg.norm(self.position - self.path[-1]) < self.done_threshold

    def get_lookahead(self):
        if self.path.shape[0] == 1:
            return self.path[-1]

        ds = np.linalg.norm(self.path[1:] - self.path[:-1], axis=-1)
        dists = np.linalg.norm(self.path - self.position, axis=-1)
        path_idx = dists.argmin()
        lookahead_idx = np.copy(path_idx)

        d = 0.
        while d < self.lookahead and lookahead_idx < self.path.shape[0]-1:
            d += ds[lookahead_idx]
            lookahead_idx += 1

        lookahead_pt = self.path[lookahead_idx]

        #We overshoot the lookahead by a little. Linear interpolate to be exact.
#        if lookahead_idx < self.path.shape[0]:
#            frac = (ds[lookahead_idx-1] - (d - self.lookahead)) / ds[lookahead_idx-1]
#            lookahead_pt = (1-frac) * self.path[lookahead_idx - 1] + frac * self.path[lookahead_idx]

        self.lookahead_pt = lookahead_pt

    def cmd_msg(self):
        """
        Don't forget to convert the error to the robot frame.
        """

        self.check_done()
        self.get_lookahead()

        planar_err = self.lookahead_pt - self.position
        err = np.array([
            planar_err[0] * cos(self.yaw) + planar_err[1] * sin(self.yaw),
            -planar_err[0] * sin(self.yaw) + planar_err[1] * cos(self.yaw)
            ])

        d_err = err - self.err_old
        act = np.zeros(2) if self.is_done else self.Kp * err + self.Kd * d_err
        self.err_old = err

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = act[0]
        msg.drive.steering_angle = act[1]
        return msg

    def viz(self, fig, axs):
        axs[0].plot(self.path[:, 0], self.path[:, 1], c='r', label='Path')
        axs[0].scatter(self.lookahead_pt[0], self.lookahead_pt[1], marker='.', c='g', label='Lookahead')
        axs[0].scatter(self.position[0], self.position[1], marker='^', c='b', label='Robot')
        axs[0].arrow(self.position[0], self.position[1], cos(self.yaw), sin(self.yaw), color='y')
        axs[0].legend()

        axs[1].plot([0, self.err_old[0]], [0, 0], c='r', label='Local X error')
        axs[1].plot([0, 0], [0, self.err_old[1]], c='g', label='Local y error')
        axs[1].set_xlim(-1, 1)
        axs[1].set_ylim(-1, 1)
        axs[1].legend()

if __name__ == '__main__':
    tracker = PDTracker()

    rospy.init_node("pd_tracker")

    odom_sub = rospy.Subscriber("/odom", Odometry, tracker.handle_odom)
    path_sub = rospy.Subscriber("/planner/path", Path, tracker.handle_path)
    cmd_pub = rospy.Publisher("/cmd", AckermannDriveStamped)

    rate = rospy.Rate(10)

    print('Waiting 1s for odom...')
    for _ in range(10):
        rate.sleep()

#    plt.show(block = False)
#    fig, axs = plt.subplots(1, 2, figsize = (8, 4))

    while not rospy.is_shutdown():
        cmd = tracker.cmd_msg()
        print('_' * 50)
        print("Path shape = {}".format(tracker.path.shape))
        print("Position = {}".format(tracker.position))
        print("Lookahead = {}".format(tracker.lookahead_pt))
        print("Ego Error = {}".format(tracker.err_old))
        print("Goal = {}".format(tracker.path[-1]))
        print("Act = {}".format(cmd))
        print("Done = {}".format(tracker.is_done))
        print("Goal Dist = {}".format(np.linalg.norm(tracker.position - tracker.path[-1])))

        cmd_pub.publish(cmd)

#        for ax in axs:
#            ax.cla()
#        tracker.viz(fig, axs)
#        plt.pause(1e-2)

        rate.sleep()
