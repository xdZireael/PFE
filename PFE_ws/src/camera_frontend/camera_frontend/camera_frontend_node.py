import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from slam_msgs.msg import LoopConstraint


ORB_FEATURES       = 1000
KF_DIST_M          = 0.20        # minimum translation for a new keyframe (m)
KF_ANGLE_RAD       = 0.15        # minimum rotation   for a new keyframe (rad)
MATCH_RATIO        = 0.75        # Lowe ratio-test threshold
MIN_MATCHES        = 10          # minimum good matches to attempt PnP
PNP_INLIERS_MIN    = 12          # minimum PnP RANSAC inliers to accept pose
PNP_REPROJ_ERR     = 4.0         # RANSAC reprojection error (px)
DEPTH_MIN_M        = 0.20        # minimum valid depth (m)
DEPTH_MAX_M        = 8.00        # maximum valid depth (m)
LC_MIN_AGE         = 10          # keyframes to skip for loop search
LC_MIN_MATCHES     = 20          # minimum inlier matches to accept a loop
LC_SCORE_THR       = 0.04        # BoW similarity score threshold


class SimpleBoW:
    """
    Lightweight bag-of-words built from ORB descriptors using k-means
    clustering.  Trains lazily once VOCAB_SIZE keyframes have been seen.
    """
    VOCAB_SIZE   = 256   
    TRAIN_AFTER  = 20    

    def __init__(self):
        self._all_desc   = []   # list of (N,32) uint8 arrays
        self._vocab      = None # (VOCAB_SIZE, 32) float32 centroids
        self._histograms = {}   # kf_id → normalised histogram

    def add(self, kf_id: int, desc: np.ndarray):
        """Add a keyframe's descriptors; trigger vocab training when ready."""
        self._all_desc.append(desc)
        if self._vocab is None and len(self._all_desc) >= self.TRAIN_AFTER:
            self._train()
        if self._vocab is not None:
            self._histograms[kf_id] = self._encode(desc)

    def query(self, desc: np.ndarray, exclude_recent: int = 0) -> list:
        """
        Returns list of (kf_id, score) sorted by descending similarity.
        Score is cosine similarity of BoW histograms (higher = more similar).
        """
        if self._vocab is None or desc is None or not self._histograms:
            return []
        q_hist  = self._encode(desc)
        max_id  = max(self._histograms.keys())
        results = []
        for kf_id, hist in self._histograms.items():
            if kf_id > max_id - exclude_recent:  
                continue
            score = float(np.dot(q_hist, hist))
            results.append((kf_id, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _train(self):
        all_f = np.vstack(self._all_desc).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        _, _, self._vocab = cv2.kmeans(
            all_f, self.VOCAB_SIZE, None, criteria,
            attempts=3, flags=cv2.KMEANS_PP_CENTERS)
        for i, desc in enumerate(self._all_desc):
            self._histograms[i] = self._encode(desc)

    def _encode(self, desc: np.ndarray) -> np.ndarray:
        """Assign each descriptor to nearest visual word → L2-normalised hist."""
        desc_f = desc.astype(np.float32)
        dists = np.sum(
            (desc_f[:, None, :] - self._vocab[None, :, :]) ** 2, axis=2)
        words  = np.argmin(dists, axis=1)
        hist   = np.bincount(words, minlength=self.VOCAB_SIZE).astype(np.float32)
        norm   = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist


class VisualKeyFrame:
    __slots__ = ('id', 'pose', 'kps', 'desc', 'pts3d')

    def __init__(self, kf_id, pose, kps, desc, pts3d):
        self.id    = kf_id
        self.pose  = pose.copy()   
        self.kps   = kps
        self.desc  = desc
        self.pts3d = pts3d         



class CameraFrontendNode(Node):

    def __init__(self):
        super().__init__('camera_frontend')

        self._orb     = cv2.ORB_create(ORB_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._bow     = SimpleBoW()
        self._bridge  = CvBridge()

        self._K         = None   
        self._cam_ready = False

        self._pose        = np.zeros(3)   
        self._last_kf_pose = np.zeros(3)
        self._prev_kps    = None
        self._prev_desc   = None
        self._prev_pts3d  = None
        self._has_prev    = False
        self._keyframes   = []            
        self._latest_depth = None         

        self._rgb_sub = self.create_subscription(
            Image, '/oakd/rgb/image_raw',
            self._on_rgb, 10)

        self._depth_sub = self.create_subscription(
            Image, '/oakd/stereo/image_raw',
            self._on_depth, 10)

        latch = QoSProfile(depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._info_sub = self.create_subscription(
            CameraInfo, '/oakd/rgb/camera_info',
            self._on_camera_info, latch)

        self._odom_pub = self.create_publisher(Odometry,        '/camera/odometry',        10)
        self._loop_pub = self.create_publisher(LoopConstraint,  '/camera/loop_constraint', 10)

        self.get_logger().info('camera_frontend ready')

    def _on_camera_info(self, msg: CameraInfo):
        if self._cam_ready:
            return
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._cam_ready = True
        self.get_logger().info(
            f'[camera] intrinsics ready  fx={self._K[0,0]:.1f}  fy={self._K[1,1]:.1f}')


    def _on_depth(self, msg: Image):
        try:
            self._latest_depth = self._bridge.imgmsg_to_cv2(msg, '16UC1')
        except Exception as e:
            self.get_logger().warn(f'depth bridge error: {e}')

    def _on_rgb(self, msg: Image):
        if not self._cam_ready or self._latest_depth is None:
            return

        try:
            bgr  = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn(f'rgb bridge error: {e}')
            return

        kps, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < MIN_MATCHES:
            return

        pts3d = self._lift_to_3d(kps, self._latest_depth)

        if self._has_prev:
            delta = self._estimate_delta(
                self._prev_kps, self._prev_desc, self._prev_pts3d,
                kps, desc, pts3d)
            if delta is not None:
                dx, dy, dth = delta
                ch, sh = math.cos(self._pose[2]), math.sin(self._pose[2])
                self._pose[0] += dx * ch - dy * sh
                self._pose[1] += dx * sh + dy * ch
                self._pose[2]  = _norm_angle(self._pose[2] + dth)

        dist  = math.hypot(self._pose[0] - self._last_kf_pose[0],
                           self._pose[1] - self._last_kf_pose[1])
        angle = abs(_norm_angle(self._pose[2] - self._last_kf_pose[2]))

        if not self._has_prev or dist > KF_DIST_M or angle > KF_ANGLE_RAD:
            self._add_keyframe(kps, desc, pts3d)
            self._last_kf_pose = self._pose.copy()

        self._prev_kps   = kps
        self._prev_desc  = desc
        self._prev_pts3d = pts3d
        self._has_prev   = True

        self._publish_odometry(msg.header.stamp)

    def _lift_to_3d(self, kps, depth: np.ndarray) -> np.ndarray:
        """Returns (N,3) float32 array.  Invalid points are (0,0,0)."""
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]
        h, w   = depth.shape
        pts    = np.zeros((len(kps), 3), dtype=np.float32)

        for i, kp in enumerate(kps):
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            z = depth[v, u] * 0.001         
            if z < DEPTH_MIN_M or z > DEPTH_MAX_M:
                continue
            pts[i] = [(u - cx) * z / fx,
                      (v - cy) * z / fy,
                      z]
        return pts

    def _estimate_delta(self, kps1, desc1, pts3d1, kps2, desc2, pts3d2):
        raw = self._matcher.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in raw
                if len([m, n]) == 2 and m.distance < MATCH_RATIO * n.distance]

        if len(good) < MIN_MATCHES:
            return None

       
        obj_pts = []   
        img_pts = []   
        for m in good:
            p3 = pts3d1[m.queryIdx]
            if p3[2] == 0.0:      
                continue
            obj_pts.append(p3)
            img_pts.append(kps2[m.trainIdx].pt)

        if len(obj_pts) < MIN_MATCHES:
            return None

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        # PnP RANSAC
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self._K, None,
            iterationsCount=100,
            reprojectionError=PNP_REPROJ_ERR,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE)

        if not ok or inliers is None or len(inliers) < PNP_INLIERS_MIN:
            return None

        R, _ = cv2.Rodrigues(rvec)
        dx  = float(tvec[2])   
        dy  = -float(tvec[0])  
        dth = math.atan2(R[1, 0], R[0, 0])
        return dx, dy, dth

    def _add_keyframe(self, kps, desc, pts3d):
        kf_id = len(self._keyframes)
        kf    = VisualKeyFrame(kf_id, self._pose, kps, desc, pts3d)
        self._keyframes.append(kf)
        self._bow.add(kf_id, desc)

        if kf_id >= LC_MIN_AGE:
            self._detect_loop(kf)

    def _detect_loop(self, cur: VisualKeyFrame):
        candidates = self._bow.query(cur.desc, exclude_recent=LC_MIN_AGE)

        for cand_id, score in candidates[:5]:     # check top-5
            if score < LC_SCORE_THR:
                break

            cand   = self._keyframes[cand_id]
            delta  = self._estimate_delta(
                cand.kps, cand.desc, cand.pts3d,
                cur.kps,  cur.desc,  cur.pts3d)

            if delta is None:
                continue

            dx, dy, dth = delta

            dist = math.hypot(dx, dy)
            if dist > 3.0 or abs(dth) > math.pi / 3:
                self.get_logger().warn(
                    f'[camera] loop rejected: implausible delta '
                    f'dist={dist:.2f}m  dth={math.degrees(dth):.1f}deg')
                continue
        
            msg = LoopConstraint()
            msg.from_id = cur.id
            msg.to_id   = cand_id
            msg.dx      = float(dx)
            msg.dy      = float(dy)
            msg.dtheta  = float(dth)
            msg.score   = float(score)
            msg.source  = 'camera'
            self._loop_pub.publish(msg)

            self.get_logger().info(
                f'[camera] loop: KF{cur.id} → KF{cand_id}  BoW={score:.3f}')
            return   # one loop per keyframe

    def _publish_odometry(self, stamp):
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'map'
        msg.child_frame_id  = 'base_link'
        msg.pose.pose.position.x = float(self._pose[0])
        msg.pose.pose.position.y = float(self._pose[1])
        half = self._pose[2] * 0.5
        msg.pose.pose.orientation.z = math.sin(half)
        msg.pose.pose.orientation.w = math.cos(half)
        self._odom_pub.publish(msg)


def _norm_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def main(args=None):
    rclpy.init(args=args)
    node = CameraFrontendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()