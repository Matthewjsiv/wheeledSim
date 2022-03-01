import numpy as np
import matplotlib.pyplot as plt

def make_racetrack(scale=1.0, npts = 20, difficulty=0.5, min_distance=0.05, direction=True):
    """
    Make a racetrack with more interesting shapes than ellipse
    Procedure:
        1. Generate a bunch of random points in a unit box
        2. Compute the convex hull
        3. Generate midpoints and displace them  to add concavity
        4. Fit a spline to get all points
        5. Scale to appropriate size
    Args:
        scale: multiplier on how much to scale up/down the track
        npts: number of points to generate convex hull from
        difficulty: Fraction of segment length we can displace midpoints by
        min_distance: Minimum distance between track points
        direction: True if cw False if ccw
    """
    pts = np.random.rand(npts, 2)
    track_pts = convex_hull(pts)
    #remove points that are too close
    dists = np.linalg.norm(track_pts[1:] - track_pts[:-1], axis=-1)
    dists = np.concatenate([dists, [float('inf')]])

    #Check with 2xmin_distance because we are adding midpoints
    track_pts = track_pts[dists > 2*min_distance]
    dists = dists[dists > 2*min_distance]
    
    #Add midpoints and displace
    midpoints = (track_pts[1:] + track_pts[:-1]) / 2
    rad = np.random.rand(midpoints.shape[0]) * (dists[:-1]/2. - min_distance) * difficulty
    ang = np.random.rand(midpoints.shape[0]) * 2 * np.pi
    disps = np.stack([np.cos(ang)*rad, np.sin(ang)*rad], axis=-1)
    disp_midpoints = midpoints + disps

#    plt.scatter(track_pts[:, 0], track_pts[:, 1], c='b', label='track points')
#    plt.scatter(midpoints[:, 0], midpoints[:, 1], c='r', label='midpoints')
#    plt.scatter(disp_midpoints[:, 0], disp_midpoints[:, 1], c='g', label='displaced midpoints')
#    plt.legend()
#    plt.show()

    ftrack_pts = [track_pts[0]]
    for mpt, tpt in zip(disp_midpoints, track_pts[1:]):
        ftrack_pts.append(mpt)
        ftrack_pts.append(tpt)

    #Lazy error checking
    if np.linalg.norm(ftrack_pts[0] - ftrack_pts[-1]) > 1e-4:
        ftrack_pts.append(ftrack_pts[0])

    ftrack_pts = np.stack(ftrack_pts, axis=0)

    #Make it so courses always start at 0,0 and have 50% chance of being cw
    ftrack_pts -= ftrack_pts[[0]]
    if direction:
        ftrack_pts[:, 1] *= -1

    return ftrack_pts

def convex_hull(pts):
    """
    find the convex hull of a set of points
    Algo:
        1. Put smallest y in hull
        2. while next point not pt0
            a. Compute angles to current point
            b. Add point with smallest (angle - current angle) to hull
    """
    #Smallest y coordinate always in hull
    #Choose this to keep all angles positive
    pt0 = pts[pts[:, 1].argmin()]
    hull = [pt0]
    disps = pts - np.expand_dims(pt0, axis=0)
    angles = np.arctan2(disps[:, 1], disps[:, 0])
    pts_sort = pts[np.argsort(angles)]

#    for i, pt in enumerate(pts_sort):
#        plt.scatter(pt[0], pt[1])
#        plt.text(pt[0], pt[1], i)
#    plt.show()

    #implement gift-wrapping algo
    curr_angle = 0
    curr_pt = pt0
    itrs = 0
    while (np.linalg.norm(curr_pt - pt0) > 1e-4) or (itrs < 1):
        disps = pts - np.expand_dims(curr_pt, axis=0)
        mask = np.linalg.norm(disps, axis=-1) > 1e-4
        check_pts = pts[mask]
        disps = disps[mask]
        angles = np.arctan2(disps[:, 1], disps[:, 0])
        angles -= curr_angle
        angles[angles < 0] += 2*np.pi
        idxs = np.argsort(angles)

        next_pt = check_pts[idxs[0]]
        curr_angle = np.arctan2(next_pt[1]-curr_pt[1], next_pt[0]-curr_pt[0])
        if curr_angle < 0:
            curr_angle += 2*np.pi

        curr_pt = next_pt
        hull.append(curr_pt)
        itrs += 1

        """
        #Viz
        for pt, angle in zip(check_pts, angles):
            plt.scatter(pt[0], pt[1], c='b')
            plt.text(pt[0], pt[1], "{:.2f}".format(angle))
        plt.scatter([x[0] for x in hull], [x[1] for x in hull], c='g')
        plt.show()
        """

    #Easier to duplicate the start point
#    hull.append(pt0)
    return np.stack(hull, axis=0)

def densify_racetrack(track_pts, ds=0.01):
    """
    upsample a track such that it has at least ds resolution
    TODO: cubic splines
    """
    upsample_pts = []
    for i in range(track_pts.shape[0]-1):
        p0 = track_pts[i]
        p1 = track_pts[i+1]
        d = np.linalg.norm(p1-p0)
        n = int(d/ds)
        alpha = np.linspace(0., 1., n)[:-1]
        disp = np.expand_dims(p1-p0, axis=0)

        pts = np.expand_dims(p0, axis=0) + np.expand_dims(alpha, axis=1) * disp
        upsample_pts.append(pts)

    upsample_pts = np.concatenate(upsample_pts, axis=0)
    return upsample_pts

if __name__ == '__main__':
    """
    script to create some racetrack files
    """
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True, help='number of tracks to make')
    parser.add_argument('--save_to', type=str, required=True, help='location to save to')
    args = parser.parse_args()

    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)

    for i in range(args.n):
        racetrack = make_racetrack(difficulty=1.0, direction=i > args.n/2)
        racetrack = densify_racetrack(racetrack)
        plt.plot(racetrack[:, 0], racetrack[:, 1], marker='.')
        plt.arrow(racetrack[0, 0], racetrack[0, 1], racetrack[10, 0] - racetrack[0, 0], racetrack[10, 1] - racetrack[0, 1], color='r', width=0.01)
        plt.show()

        np.save(os.path.join(args.save_to, 'track_{}'.format(i+1)), racetrack)
