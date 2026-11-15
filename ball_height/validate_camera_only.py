"""How good is the reconstruction when no shadow is available (cloud)?
Re-run everything WITHOUT the shadow measurements and compare, frame by frame, with the direct single-frame shadow
triangulation (which never uses the flight model).  usage: python validate_camera_only.py"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reconstruct as rc
HERE = os.path.dirname(os.path.abspath(__file__))


def compare(res_cam, res_full, label=''):
    m = res_cam.merge(res_full[['frame', 'z_m', 'y_m', 'x_m', 'z_shadow_m', 'method', 'shadow_used']], on='frame', suffixes=('', '_full'))
    ok = m.state.eq('flight') & m.shadow_used_full & m.z_shadow_m_full.notna() & m.z_m.notna() & m.method_full.eq('flight+shadow')
    e = (m.z_m - m.z_shadow_m_full)[ok]
    ey = (m.y_m - m.y_m_full)[ok]
    q = np.percentile(np.abs(e), [50, 68, 90, 95])
    print('%-34s n=%5d  z error: bias %+.3f  median|e| %.3f  68%% %.3f  90%% %.3f  95%% %.3f  rms %.3f | depth(y) median|e| %.2f' % (
        label, ok.sum(), e.mean(), q[0], q[1], q[2], q[3], np.sqrt((e ** 2).mean()), np.median(np.abs(ey))))
    return m[ok].assign(err=e)


if __name__ == '__main__':
    full = pd.read_csv(os.path.join(HERE, 'ball_height.csv'))
    res, fl, ct, outs = rc.main(use_shadow=False, write=False, verbose=False)
    m = compare(res, full, 'camera only')
    m.to_csv(os.path.join(HERE, 'validation_camera_only.csv'), index=False)
