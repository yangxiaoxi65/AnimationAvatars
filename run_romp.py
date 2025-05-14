import glob
from tqdm import tqdm
import cv2
import numpy as np
import romp


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    settings = romp.main.default_settings
    romp_model = romp.ROMP(settings)

    results = []
    for p in tqdm(sorted(glob.glob(f"{args.data_dir}/images/*"))):
        img = cv2.imread(p)
        result = romp_model(img)
        if result["body_pose"].shape[0] > 1:
            result = {k: v[0:1] for k, v in result.items()}
        results.append(result)

    results = {
        k: np.concatenate([r[k] for r in results], axis=0) for k in result
    }

    np.savez(f"{args.data_dir}/poses.npz", **{
        "betas": results["smpl_betas"].mean(axis=0),
        "global_orient": results["smpl_thetas"][:, :3],
        "body_pose": results["smpl_thetas"][:, 3:],
        "transl": results["cam_trans"],
    })
