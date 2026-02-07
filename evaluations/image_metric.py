import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from cleanfid import fid
from pytorch_msssim import ms_ssim
import lpips


def load_images(folder, transform, device):
    """Load and preprocess all PNG images from a folder."""
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
    return torch.stack(images).to(device)


def compute_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def main():
    parser = argparse.ArgumentParser(
        description='Compute image quality metrics (FID, KID, CLIP-FID, LPIPS, PSNR, MS-SSIM) between two folders.'
    )
    parser.add_argument('--folder1', type=str, required=True,
                        help='Path to the first folder (ground truth images)')
    parser.add_argument('--folder2', type=str, required=True,
                        help='Path to the second folder (predicted images)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output .txt file to save metrics')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    folder1 = args.folder1
    folder2 = args.folder2

    images1 = load_images(folder1, transform, device)
    images2 = load_images(folder2, transform, device)

    assert images1.size(0) == images2.size(0), \
        "Folders must contain the same number of images."

    metrics = []

    # FID
    fid_score = fid.compute_fid(folder1, folder2)
    print(f'FID: {fid_score:.4f}')
    metrics.append(f'FID: {fid_score:.4f}')

    # KID
    kid_score = fid.compute_kid(folder1, folder2)
    print(f'KID: {kid_score:.4f}')
    metrics.append(f'KID: {kid_score:.4f}')

    # CLIP-FID
    clip_fid_score = fid.compute_fid(folder1, folder2, mode="clean", model_name='clip_vit_b_32')
    print(f'CLIP-FID: {clip_fid_score:.4f}')
    metrics.append(f'CLIP-FID: {clip_fid_score:.4f}')

    # LPIPS
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_scores = []
    for img1, img2 in zip(images1, images2):
        score = lpips_model(img1.unsqueeze(0), img2.unsqueeze(0))
        lpips_scores.append(score.item())
    avg_lpips = np.mean(lpips_scores)
    print(f'LPIPS: {avg_lpips:.4f}')
    metrics.append(f'LPIPS: {avg_lpips:.4f}')

    # PSNR
    psnr_scores = []
    for i1, i2 in zip(images1, images2):
        psnr_val = compute_psnr(i1, i2)
        if isinstance(psnr_val, float) and np.isinf(psnr_val):
            psnr_scores.append(float('inf'))
        else:
            psnr_scores.append(psnr_val.item())

    finite_psnr_scores = [s for s in psnr_scores if not np.isinf(s)]
    if finite_psnr_scores:
        avg_psnr = np.mean(finite_psnr_scores)
        print(f'PSNR: {avg_psnr:.4f}')
        metrics.append(f'PSNR: {avg_psnr:.4f}')
    else:
        print('PSNR: inf (identical images)')
        metrics.append('PSNR: inf (identical images)')

    # MS-SSIM
    ms_ssim_scores = ms_ssim(images1, images2, data_range=1.0, size_average=False)
    avg_ms_ssim = ms_ssim_scores.mean().item()
    print(f'MS-SSIM: {avg_ms_ssim:.4f}')
    metrics.append(f'MS-SSIM: {avg_ms_ssim:.4f}')

    # Save results
    with open(args.output, 'w') as f:
        f.write('\n'.join(metrics))
    print(f'Metrics saved to {args.output}')


if __name__ == "__main__":
    main()
