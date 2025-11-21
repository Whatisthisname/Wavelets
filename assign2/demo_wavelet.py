#!/usr/bin/env python3
"""Demo: Daubechies 4-tap wavelet (db2) and simple compression tests.

Shows filter coefficients, scaling/wavelet functions, and demonstrates
wavelet-domain compression on a sample image using PyWavelets and scikit-image.

Saves results to `outputs/` in the current folder.
"""

import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def show_db2_filters_and_wavefun(outdir):
    w = pywt.Wavelet("db2")
    dec_lo = np.array(w.dec_lo)
    dec_hi = np.array(w.dec_hi)
    rec_lo = np.array(w.rec_lo)
    rec_hi = np.array(w.rec_hi)

    print('Daubechies "db2" (4-tap) filter coefficients:')
    print("dec_lo:", dec_lo)
    print("dec_hi:", dec_hi)
    print("rec_lo:", rec_lo)
    print("rec_hi:", rec_hi)

    # scaling (phi) and wavelet (psi) functions
    phi, psi, x = w.wavefun(level=8)

    plt.figure(figsize=(8, 4))
    plt.plot(x, phi, label="scaling (phi)")
    plt.plot(x, psi, label="wavelet (psi)")
    plt.title("Daubechies db2: scaling and wavelet functions")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(outdir, "db2_wavefun.png")
    plt.savefig(p)
    plt.close()
    print("Saved wavefun plot to", p)


def show_orthogonal_wavelets(wavelet_name, phi, psi, x, outdir):
    """ """
    w = pywt.Wavelet(wavelet_name)
    # Use the reconstruction (synthesis) low-pass/high-pass filters.

    # interpolation helper: evaluate phi at arbitrary points using linear interp
    def interp(f_vals, x_vals, pts):
        return np.interp(pts, x_vals, f_vals, left=0.0, right=0.0)

    plt.figure(figsize=(12, 12))
    depth = 4
    for j in range(0, depth):
        plt.subplot(depth, 1, j + 1)
        for k in range(0, 2**j + 2 ** (j + 1) - 2):
            plt.plot(x, interp(psi, x, 2**j * x - k))

    outp = os.path.join(outdir, "orthogonal_set.png")
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()

    print("Saved refinement/wavelet comparison to", outp)


def verify_refinement_and_wavelet(wavelet_name, phi, psi, x, outdir):
    """Verify the refinement equation phi(x) = sqrt(2) * sum_k h_k phi(2x - k)
    and the wavelet reconstruction psi(x) = sqrt(2) * sum_k g_k phi(2x - k).
    Uses interpolation on the sampled phi/psi values and plots comparisons.
    """
    w = pywt.Wavelet(wavelet_name)
    # Use the reconstruction (synthesis) low-pass/high-pass filters.
    # These correspond to the scaling coefficients used in the refinement equation.
    h = np.array(w.rec_lo) * np.sqrt(2)
    g = np.array(w.rec_hi) * np.sqrt(2)
    print(g)
    print(h)

    # interpolation helper: evaluate phi at arbitrary points using linear interp
    def interp(f_vals, x_vals, pts):
        return np.interp(pts, x_vals, f_vals, left=0.0, right=0.0)

    # reconstruct phi from scaled/shifted copies
    subs = []
    recon_phi = np.zeros_like(phi)
    for k, hk in enumerate(h):
        pts = 2 * x - k
        evals = hk * interp(phi, x, pts)
        subs.append(evals)
        recon_phi += evals

    # plot comparisons
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, phi, label="phi(x) = sum of other terms", c="black")
    # plt.plot(x, recon_phi, "--", label="phi (reconstructed)")
    [
        plt.plot(x, sub, linewidth=0.8, label=f"{h_:.4f} * phi(2x-{k})")
        for k, (h_, sub) in enumerate(zip(h, subs))
    ]
    plt.title("Scaling function as linear combination of smaller scaling function")
    plt.legend()

    # reconstruct psi from phi using high-pass (reconstruction) coeffs
    subs = []
    recon_psi = np.zeros_like(psi)
    for k, gk in enumerate(g):
        pts = 2 * x - k
        evals = gk * interp(phi, x, pts)
        subs.append(evals)
        recon_psi += evals

    plt.subplot(1, 2, 2)
    plt.plot(x, psi, label="psi(x) = sum of other terms", c="black")
    [
        plt.plot(x, sub, linewidth=0.8, label=f"{g_:.4f} * phi(2x-{k})")
        for k, (g_, sub) in enumerate(zip(g, subs))
    ]
    # plt.plot(x, recon_psi, "--", label="psi (from phi)")
    plt.title("Wavelet from linear combination of smaller scaling function")
    plt.legend()

    outp = os.path.join(outdir, "refinement_and_wavelet.png")
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()

    print("Saved refinement/wavelet comparison to", outp)


def show_2d_separable_wavelets(phi, psi, x, outdir, upsample=8):
    """Build 2D separable wavelet images from 1D phi and psi samples.
    Creates LL, LH, HL, HH patterns via outer products and saves a figure.
    """
    # trim to a central window for nicer visualization
    L = len(phi)
    start = L // 4
    end = start + L // 2
    p = phi[start:end]
    q = psi[start:end]

    # create 2D patterns via outer product
    LL = np.outer(p, p)
    LH = np.outer(p, q)
    HL = np.outer(q, p)
    HH = np.outer(q, q)

    # upscale for visibility
    def upscale(img, k):
        return np.kron(img, np.ones((k, k)))

    LLu = upscale(LL, upsample)
    LHu = upscale(LH, upsample)
    HLu = upscale(HL, upsample)
    HHu = upscale(HH, upsample)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title("LL (phi x phi)")
    plt.axis("off")
    plt.imshow(LLu, cmap="RdBu", interpolation="nearest")

    plt.subplot(2, 2, 2)
    plt.title("LH (phi x psi)")
    plt.axis("off")
    plt.imshow(LHu, cmap="RdBu", interpolation="nearest")

    plt.subplot(2, 2, 3)
    plt.title("HL (psi x phi)")
    plt.axis("off")
    plt.imshow(HLu, cmap="RdBu", interpolation="nearest")

    plt.subplot(2, 2, 4)
    plt.title("HH (psi x psi)")
    plt.axis("off")
    plt.imshow(HHu, cmap="RdBu", interpolation="nearest")

    outp = os.path.join(outdir, "db2_2d_wavelets.png")
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()
    print("Saved 2D separable wavelet figure to", outp)


def compress_image_wavelet(
    img, wavelet="db2", level=3, keep_fractions=(0.5, 0.1, 0.05, 0.02), outdir="outputs"
):
    # full-wavelet decomposition
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    total = arr.size
    print("\nTotal wavelet coefficients:", total)

    results = []

    for keep in keep_fractions:
        k = max(1, int(np.floor(total * keep)))
        # compute threshold to keep largest-k coefficients by magnitude
        flat = np.abs(arr).ravel()
        if k >= flat.size:
            thr = 0.0
        else:
            thr = np.partition(flat, -k)[-k]

        arr_thr = arr * (np.abs(arr) >= thr)
        coeffs_thr = pywt.array_to_coeffs(
            arr_thr, coeff_slices, output_format="wavedec2"
        )
        rec = pywt.waverec2(coeffs_thr, wavelet=wavelet)
        # crop reconstruction to original shape (waverec2 may output slightly larger array)
        rec = rec[: img.shape[0], : img.shape[1]]

        nonzero = np.count_nonzero(arr_thr)
        comp_ratio = nonzero / total
        psnr = peak_signal_noise_ratio(img, rec, data_range=img.max() - img.min())
        ssim = structural_similarity(img, rec, data_range=img.max() - img.min())

        print(
            f"Keep {keep * 100:.2f}% coeffs -> kept {nonzero}/{total} ({comp_ratio:.4f}) | PSNR={psnr:.2f} dB | SSIM={ssim:.4f}"
        )

        # save reconstructed image
        fname = os.path.join(outdir, f"recon_keep_{int(keep * 1000)}.png")
        plt.imsave(fname, np.clip(rec, 0, 1), cmap="gray")

        results.append(
            {
                "keep_fraction": keep,
                "nonzero": nonzero,
                "total": total,
                "comp_ratio": comp_ratio,
                "psnr": psnr,
                "ssim": ssim,
                "recon_path": fname,
            }
        )

    return results


def make_comparison_figure(img, results, outdir):
    n = len(results) + 1
    plt.figure(figsize=(4 * n, 4))
    plt.subplot(1, n, 1)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(img, cmap="gray")

    for i, r in enumerate(results, start=2):
        rec = plt.imread(r["recon_path"])
        plt.subplot(1, n, i)
        plt.title(
            f"keep={r['keep_fraction'] * 100:.1f}%\nPSNR={r['psnr']:.1f}dB\nSSIM={r['ssim']:.3f}"
        )
        plt.axis("off")
        plt.imshow(rec, cmap="gray")

    outp = os.path.join(outdir, "comparison.png")
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()
    print("Saved comparison figure to", outp)


def main():
    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    # show filters and get sampled phi/psi
    show_db2_filters_and_wavefun(outdir)
    w = pywt.Wavelet("db2")
    phi, psi, x = w.wavefun(level=8)

    # verify refinement equation and wavelet construction
    verify_refinement_and_wavelet("db2", phi, psi, x, outdir)
    show_orthogonal_wavelets("db2", phi, psi, x, outdir)
    exit()

    # show 2D separable wavelets (LL/LH/HL/HH)
    show_2d_separable_wavelets(phi, psi, x, outdir)

    # load sample image
    img = img_as_float(data.camera())
    # normalize to [0,1]
    img = (img - img.min()) / (img.max() - img.min())

    # compression experiments
    results = compress_image_wavelet(
        img,
        wavelet="db2",
        level=3,
        keep_fractions=(0.5, 0.1, 0.05, 0.02),
        outdir=outdir,
    )

    make_comparison_figure(img, results, outdir)

    print("\nSummary:")
    for r in results:
        print(
            f"keep={r['keep_fraction']:.3f}: kept {r['nonzero']}/{r['total']} -> ratio={r['comp_ratio']:.4f}, PSNR={r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
