# Why VAE reconstructions fixed detection on Synthbuster

Nov 26, 2026 · @Druin

## Summary

Adding VAE reconstructions of the COCO real photos to training raised the detector's Synthbuster true positive rate (TPR) from 13% to 81%, while 98% of RAISE real photos were still called real. Before, the model ranked fakes above reals reasonably well (AUC 0.84) but scored almost every Synthbuster image as real. Reconstructions are fakes that share everything with their real photo except the autoencoder's trace, so they stop the model from separating the classes by how the images were stored. The gains are largest where the generator decodes through a latent autoencoder and smallest for GLIDE, which has none.

## Setup

Both runs train the same tri-branch network (CLIP with LoRA, a noise-residual CNN, a spectrum CNN) with one objective, real vs fake, and are tested on Synthbuster, whose nine generators were never seen in training.

| Run | Training COCO IDs | Fake sources | Re-encode before training |
| --- | --- | --- | --- |
| coco10k | 9,000 COCO IDs | LDM, SD 1.5, SDXL-Turbo (DRCT-2M) | every image, JPEG q60–70 |
| coco15k\_recon | 14,000 COCO IDs | the same three + VAE reconstructions of the COCO reals | every image, JPEG q60–70 |

Every COCO ID contributes its real photo and one fake per source, so content is matched across classes. Train, validation and test IDs never overlap, and the 0.5 decision threshold is the one set during training on COCO.

## Results

Overall, Synthbuster AUC rose from 0.838 to 0.965, TPR from 13.1% to 80.7% and balanced accuracy from 0.57 to 0.89, while TNR on RAISE reals dropped only from 100% to 97.9%. Logloss fell from 3.44 to 0.38, now well below the 0.69 of guessing.

&#91;embedded content: eval\_results.json of runs coco10k and coco15k\_recon · Synthbuster, native resolution, 9 generators\]

Every generator improved on both measures. The Stable Diffusion versions now sit at the top; the two pixel-space generators sit at the bottom.

## Why TPR was low before

The first model's scores shifted on Synthbuster: it still ranked fakes above reals (AUC 0.69–0.90 per generator), but nearly all scores fell below 0.5, so TPR was 0.4–30% while TNR was 100%. Logloss of 1.4–2.8 against a guessing baseline of 0.69 shows it was confidently wrong about the fakes.

The likely cause is that the three DRCT fake sets differ from the COCO reals in ways that have nothing to do with generation, and every training fake carried them:

- **Compression history:** every DRCT fake was saved at JPEG quality 75, every COCO photo at about 96. The 60–70 re-encode equalises the final compression, but faint traces of the first save survive.
- **Geometry:** every DRCT fake is a square 256×256 or 512×512 image; COCO photos are mostly 640×480 or 640×427.

With those cues present in 100% of the training fakes, the model could lean on "was stored like a DRCT fake" instead of generation traces. Synthbuster fakes were stored differently (uncompressed PNG), so they lacked the cue and looked real. This explanation fits the numbers but has not yet been confirmed by the fake-history probe in evaluate\_coco.py.

## Why the reconstructions were crucial

A VAE reconstruction is a real COCO photo passed through a latent-diffusion autoencoder (encode, then decode, no denoising). It is labelled fake, yet it keeps the photo's content, framing and 640×480 size, and it never went through DRCT's q75 save. Two things follow:

1. **The storage shortcuts stop working.** Fakes are no longer always square and q75-saved, so those cues no longer separate the classes, and a model that relies on them now misclassifies the reconstructions.
2. **One cue is left that separates every fake from every real: the decoder's trace.** For a real photo and its own reconstruction, the decoder's trace is the only difference by construction, so the model is pushed to learn it.

The per-generator pattern supports this mechanism:

- SD 1.3, 1.4, 2 and XL decode through the same kind of autoencoder and now reach 95–98% TPR.
- Midjourney, DALL·E 3 and Firefly gained 65–83 points of TPR. That is consistent with latent decoders, but their architectures are not public, so this is an inference.
- GLIDE generates directly in pixel space, with no autoencoder, and gained least: 47% TPR, AUC 0.85.
- DALL·E 2 also decodes in pixel space and is second lowest at 64%.

Aligned real/fake pairs are the idea behind Rajan et al. (ICLR 2025), who train on reals and their autoencoder reconstructions, and B-Free (CVPR 2025), which reported about 20 points of balanced accuracy gained by aligned training data.

## JPEG 75 after vs before the re-encode

The two tests apply the same JPEG 75 save to real COCO photos, but at different points in the pipeline, so they test different things. In training, every image is re-encoded at a random quality of 60–70 before it reaches the model. What differs between the classes is what happened to the file before that:

| Image | History before the model sees it |
| --- | --- |
| COCO real | camera → saved at q≈96 → re-encoded at q60–70 |
| DRCT fake | generator → saved at q75 → re-encoded at q60–70 |
| VAE reconstruction | q≈96 photo → autoencoder decode → PNG → re-encoded at q60–70 |

The two shortcut tests then put a q75 save on a real photo:

| Test | Order applied to a real photo | Question it answers |
| --- | --- | --- |
| jpeg\_75 after (first shortcut test) | q≈96 → re-encode q60–70 → save q75 | Is the model robust to one more save on top? |
| as\_fake\_jpeg75 (fake-history probe) | q≈96 → save q75 → re-encode q60–70 | Does the model treat "was saved at q75" as a fake cue? |

**After** changes almost nothing that matters: a q75 save on top of a q60–70 image keeps detail the coarser save already removed, and the image's earlier history is untouched. **Before** gives a real photo exactly the DRCT fakes' history. JPEG traces survive a second save because double compression leaves periodic patterns in the DCT coefficient histograms that depend on both quality tables; the random crop comes after the re-encode, so the two 8×8 grids stay aligned. If real photos given that history are called fake far more often than clean ones, the model learned the storage history rather than generation traces.

## Caveats and next checks

- **Two things changed at once.** The second run added reconstructions *and* went from 9,000 to 14,000 training IDs. A coco15k run without reconstructions would separate the two effects.
- **The shortcut explanation is not yet confirmed.** Running the fake-history probe on both models would test it: `as_fake_jpeg75` and `as_fake_square512` should flag far more real photos as fake under coco10k than under coco15k\_recon.
- **The threshold comes from COCO.** At 0.5 the new model keeps 97.9% of RAISE reals; AUC (0.965 overall) is the threshold-free measure to quote.
- **Four Synthbuster generators are only partly unseen.** SD 1.3 and 1.4 share SD 1.5's autoencoder, and SDXL shares SDXL-Turbo's. DALL·E 2/3, Firefly, Midjourney and GLIDE are the true generalisation test.
- **Open question:** which autoencoder made the reconstructions (SD 2 by default in vae\_reconstruct.py)? It matters for which Synthbuster generators it most directly covers.
