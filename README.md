# CLTW: Chinese Long Text in the Wild

Official repository of the **CLTW** dataset, a benchmark for **Chinese long-text recognition in complex real-world street scenes**.


> **Update Notice**
> The updated version of the CLTW dataset and the results of the supplementary degradation severity analysis are currently being uploaded.
> At this moment, the files available in **GitHub Releases** correspond to a previous version of the dataset, whose directory structure has not yet been updated.

## Quick Links

- **Dataset download**: [GitHub Releases](https://github.com/wangah1007/CLTW/releases)
- **Supplementary material**: [supplementary.pdf](./supplementary.pdf)

## Overview

CLTW is a Chinese scene text dataset and benchmark designed for **long-text recognition in complex real-world environments**.

Compared with existing Chinese scene text benchmarks dominated by short words or short text lines, CLTW focuses on **text-heavy street-scene images** with substantially longer textual content and more severe real-world degradations.

The dataset supports the evaluation of both:

- **conventional OCR models**
- **multimodal large language models (MLLMs)**

under a unified long-text recognition benchmark.

## Dataset Statistics

- **Total images**: 14,183
- **Total text lines**: 140,985
- **Total characters**: 2,844,932
- **Train / Val / Test split**: 9,922 / 1,429 / 2,832

### Text-load Statistics

- **Chars / image**: median 145, mean 200.6, max 3600
- **Chars / line**: median 19, mean 20.2, max 154
- **Avg. lines / image**: 9.94
- **Chinese character ratio**: 82.6%

## Categories

CLTW covers eight representative scenario / degradation categories:

- **REF**: Reflective Scene Text
- **UND**: Underexposed Scene Text
- **PER**: Perspective-distorted Scene Text
- **SSB**: Small-scale & Blurry Scene Text
- **OCC**: Occluded Scene Text
- **SCR**: Screen-based Scene Text
- **CBG**: Complex-background Scene Text
- **SPM**: Special-material Scene Text

Among them, five categories are organized with a **degraded-normal paired design**:

- REF
- UND
- PER
- SSB
- OCC

The remaining three are carrier-related categories:

- SCR
- CBG
- SPM

## Annotation and Split

Each text instance is annotated at the **text-line level** with:

- a **four-point quadrilateral bounding box**
- the corresponding **text transcription**

For paired degraded-normal samples, both images in the pair are assigned to the **same subset** to avoid cross-subset leakage.

## Download

The CLTW dataset is released through **GitHub Releases**:

- [CLTW Releases](https://github.com/wangah1007/CLTW/releases)

If the dataset is distributed in multiple archive parts, please download **all parts** and extract them into the same directory.

## Supplementary Material

The supplementary document for the CLTW paper is available here:

- [supplementary.pdf](./supplementary.pdf)

## Supplementary Severity Analysis Code

We also provide source code for a supplementary severity-aware analysis pipeline over the eight degradation categories in CLTW.

This code is intended for fine-grained post-hoc analysis and can be used to compute continuous image-level severity descriptors within each degradation dimension for sample ranking, stratified evaluation, and robustness analysis.

- Source code: [`project/`](./project/)
- Usage details: [`project/README.md`](./project/README.md)

Please note that these scores are provided as supplementary severity descriptors for within-dimension analysis and do not replace the original discrete category annotations of CLTW.
}
