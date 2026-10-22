# Milestone 1 — Project Report

**Project:** SerpentGuard — Snake Identification<br>
**Team:** Group 5 — DS and AI Lab Project<br>
**Report status:** Proposal and literature review<br>
**Date:** 22 October 2026

## 1. Objective and intended use

Build and evaluate a snake-specific system for a defined Indian-region species list. The primary interaction should use a phone camera feed; a saved-photo path can support testing and fallback. The system should:

1. Detect whether an image plausibly contains a snake, then classify the visible snake using a transfer-learned image model.
2. Return the top three candidate species, calibrated confidence (or an explicit “uncertain / outside supported classes” result), common and scientific names, and a short explanation of useful visual cues.
3. Analyze sampled camera frames in real time, smooth predictions across nearby frames, and show a result only when the evidence is stable enough. This is a target for a later prototype, not a feature implemented in this milestone.
4. Run the fine-tuned classifier on-device without a network connection, if model size and measured device performance allow. The camera and photo paths should use the same model and evaluation rules.
5. Optionally use a user-supplied coarse location (state or region) to re-rank candidates by known range. Location must not silently override visual evidence.
6. Be usable as a research/demo tool on ordinary phone photographs and camera frames, including varied backgrounds, lighting, scale, pose, and image quality.

The intended role is **decision support and education**, not a definitive field or clinical identification. A snakebite response must never be delayed or determined by this prototype. Any eventual first-aid information must be reviewed by qualified local medical authorities and shown separately from model predictions.

### Research questions

- How well can a compact pretrained vision model identify the selected regional species on a test split that separates observations, not merely image files?
- Does location information improve macro-F1 and reduce confusion among lookalikes without making the model brittle when location is missing or wrong?
- How much do long-tail balancing, targeted augmentation, and an uncertainty/rejection option improve performance on rare, visually similar, or out-of-scope snakes?
- Can the resulting model deliver useful top-k suggestions with acceptable latency on the intended hardware?
- Can an on-device model provide stable camera predictions with useful latency while remaining offline?

## 2. Problem and existing evidence

Snake recognition is a fine-grained visual classification problem: markings change with age and geography, similar-looking species can differ medically, and photos may show only part of an animal or include distracting backgrounds. Dataset imbalance is substantial. SnakeCLEF 2021 describes 414,424 photos across 772 species and 188 countries; the most represented species had 22,163 images while one species had only 10. Its dataset included online biodiversity observations and noisy Flickr examples, plus an expert-identified private test set. The challenge explicitly warns about age- and location-dependent morphology and venomous/non-venomous lookalikes. [SnakeCLEF 2021 challenge](https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge)

A published global Vision Transformer study trained on 386,006 photos covering 772 species from 188 countries. The dataset included 198 medically important venomous species and 574 other species. The authors report **96.0% top-1 accuracy**, **92.2% species macro-F1**, **99.0% genus top-1 accuracy**, and **94.2% country macro-F1** on a 23,673-image test set. Geographic filtering improved country F1 and species F1 by 2.93 and 3.11 percentage points respectively in their experiment, and the authors report a 33.3% relative error reduction from incorporating geographic metadata. These are strong research results on that study’s benchmark, not a promise of performance on a new Indian pilot or phone photos. [PLOS study: methods and results](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010647)

The same study exposes important caveats: the data are long-tailed; only 772 of roughly 3,921 known species met the ten-image training threshold; 77% of photos came from the United States and Canada; and test images from iNaturalist may be easier than urgent field images (blur, partial bodies, or poor lighting). The authors call for field testing. Country-level F1 exceeded 90% in 88% of countries but fell as low as 36% in Martinique, illustrating that global averages can hide regional weakness. [PLOS study: dataset coverage and limitations](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010647)

An earlier challenge focused on 45 North American species reported 84% accuracy and F1 0.83 on a post-challenge iNaturalist set; on expert-labelled private photos its reported accuracy was 72% across 27 represented species (and F1 0.73). Performance varied by species, and human participants did better on some poor-quality images. This is useful evidence that curated/online test results may overstate real-world reliability. [Durso et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8093445/)

SnakeCLEF 2024 shifted attention toward medically important scenarios and evaluated generalization in India, Central America, and southern Africa, including asymmetric error costs such as a venomous snake predicted harmless. This suggests that raw top-1 accuracy alone is insufficient for the proposed project. [SnakeCLEF 2024 overview](https://dspace.zcu.cz/items/64e8cf26-55b0-42a1-bc2f-ac6da22c5173)

## 3. Data samples and proposed data plan

The sample panel below is from the augmentation figure in the open-access PLOS snake-recognition study. It shows two underlying snake photographs (top and bottom rows) alongside transformed versions used to illustrate training augmentation. The original figure caption credits the source observations to chiuluan (iNaturalist 207060926) and Alex Karasoulos (iNaturalist 207359674), both marked CC-BY in the article. These examples demonstrate variation in pose, framing, background, and capture conditions; they are **illustrative literature samples, not a claim that these files are already in this project’s training set**. [Original Figure 2 and attribution](https://journals.plos.org/plosntds/article/figure?id=10.1371/journal.pntd.0010647.g002)

![Two snake source photographs and augmented views from the PLOS study's Figure 2](assets/snakeclef_samples.png)

**Planned data sources:** begin with legally usable, research-grade iNaturalist observations and/or a released SnakeCLEF subset, after checking dataset terms, image licenses, taxonomic labels, and geographic coverage. For an India-focused pilot, prioritize species confirmed in the chosen target states, and supplement sparse species only with expert-reviewed records. Maintain source URL/observation ID, license, label provenance, and location uncertainty for each image. Deduplicate near-identical images and group all images from one observation/photographer where feasible before splitting.

**Data risks to address:** mislabeled or disputed taxa; rare classes; geographic and platform bias; repeated images of the same individual; captive snakes; class overlap and taxonomic updates; photos containing multiple snakes; and leakage when near-duplicates cross train/test splits. Keep a separate out-of-scope/unknown set and a hard, expert-reviewed lookalike subset.

## 4. Existing solutions and comparison

| Option | What it offers | Strengths | Limitations relevant to this project |
|---|---|---|---|
| Google Search / Google Lens | Camera or uploaded image starts a visual search and returns visually similar pages and information; Lens can identify plants and animals | Fast, broad index and useful source discovery; works from a camera or image | Search matches are not a calibrated, fixed-label snake classifier. Results can vary with the web and do not provide a project-specific measured abstention or risk score. [Google Lens overview](https://blog.google/products-and-platforms/devices/google-lens/google-lens-features/), [Search by image help](https://support.google.com/websearch/answer/1325808) |
| Gemini Live | Users can share a camera feed and ask spoken questions about what it sees | Natural, interactive live-camera assistance and general visual reasoning | General-purpose responses are not a fixed, regionally evaluated snake classifier; availability, connectivity, and response behavior depend on the service. [Gemini Live camera guide](https://blog.google/products-and-platforms/products/gemini/gemini-live-camera-how-to/) |
| ChatGPT camera/video | Advanced Voice supports camera/video sharing on supported mobile experiences; the newer Live voice experience has different capabilities | Conversational follow-up and broad image understanding | Not trained or calibrated for this project's fixed regional class set; product mode and access vary. A direct answer can sound confident without a matched snake-specific validation result. [ChatGPT Voice FAQ](https://help.openai.com/en/articles/20001274-gpt-live-faq), [Voice with video](https://m.chatgpt.com/features/voice-with-video/) |
| Direct response from a general LLM | User describes the snake, or supplies an image where supported, and asks for a name | No dedicated app or dataset setup; can explain its reasoning conversationally | Depends on the description or image and the model's general knowledge; no guaranteed label set, reproducible score, or reliable way to distinguish uncertainty from a plausible-sounding guess. It is a useful qualitative comparator, not a ground-truth system. |
| Global SnakeCLEF/PLOS ViT research model | Global, fine-grained image classification with optional geography | Published large-scale benchmark; 772 species and strong reported accuracy and macro-F1 | Benchmark performance does not transfer automatically to an Indian field set; long-tail and field-image gaps remain. Model, data, and deployment terms need review. |
| SnakeCLEF challenge systems | Shared datasets and held-out evaluation; 2024 addresses medically important scenarios and India | Reproducible research framing; includes domain shift and risk-sensitive evaluation. DINOv2 feature extraction was explored for SnakeCLEF 2024, with code released by the authors. [2024 study and code](https://arxiv.org/abs/2407.06178) | Competition systems are not automatically consumer products; datasets and deployment constraints differ. |
| Prior mobile and real-time snake systems | Research prototypes include offline mobile classification and video-based recognition; a recent Serpent-ID paper reports YOLOv8-nano, TensorFlow Lite, and int8 quantization for real-time mobile use | Establish that offline/mobile and live video snake recognition are not new ideas | Species coverage, data, independent validation, safety behavior, and code availability vary. These are direct precedents, so the project must compare against them instead of claiming novelty for camera input alone. [Australian offline/online iOS study](https://aisel.aisnet.org/hicss-54/da/mobile_services/3/), [video-processing system](https://rda.sliit.lk/entities/publication/a7482c0f-95a5-441d-8340-ba9750477b44), [Serpent-ID](https://ieeexplore.ieee.org/abstract/document/11507553) |
| Proposed Group 5 prototype | Region-scoped, fine-tuned on-device classifier that analyzes sampled live-camera frames, alongside a photo fallback | Aims to combine local species coverage, no required LLM/API call, consistent predictions over time, top-k output, and explicit abstention | Still a proposal. Feasibility depends on obtaining enough licensed, expert-checked local data and meeting latency/model-size targets on selected phones. No project model or measured baseline exists yet. |

**What we would do differently:** camera input is a usability choice, not our novelty claim: Lens, Gemini Live, ChatGPT camera features, and prior snake prototypes already cover parts of that experience. Our proposed contribution is to fine-tune and deploy a compact model for a clearly bounded Indian species list, run classification locally on sampled frames, and combine temporal stability with a calibrated reject option. We would test it against photo-only inference, the same model without temporal smoothing, and accessible alternatives on the same locked images/clips. If data rights, expert labels, or device benchmarks make the offline plan infeasible, we will report that and narrow the claim rather than substitute an LLM API and call it our core model. |

### Proposed live-camera and offline design

The camera stream should not be sent continuously to a language-model service. A lightweight snake-presence detector can screen frames; the local species model then evaluates selected frames or a tracked crop at a controlled rate. A short rolling window can combine class scores and suppress flickering labels. The interface should show “snake not found,” “uncertain,” or “possible match” when appropriate, and avoid presenting a species as safe to handle. A still-image mode should remain available for reproducible evaluation and phones with limited compute.

The candidate model path is transfer learning from a compact image backbone, fine-tuned on the target species list, then exported and benchmarked in a mobile runtime such as TensorFlow Lite or ONNX Runtime Mobile. Quantization is an option only after checking its effect on rare classes and confidence calibration. Training may need a stronger backbone; the exported model should be chosen from measured accuracy, macro-F1, model size, memory, battery use, and p50/p95 latency on agreed devices. The detector and classifier can be separate initially; combining them is a later optimization, not a milestone assumption.

Before implementation, the team must select target phones and species, confirm dataset licenses and expert review, and set a frame rate and latency target. Evaluate on held-out clips grouped by encounter/observation, not random adjacent frames. Compare a single sampled frame with temporal aggregation, and include motion blur, partial visibility, camera movement, and no-snake scenes. Report false detections and stability as well as classification metrics. The repository currently has no dataset, training code, model, or live-camera implementation, so all of this remains proposed work.

## 5. Baselines and target improvements

### Published external baselines

| Reference | Scope/test set | Reported result | How to interpret it |
|---|---|---|---|
| Durso et al. / SnakeCLEF-era EfficientNet pipeline | 45 North American species; online and private expert-labelled test sets | 84% accuracy / 0.83 F1 on post-challenge iNaturalist; 72% accuracy / 0.73 F1 on 27-class private set | Strong early specialist baseline; private-image performance was lower |
| Global ViT study | 772 species; 23,673-image independent test | 96.0% top-1 accuracy; 92.2% species macro-F1; 94.2% country macro-F1 | Current literature reference found for broad global snake classification; not directly comparable to a new regional split |
| SnakeCLEF 2021 challenge data | 772 species; 188 countries | 414,424 images total; long-tailed class sizes of 10–22,163 | Benchmark scale and imbalance context, not itself a single model score |

### Project baseline (to be measured)

There is **no measured project baseline yet**: the repository has no dataset, training code, model, or evaluation split. At the next milestone, establish a reproducible baseline with (1) majority-class prediction, (2) a pretrained ResNet/EfficientNet classifier fine-tuned on the same training set, and (3) the same model plus location metadata. Record the split manifest, class counts, software/model versions, and random seed. Do not fill this table with invented numbers.

| Metric | Current project value | Measurement planned |
|---|---:|---|
| Top-1 accuracy | Not measured | Fraction of held-out images whose first prediction matches the reference label |
| Top-3 accuracy | Not measured | Fraction where reference label appears in the first three suggestions |
| Macro-F1 | Not measured | Mean per-class F1; reflects rare classes |
| Balanced accuracy | Not measured | Mean per-class recall |
| Calibration / selective risk | Not measured | Reliability or ECE plus error rate after abstaining below a confidence threshold |
| Dangerous confusion | Not measured | Per-class and weighted errors, especially venomous → non-venomous predictions for target region |
| Latency / model size | Not measured | Median and p95 prediction time on target hardware; artifact size |

**Improvement plan:** (a) use transfer learning and train/validation class-aware sampling or loss weighting; (b) augment realistic lighting, crop, rotation, blur, and compression without destroying diagnostic pattern cues; (c) evaluate a snake detector/crop stage against full-image classification; (d) test a location-aware re-ranker only when location is present, with geographic leakage controlled; (e) calibrate probabilities and add an unknown/abstain pathway; (f) obtain expert review for rare classes and high-cost confusions. Compare each change through controlled ablations on the exact same held-out data.

Targets should be set after collecting the first reliable split. A reasonable milestone objective is statistically better macro-F1 and top-3 recall than the pretrained regional baseline while reducing dangerous confusion, with no material regression on common species. The global ViT’s 96% top-1 and 92.2% macro-F1 are reference points, not promises or pass thresholds for a different dataset.

## 6. Evaluation against the best available options

Use a **locked, observation-level test set** with expert-verified labels, class counts, region, source, and image-quality strata. Split by observation and, where possible, contributor/individual to prevent near-duplicate leakage. Keep a second external set from a different source or collection period for a more realistic distribution-shift test.

1. Run the project baseline and every proposed model on identical images and label mappings.
2. Where downloadable/reusable checkpoints or accessible tools permit, run the published global model and/or a SnakeCLEF 2024 reference solution on the same supported classes. For iNaturalist or other user-facing tools, record the returned ranked suggestions on the same images, with tool/version/date, location settings, and missing/unsupported cases. If direct execution is unavailable, mark that comparison unavailable rather than inferring a score from a paper.
3. Report top-1 and top-3 accuracy, macro-F1, per-class precision/recall, balanced accuracy, confusion matrices, calibration, coverage-versus-risk for abstention, and latency/model size. Provide bootstrap confidence intervals and group results by common/rare classes, region, quality, and medically important status.
4. Predefine an asymmetric error analysis. Count venomous-to-nonvenomous confusions separately; report sensitivity/recall for medically important species and a cost-weighted score alongside standard metrics. Never collapse this into a single accuracy claim.
5. Test location use with three settings: image only, image plus correct coarse location, and image plus missing/incorrect location. This reveals whether geography helps and whether it can harm.
6. For the live workflow, use held-out clips and report clip-level top-1/top-3 correctness, time to first stable result, label changes per minute, false detections, and abstention coverage. Compare sampled-frame prediction with temporal aggregation. Do not treat adjacent frames as independent test examples.
7. Compare Google Lens/Search, Gemini Live, and ChatGPT Advanced camera/video on the same safe, consented test images or clips where their interfaces permit. Record exact prompt, product mode, date, location setting, top returned species/pages, response time, and whether the tool abstains. Treat conversational LLM results as qualitative if the service does not expose ranked, reproducible predictions; do not infer a score from anecdotal examples. A direct text-only LLM response should be a separate condition based on the same written description.
8. Have herpetology reviewers inspect a stratified sample of errors, uncertain predictions, and high-risk predictions. Compare model output with expert labels and, where feasible, independent human identification on the same blinded test examples.

**Market comparison caveat:** iNaturalist/Seek and regional services are useful practical comparators, but their coverage, models, and interfaces change over time and they do not publish a directly comparable snake-only score on this project’s proposed test set. A fair “best available” claim requires executing the tools against the same locked images; until then, the report treats the literature ViT and SnakeCLEF as research baselines and consumer tools as workflow references.

## 7. Expected product outputs (illustrative mockups)

These are mock outputs for the proposed interface, not predictions produced by a trained model.

### Example A — confident, in-scope photo

```text
┌──────────────────────────────────────────────┐
│ Snake image analyzed                          │
│                                              │
│ 1. Candidate species A                 0.82  │
│ 2. Candidate species B                 0.11  │
│ 3. Candidate species C                 0.04  │
│                                              │
│ Region: user-selected state                   │
│ Visual cues: band pattern; head/body shape    │
│ Confidence: review recommended                │
│ [View species notes] [Report a correction]    │
└──────────────────────────────────────────────┘
```

A real release should show verified common/scientific names and concise, sourced distinguishing features. Percentages are placeholders and must not be read as actual model confidence.

### Example B — uncertain or unsupported image

```text
┌──────────────────────────────────────────────┐
│ No reliable species-level match               │
│                                              │
│ Best supported level: Genus / Unknown         │
│ Top candidates are visually similar.          │
│ Confidence is below the display threshold.    │
│                                              │
│ Please do not handle the snake.               │
│ [Retake from a safe distance] [See guidance]  │
└──────────────────────────────────────────────┘
```

The “guidance” content requires local expert/medical review. The model should never present a low-confidence species guess as certainty.

## 8. Gaps, opportunities, and project response

| Gap in existing work | Opportunity | Planned response |
|---|---|---|
| Long-tailed data and weak coverage for rare species | Use class-aware evaluation and seek expert-verified regional additions | Track per-class sample counts; avoid presenting macro performance alone; focus curation on rare target species |
| Geographic bias and domain shift | Evaluate a bounded local use case with a second-source test set | Define the regional list, keep location as an optional feature, measure cross-region/source performance |
| Similar-looking species with different risk | Make confusion costs explicit | Report venomous → non-venomous errors and medically important recall; use abstention when ambiguous |
| Online benchmark images may be cleaner than incident photos | Curate hard, real-world examples | Add blurred, partial, low-light, and cluttered images with expert labels; test separately |
| High headline accuracy can mask rare-class failure and overconfidence | Calibrate and communicate uncertainty | Include macro-F1, class-level results, calibration, top-k candidates, and unknown detection |
| Global research model may be difficult to adapt/deploy locally | Build a reproducible, appropriately sized student prototype | Compare efficient pretrained backbones, model size, and inference time on agreed hardware |
| Consumer tools do not offer directly comparable snake-specific scores | Run matched-image comparative evaluation | Freeze test images and tool settings; report date/version and unsupported predictions |
| Camera access is already available in general assistants and earlier snake prototypes | Make the differentiator the local model and measurable behavior, not the camera button | Fine-tune a regional model; target offline inference and temporal stability; compare systems on the same images/clips |
| Live video can produce flickering or confident errors, especially in poor conditions | Evaluate prediction stability and abstention across a clip | Measure per-clip correctness, time-to-stable-result, false detections, and reject coverage under motion/occlusion |
| A hosted LLM API adds network dependence and recurring service cost | Keep the core species prediction independent of a hosted LLM | Run the exported classifier on device; any future language interface is optional and not the species-identification authority |

## 9. Milestone 1 conclusion and next work

Published work establishes that snake classification has been attempted at global scale and in mobile, offline, and video prototypes. General-purpose products also offer visual search and live camera interaction, so camera input by itself is not a new contribution. A credible project direction is a regional model fine-tuned for a defined Indian species list, with a feasible on-device path, stable clip-level results, and explicit uncertainty. Those benefits remain hypotheses until the team builds and tests the model on licensed, independently reviewed data and representative phones.

**Immediate next steps:** confirm target geography/species, user, and phone hardware; check dataset licenses and expert-label availability; define medically important taxonomy and label policy; create grouped image and clip splits; build a photo baseline; then benchmark an on-device export and add camera sampling/temporal smoothing only after its evaluation criteria are fixed.

## References

1. Picek et al. (2021), [SnakeCLEF 2021 challenge description and dataset](https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge).
2. Picek et al. (2022), [An artificial intelligence model to identify snakes from across the world](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010647), *PLOS Neglected Tropical Diseases*.
3. Durso et al. (2021), [Supervised learning computer vision benchmark for snake species identification from photographs](https://pmc.ncbi.nlm.nih.gov/articles/PMC8093445/), *Frontiers in Artificial Intelligence*.
4. Picek et al. (2024), [Overview of SnakeCLEF 2024: Revisiting Snake Species Identification in Medically Important Scenarios](https://dspace.zcu.cz/items/64e8cf26-55b0-42a1-bc2f-ac6da22c5173).
5. iNaturalist, [Computer Vision FAQ](https://www.inaturalist.org/posts/47454-faq) and [Seek app information](https://www.inaturalist.org/pages/seek_app).
6. Google, [Google Lens features](https://blog.google/products-and-platforms/devices/google-lens/google-lens-features/) and [Search by image help](https://support.google.com/websearch/answer/1325808), accessed 22 October 2026.
7. Google, [Gemini Live camera guide](https://blog.google/products-and-platforms/products/gemini/gemini-live-camera-how-to/), accessed 22 October 2026.
8. OpenAI, [ChatGPT Voice FAQ](https://help.openai.com/en/articles/20001274-gpt-live-faq) and [Voice with video](https://m.chatgpt.com/features/voice-with-video/), accessed 22 October 2026.
9. Divya et al. (2021), [Snake Detection and Classification using Deep Learning](https://aisel.aisnet.org/hicss-54/da/mobile_services/3/), describing offline on-device and cloud-assisted iOS prototypes.
10. [Smart Snake Identification System using Video Processing](https://rda.sliit.lk/entities/publication/a7482c0f-95a5-441d-8340-ba9750477b44), project record describing offline and online video recognition.
11. [Serpent-ID: A Deep Learning-Based Mobile Application for Rapid Snake Identification and Bite Management Guidance](https://ieeexplore.ieee.org/abstract/document/11507553), IEEE Xplore record; reported implementation details should be checked against the full paper before using them as a benchmark.
12. [Transfer Learning with Self-Supervised Vision Transformers for Snake Identification](https://arxiv.org/abs/2407.06178) and the authors' [SnakeCLEF 2024 implementation](https://github.com/dsgt-kaggle-clef/snakeclef-2024).

<div style="page-break-before: always"></div>

# Team review and acknowledgment

**This is the final page of the report. Each team member must complete and sign one row before submission.** By signing, each member confirms that they have reviewed the submitted document, understand its contents, and agree with the submission.

| Team member name | Roll Number | Signature / acknowledgment | Date |
|---|---|---|---|
|Jayesh Bansal|24f100189|![Jayesh Bansal](https://drive.google.com/uc?export=download&id=1imepTm_t9vgJVL367tWU-LPYcLq9Zet0)|22/10/2026|
|Vikas |23f3001800|Vikas |22/10/2026|
|Jayesh Bansal|24f100189|Pending acknowledgment|22/10/2026|
|  |  |  |  |
|  |  |  |  |
