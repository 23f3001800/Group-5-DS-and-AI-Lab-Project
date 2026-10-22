# Milestone 1: Project Report

Project: SerpentGuard AI, snake detection and classification<br>
Team: Group 5, DS and AI Lab Project. Team lead Jayesh Bansal; members Vikas, Puneet Bajaj and Manas Rastogi<br>
Status: problem definition, literature review and proposed plan. No model has been trained yet.<br>
Date: 22 October 2026

## 1. Problem statement

Snakebite kills far more people in India than official records show. The Registrar General of India's Million Death Study estimated 46,000 snakebite deaths in 2005 (99% CI 41,000 to 51,000), more than 20 times the official figure [15]. Most encounters with snakes end without a bite, but they often end in panic, in a harmless snake being killed, or in someone walking up to a dangerous snake they took for a harmless one. In the moment, most people cannot tell a spectacled cobra from a rat snake or a common krait from a wolf snake, and these lookalikes carry very different risks.

SerpentGuard AI is a two-stage vision system. It finds a snake in a photo or a live camera feed, suggests the species from a fixed list of 27 Indian species, and says when it is unsure. We intend it for decision support and education. It does not diagnose bites or recommend treatment.

### 1.1 Scope and boundaries

| In scope | Out of scope |
|---|---|
| The 27 Indian snake species with at least 200 research-grade, CC-licensed iNaturalist observations each (section 3.1), including all of India's "Big Four" | Species outside the list, which the system reports as "unknown / not supported" |
| Two input modes that share the same models: a still photo, and a live camera feed processed frame by frame | Bite diagnosis, dosing, treatment advice, or any claim that an area is safe |
| Top-3 species, calibrated confidence, an explicit "uncertain" state, and each candidate's WHO medical-importance category | Generated (LLM) text. All safety text is fixed and quoted from official sources |
| Optional coarse location (Indian state) used to re-rank candidates | Letting location override what the image shows |
| A web demo on Hugging Face Spaces for the course, and an offline on-device export benchmarked on one phone | A mobile app published to an app store |

### 1.2 Stakeholders

| Stakeholder | Interest | How they use or judge the system |
|---|---|---|
| People in snake-prone homes, farms and campuses | Knowing whether to keep away and whom to call | Primary users, pointing a phone at the snake from a safe distance |
| Snake rescuers and forest department staff | Faster triage of rescue calls from photos sent by the public | Users of the photo mode and a source of real-world feedback |
| Doctors and first responders | Context only, since species ID never replaces clinical assessment | Not users of the tool, which is one reason diagnosis is out of scope |
| Herpetologists and citizen-science communities (iNaturalist) | Accurate labels and fewer harmless snakes killed | Source of labels and possible reviewers of hard cases |
| Course team (instructors and TAs) | A reproducible end-to-end deep learning project | Assess each milestone against the measurable objectives |
| Our team | Hands-on work in detection, fine-grained classification, calibration and edge deployment | Builders |

### 1.3 Objectives and measurable targets

These targets are proposals. We will test them at Milestone 5 on a locked test set grouped by observation and by observer (section 3.3), and revisit them once the Milestone 2 split exists. If we change a target, the report will say what changed and why.

| # | Objective | Metric | Target (proposal) | Basis |
|---|---|---|---|---|
| O1 | Identify the 27 species | Macro-F1 / top-3 accuracy | ≥ 0.80 / ≥ 0.93 | A 45-species benchmark reported F1 0.83 [3]. The global ViT reached 92.2% macro-F1 on 772 species with about 386k photos [2] |
| O2 | Catch dangerous snakes | Recall on the 7 WHO Category 1 and 2 species, counted as "medically important" | ≥ 0.95 | Safety goal, using the WHO SEARO 2016 category list [15] |
| O3 | Avoid the most harmful error | Share of medically important images predicted as a not-listed species (non-abstained predictions only) | ≤ 2% | Asymmetric error cost, as in SnakeCLEF 2024 [4] |
| O4 | Know when it does not know | Expected calibration error after temperature scaling; error at 80% coverage; share of out-of-list snakes flagged "unknown" | ECE ≤ 0.05; error ≤ 5%; ≥ 70% flagged | Calibration method [20] |
| O5 | Find the snake first | Detector precision and recall at IoU 0.5 | ≥ 0.90 each | Prior YOLO work reported 87% to 88% precision [13, 14] |
| O6 | Work on live video | Pipeline speed; label flips per 10-second clip; time to first stable label | ≥ 10 FPS on a laptop CPU; ≤ 1 flip; ≤ 2 s | Usability goal for the live demo |
| O7 | Run offline on a phone (stretch) | Size of both exported int8 models; median latency per frame on one mid-range Android phone | ≤ 25 MB; ≤ 150 ms | Edge goal, measured on the device |
| O8 | Use location safely | Macro-F1 change with the correct state, and with a missing or wrong state | ≥ +2 points / no worse than −1 point | The global study gained about 3 points from geography [2] |

### 1.4 Research questions

- How well does a compact pretrained vision model identify the 27 species on a test set that keeps observations and observers apart, so no snake or photographer appears in both training and test?
- If we run the same image model on each frame of a live feed and add tracking and smoothing over time, are the labels stable and correct under motion, blur and occlusion?
- Does calibration plus a reject option keep dangerous-species errors low without abstaining on most images?
- Does coarse location help with lookalikes, and does it hurt when it is missing or wrong?
- Can the detector and classifier meet the latency and size targets once exported for on-device use?

## 2. Problem and existing evidence

Snake recognition is a fine-grained visual classification problem. Markings change with age and geography, similar-looking species can differ medically, and photos may show only part of an animal or include distracting backgrounds. The data are also heavily imbalanced. SnakeCLEF 2021 describes 414,424 photos across 772 species and 188 countries; the most represented species had 22,163 images and one species had only 10. Its dataset included online biodiversity observations and noisy Flickr examples, plus an expert-identified private test set, and the challenge warns about age- and location-dependent morphology and venomous/non-venomous lookalikes [1].

A global Vision Transformer study (Bolon, Picek et al. 2022) trained on 386,006 photos covering 772 species from 188 countries, including 198 medically important venomous species. The authors report 96.0% top-1 accuracy, 92.2% species macro-F1, 99.0% genus top-1 accuracy and 94.2% country macro-F1 on a 23,673-image test set. Adding geographic information improved country F1 and species F1 by 2.93 and 3.11 points, which the authors describe as a 33.3% relative error reduction. They released the trained model (ONNX) and data under a BSD-3-Clause licence, with code on GitHub [2], so we can run it on our own test images.

The same study points out its limits. Only 772 of roughly 3,921 known species met the ten-image training threshold, 77% of photos came from the United States and Canada, and iNaturalist test images may be easier than urgent field photos with blur, partial bodies or poor lighting. Country-level F1 exceeded 90% in 88% of countries but fell to 36% in Martinique, so a high global average can hide weak regions [2].

An earlier challenge on 45 North American species reported 84% accuracy and F1 0.83 on a post-challenge iNaturalist set. On expert-labelled private photos, accuracy was 72% across the 27 species represented (F1 0.73). Performance varied by species, and human participants did better on some poor-quality images, which suggests curated online test sets overstate real-world reliability [3].

SnakeCLEF 2023 and 2024 moved to medically important scenarios. They evaluated generalization in India, Central America and southern Africa and weighted errors unevenly, so a venomous snake predicted as harmless costs more [4]. Top-1 accuracy alone is therefore not enough for this project.

For detection, two studies report YOLO-based snake detectors. Naresh et al. (2024) report 87% detection precision [13]. Najm et al. (2025) trained YOLOv8-Nano on 8,500 images of 10 species and report precision 0.88, recall 0.91 and mAP@0.5 0.892 at 142 FPS in their Table 1. Their abstract gives 92.7 mAP@0.5, which does not match that table, so we quote the table [14]. Neither paper reports calibrated confidence or abstention.

## 3. Data samples and data plan

The sample panel below is the augmentation figure from the open-access PLOS snake-recognition study. It shows two snake photographs (top and bottom rows) next to transformed versions used for training augmentation. The original caption credits the source observations to chiuluan (iNaturalist 207060926) and Alex Karasoulos (iNaturalist 207359674), both marked CC-BY in the article. They are examples from the literature, not files from our training set. [Original Figure 2 and attribution](https://journals.plos.org/plosntds/article/figure?id=10.1371/journal.pntd.0010647.g002)

![Two snake source photographs and augmented views from the PLOS study's Figure 2](assets/snakeclef_samples.png)

### 3.1 How much data is available

We queried the public iNaturalist API on 22 October 2026 [16] for research-grade snake observations in India (place 6681, taxon 85553, community-agreed species ID) whose photos carry CC0 or CC-BY family licences. That returned 24,092 observations across 273 species. A random sample of 200 observations averaged 1.99 photos each, and 23 of the 200 (12%) had obscured coordinates.

Twenty-seven species have at least 200 such observations, 18,413 in total or roughly 36,000 images. These 27 species are our class list. The other 246 species, about 5,700 observations, give us a ready-made out-of-list "unknown" set.

To check the method, we queried a US-only rattlesnake (*Crotalus atrox*), which returns 0 Indian records, and re-queried *Naja naja* by taxon id, which reproduced its count exactly. Queries must use taxon ids: a name search for "Naja naja" returned a damselfly.

| # | Species | Common name | CC obs. | WHO category in India [15] |
|---|---|---|---:|---|
| 1 | *Fowlea piscator* | Checkered keelback | 2,611 | not listed |
| 2 | *Ptyas mucosa* | Oriental rat snake | 2,000 | not listed |
| 3 | *Naja naja* | Spectacled (Indian) cobra | 1,505 | Category 1 |
| 4 | *Craspedocephalus malabaricus* | Malabar pit viper | 1,120 | Category 2 |
| 5 | *Amphiesma stolatum* | Buff-striped keelback | 903 | not listed |
| 6 | *Daboia russelii* | Russell's viper | 842 | Category 1 |
| 7 | *Craspedocephalus gramineus* | Bamboo pit viper | 714 | Category 2 |
| 8 | *Indotyphlops braminus* | Brahminy blind snake | 696 | not listed |
| 9 | *Coelognathus helena* | Common trinket snake | 691 | not listed |
| 10 | *Lycodon aulicus* | Indian wolf snake | 670 | not listed |
| 11 | *Python molurus* | Indian rock python | 619 | not listed |
| 12 | *Dendrelaphis tristis* | Common bronzeback | 592 | not listed |
| 13 | *Bungarus caeruleus* | Common krait | 527 | Category 1 |
| 14 | *Echis carinatus* | Saw-scaled viper | 525 | Category 1 |
| 15 | *Ahaetulla oxyrhynca* | Indian vine snake | 487 | not listed |
| 16 | *Ahaetulla borealis* | Northern Western Ghats vine snake | 472 | not listed |
| 17 | *Hypnale hypnale* | Hump-nosed pit viper | 469 | Category 1 (south-west) |
| 18 | *Oligodon taeniolatus* | Streaked kukri snake | 466 | not listed |
| 19 | *Eryx conicus* | Rough-scaled sand boa | 425 | not listed |
| 20 | *Lycodon travancoricus* | Travancore wolf snake | 365 | not listed |
| 21 | *Ahaetulla farnsworthi* | Farnsworth's vine snake | 276 | not listed |
| 22 | *Lycodon fasciolatus* | Russell's wolf snake | 271 | not listed |
| 23 | *Sahyadriophis beddomei* | Nilgiri keelback | 246 | not listed |
| 24 | *Boiga trigonata* | Common cat snake | 240 | not listed |
| 25 | *Oligodon arnensis* | Banded kukri snake | 237 | not listed |
| 26 | *Rhabdophis plumbicolor* | Green keelback | 229 | not listed |
| 27 | *Eryx johnii* | Red sand boa | 215 | not listed |

"Not listed" means the species is not in WHO's Category 1 or 2 for India. It does not mean the snake is harmless to handle, and the interface will never say so. WHO also notes that bites by the non-venomous checkered keelback (our class 1) are common and that medical staff sometimes mistake them for venomous bites [15].

### 3.2 Sources by task

| Task | Source | Licence / terms | Use |
|---|---|---|---|
| Species classifier | iNaturalist research-grade observations from India with CC-licensed photos [16] | Set per photo: CC0, CC-BY or CC-BY-NC(-SA). Non-commercial academic use | Train, validation and test splits. The repository holds a manifest (observation id, photo id, licence, URL); images are downloaded by script and never committed |
| Pre-training or extra data | SnakeCLEF 2024 training set: 1,784 species, about 17 GB at 500 px or 4.5 GB at 240 px [18] | The page states no licence; to be confirmed before use | Optional. Deduplicate against our test set by observation id |
| Published baseline | Bolon et al. 2022 ViT (ONNX, 1.13 GB) with its released data and code [2] | BSD-3-Clause | Run on our test images (section 6) |
| Snake detector | Open Images V7, where `Snake` is a boxable class (`/m/078jl`) [17], plus Roboflow Universe community sets [19] | Open Images is CC BY 4.0; Roboflow varies by project | Train the single-class detector, with non-snake scenes as negatives |
| Medical importance | WHO SEARO 2016 guidelines, India box on PDF p. 87 [15] | WHO publication | Per-species category used in metrics O2 and O3 |
| Live video evaluation | We found no public species-labelled snake video set; Wikimedia Commons has 0 to 2 videos for each species we checked | None | Build our own clips (section 3.4) |

### 3.3 Splits and leakage control

All photos of one observation stay in one split, and so do all observations from one iNaturalist user, so the same person, camera or individual snake cannot appear on both sides. The test set takes only observations uploaded on or after 1 January 2022. The PLOS and SnakeCLEF 2021 data were collected before then, so the published model has not seen our test images; we will confirm that model's exact data cutoff at Milestone 2. Perceptual hashing removes near-duplicate images across all splits. Each species is split roughly 70/15/15, with a minimum test count per class. A script checks that no observation id, observer id or image hash appears in more than one split, and its output goes in the Milestone 2 report.

No herpetologist reviewer is confirmed, so our labels are iNaturalist research grade: at least two identifiers, with more than two-thirds agreeing at species level. We will check a stratified sample of 200 images ourselves against field-guide characters and report disagreements. If an expert reviewer becomes available, they will start with the hardest lookalike pairs.

### 3.4 Video data

A video is a sequence of images, so the same detector and classifier run on every frame and we do not train a separate video model. We need video data only to evaluate live behaviour: how stable the label is, how long the first label takes, and how often the system raises a false alarm. We will build three clip sets, all from test-split images so nothing leaks from training:

1. Synthetic clips. Each test photo becomes a 10-second, 15 FPS clip with a random camera path (pan, zoom, rotation, hand shake), motion blur, low-light and noise simulation, JPEG compression, and leaf-shaped masks covering 30% to 70% of the snake. The clip keeps the photo's label, and every generation parameter is logged and seeded.
2. Re-captured clips. Team members film test photos on a screen with a moving phone, which adds real sensor noise and the autofocus and exposure changes that synthetic warps miss.
3. No-snake clips built from Open Images scenes without snakes, to measure false alarms.

Neither synthetic nor re-captured clips show a snake actually moving. We will report that limitation and add real clips if we find any with usable licences.

### 3.5 Data risks

Disputed or mis-identified taxa, rare classes, geographic and platform bias, repeated images of the same individual, captive snakes, taxonomic renames (for example *Xenochrophis* to *Fowlea* and *Trimeresurus* to *Craspedocephalus*), photos with several snakes, obscured coordinates (12% of sampled observations) and the mix of licences. The Milestone 2 data report will include a check for each.

## 4. Existing solutions and comparison

| Option | What it offers | Strengths | Limitations relevant to this project |
|---|---|---|---|
| Google Search / Google Lens | A camera or uploaded image starts a visual search that returns visually similar pages; Lens can identify plants and animals | Fast, with a broad index; works from a camera or an image | Search matches are not a calibrated classifier with a fixed snake label set. Results vary with the web, there is no measured abstention or risk score, and it needs internet [6] |
| Gemini Live | Users share a camera feed and ask spoken questions | Natural, interactive live-camera help | General-purpose answers that have not been evaluated on regional snakes. Needs a live connection [7] |
| ChatGPT camera/video | Voice mode supports camera and video sharing in supported mobile apps | Conversational follow-up and broad image understanding | Not trained or calibrated for our class list, and a direct answer can sound confident without any validation [8] |
| Direct response from a general LLM | The user describes the snake or uploads an image | No setup, and it explains its reasoning | No fixed label set, no reproducible score, and no reliable way to tell real uncertainty from a plausible guess. Useful as a qualitative comparison only |
| Global PLOS ViT (Bolon et al. 2022) | A 772-species classifier with optional geography | 92.2% macro-F1 on its benchmark; weights and code released under BSD-3 | The ONNX model is about 1.1 GB, too large for a phone; trained mostly on North American photos; not tested on Indian field photos [2] |
| SnakeCLEF challenge systems | Shared datasets and held-out evaluation, with a 2023 and 2024 focus on medically important scenarios including India | Reproducible research setup; DINOv2 features explored, with released code [12] | Evaluated on static photos on a server GPU and not designed for live, on-device video |
| YOLO snake detectors | Real-time detection: 87% precision [13]; YOLOv8-Nano at 0.88 precision and 142 FPS on 10 species [14] | Show that real-time detection on edge hardware is practical | Detection only, or coarse species; neither reports calibrated abstention |
| Earlier mobile and video snake systems | Offline iOS classification [9], video-based recognition [10], and Serpent-ID with YOLOv8-nano and TensorFlow Lite [11] | Show that offline and video snake recognition already exist | Species coverage, independent validation and safety behaviour vary. We compare against them and do not claim camera input as new |
| SerpentGuard AI (ours, proposed) | Detector, tracker and 27-species classifier on photos and on every live frame; calibrated abstention; offline export | Fixed Indian class list, measured dangerous-species recall, temporal stability, and no LLM or API in the loop | Still a proposal with no measured model. Labels come from community agreement without expert review |

General assistants and earlier snake apps already accept camera input. What we add is one detector and one fine-tuned classifier for a bounded list of Indian species, run on photos and on every live frame with tracking and smoothing, plus a calibrated reject option and an evaluation that weights the most harmful error: a dangerous snake labelled as a not-listed species. We compare the system against the released global model on the same Indian test images.

## 5. Proposed architecture

```text
photo or camera frame ──► YOLOv8n snake detector (1 class, int8 for edge)
                              │ boxes
                              ▼
                      ByteTrack: one ID per snake across frames (video only)
                              │ crop per track
                              ▼
                      MobileViT classifier: 27 species + calibrated softmax
                              │ per-frame probabilities
                              ▼
          temporal smoothing: average over the last ~15 frames of a track
                              │
                              ▼
       decision: top-3 + WHO category, or "uncertain" / "unknown" / "no snake"
                              │
                              ▼
         fixed safety text from an offline knowledge base (quoted, not generated)
```

- Detector: YOLOv8n fine-tuned as a single-class snake detector on Open Images and Roboflow boxes, with ByteTrack tracking as provided by Ultralytics [21, 22]. Photo mode runs the same detector on a single frame.
- Classifier: MobileViT [23], chosen because it is small enough for a phone. The baselines are ResNet-50 and EfficientNet-B0. We will compare full fine-tuning with LoRA [24] in Milestone 4.
- Calibration and abstention: temperature scaling on the validation split [20]. The system shows a label only when the smoothed top-1 probability passes a threshold set on validation data to meet the O4 coverage target.
- Location (optional): candidates are re-ranked by how often each species appears in each state, counted on the training split only.
- Safety knowledge base: species facts and first-aid text quoted word for word from official sources such as WHO SEARO 2016 [15], always shown with the source. The system generates no text, and "no snake detected" is never shown as "safe".
- Deployment: a Gradio app on Hugging Face Spaces with photo upload and webcam streaming, which meets the Milestone 6 requirement. As a stretch goal, both models are exported to TFLite or ONNX int8 and benchmarked on one Android phone (O7).

## 6. Baselines and evaluation plan

### 6.1 Published external baselines

| Reference | Scope / test set | Reported result | How we use it |
|---|---|---|---|
| Durso et al. 2021 [3] | 45 North American species | 84% accuracy and 0.83 F1 on iNaturalist; 72% and 0.73 on the private expert set | Scale reference for O1 |
| Bolon et al. 2022, global ViT [2] | 772 species; 23,673-image test set | 96.0% top-1; 92.2% macro-F1 | We run it on our post-2022 Indian test images, restricted to our 27 classes |
| Naresh et al. 2024 [13] | Snake detection | 87% precision | Detector reference for O5 |
| Najm et al. 2025 [14] | 10 species, YOLOv8-Nano | Precision 0.88, recall 0.91, mAP@0.5 0.892, 142 FPS (Table 1) | Detector and speed reference for O5 and O6 |

### 6.2 Project baselines (to be measured)

We have no measured project result yet. At Milestones 3 and 4 we will measure the following on identical splits:

1. majority class;
2. ResNet-50 and EfficientNet-B0 fine-tuned on full images;
3. MobileViT on full images;
4. detector crop followed by MobileViT;
5. model 4 with location;
6. model 4 with temperature scaling and abstention.

For every run we record the split manifest, class counts, library versions and random seeds.

### 6.3 Metrics

Top-1 and top-3 accuracy, macro-F1, balanced accuracy, per-class precision and recall, the confusion matrix, ECE with a reliability diagram, the risk-coverage curve, recall on medically important species, and the rate at which dangerous snakes are labelled as not-listed species. For the detector we report precision, recall and mAP@0.5. For video we report clip-level accuracy, label flips per clip, time to the first stable label and false alarms per minute. For deployment we report FPS, median and p95 latency, and model size. Headline metrics come with bootstrap 95% confidence intervals.

### 6.4 Evaluation protocol

1. Every model is scored on the same locked, grouped test set (section 3.3), stratified by species, region, image quality and WHO category.
2. Location ablation: image only, image with the correct state, and image with a missing or wrong state.
3. Video: the synthetic, re-captured and no-snake clips from section 3.4, comparing single-frame decisions with temporal smoothing. Adjacent frames never count as independent test samples.
4. General-purpose tools: we send a fixed, stratified 100-image subset to two multimodal LLM APIs (Gemini and GPT) with a fixed prompt that asks for the top 3 from our 27-species list or "unknown". We run Google Lens by hand on 30 of those images. Gemini Live and ChatGPT camera mode give no reproducible ranked output, so we record them only as a qualitative demo. For each tool we log the version, date and prompt, and we estimate the API cost before running.
5. Error analysis on a stratified sample of failures, uncertain predictions and high-risk predictions, checked by the team against field-guide characters and, if possible, by an expert.

## 7. Expected product outputs (illustrative mockups)

These are mock outputs for the proposed interface, not predictions from a trained model.

### Example A: confident, in-scope photo

```text
┌──────────────────────────────────────────────┐
│ Snake detected                                │
│                                              │
│ 1. Candidate species A                 0.82  │
│ 2. Candidate species B                 0.11  │
│ 3. Candidate species C                 0.04  │
│                                              │
│ WHO category (India): shown per candidate     │
│ Visual cues: band pattern; head/body shape    │
│ Keep a safe distance. [Official guidance]     │
│ [View species notes] [Report a correction]    │
└──────────────────────────────────────────────┘
```

The percentages are placeholders. A real release shows verified common and scientific names and sourced distinguishing features.

### Example B: uncertain or unsupported image

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

"See guidance" shows only text quoted from official sources, with the source named, and the model never presents a low-confidence guess as certain.

## 8. Gaps, opportunities, and project response

| Gap in existing work | Opportunity | Planned response |
|---|---|---|
| Long-tailed data and weak coverage of rare species | Class-aware training and evaluation | A fixed 27-class list with at least 200 observations each; other species become "unknown"; per-class metrics |
| Geographic bias (77% of the global set is from the US and Canada) | A bounded Indian use case | Indian-only training and test data, compared with the global model on the same images |
| Similar-looking species with different risk | Explicit error costs | Recall on WHO-category species (O2) and the dangerous-as-not-listed error rate (O3); abstain when ambiguous |
| Benchmark images are cleaner than field photos | Harder test conditions | Results by image quality; synthetic blur, low light and occlusion; re-captured clips |
| High headline accuracy hides overconfidence | Calibrated, communicated uncertainty | Temperature scaling, a reject option and risk-coverage reporting |
| The global model is too large for a phone (about 1.1 GB) | Compact, exportable models | YOLOv8n and MobileViT with int8 export, with size and latency measured (O7) |
| General assistants already take camera input | Compete on measured behaviour | The same models on photos and every live frame, with stability metrics (O6) |
| Live video flickers and can be confidently wrong | Tracking, smoothing and abstention | ByteTrack, frame voting and an "uncertain" state, scored with clip-level metrics |
| Hosted LLMs need a network and can generate unsafe text | No LLM in the loop | Offline models and fixed, quoted safety text only |

## 9. Conclusion and milestone plan

Snake classification has been done at global scale and in mobile and video prototypes, and general assistants already accept camera input. Our contribution is a measured system scoped to India: one detector and one classifier that work on photos and on every live frame, with calibrated abstention and an explicit dangerous-species error metric, compared on the same images against the released global model. The data for the core system is public (section 3.1). The one gap, labelled snake video, we cover by building clips from test images (section 3.4).

| Milestone (due) | Deliverable | Done when |
|---|---|---|
| M2 Dataset (29 Oct) | Download and manifest scripts, EDA notebook, grouped splits, leakage check, WHO labels, synthetic-clip generator | The leakage script reports 0 overlaps and the class-count table matches the manifest |
| M3 Architecture (5 Nov) | The detector, tracker and classifier pipeline run end to end on a data subset and on one synthetic clip | One command produces labelled frames and metrics on the subset |
| M4 Training (19 Nov) | Baselines, MobileViT full fine-tuning against LoRA, augmentation and regularisation sweeps, detector training | A hyperparameter table with at least 6 runs, with checkpoints and configs saved |
| M5 Evaluation (26 Nov) | All metrics in section 6.3 against targets O1 to O8, the PLOS baseline, and error analysis | Every target reported as met or missed, with confidence intervals |
| M6 Deployment (10 Dec) | Hugging Face Space (photo and webcam), offline export benchmark, documentation | A public Space link and a latency table from one phone |

## References

1. Picek L. et al. (2021). [SnakeCLEF 2021 challenge description and dataset](https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge). AIcrowd.
2. Bolon I., Picek L., Durso A. M., Alcoba G., Chappuis F., Ruiz de Castañeda R. (2022). [An artificial intelligence model to identify snakes from across the world: Opportunities and challenges for global health and herpetology](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010647). *PLOS Neglected Tropical Diseases* 16(8): e0010647. Model and data: [LINDAT release](https://lindat.mff.cuni.cz/repository/xmlui/handle/20.500.12800/1-4773) (BSD-3-Clause); code: [picekl/PlosNTD-GlobalSnakeID](https://github.com/picekl/PlosNTD-GlobalSnakeID).
3. Durso A. M. et al. (2021). [Supervised learning computer vision benchmark for snake species identification from photographs](https://pmc.ncbi.nlm.nih.gov/articles/PMC8093445/). *Frontiers in Artificial Intelligence*.
4. Picek L., Hrúz M., Durso A. M. (2024). [Overview of SnakeCLEF 2024: Revisiting Snake Species Identification in Medically Important Scenarios](https://dspace.zcu.cz/items/64e8cf26-55b0-42a1-bc2f-ac6da22c5173). CLEF 2024 Working Notes.
5. iNaturalist, [Computer Vision FAQ](https://www.inaturalist.org/posts/47454-faq) and [Seek app information](https://www.inaturalist.org/pages/seek_app).
6. Google, [Google Lens features](https://blog.google/products-and-platforms/devices/google-lens/google-lens-features/) and [Search by image help](https://support.google.com/websearch/answer/1325808), accessed 22 October 2026.
7. Google, [Gemini Live camera guide](https://blog.google/products-and-platforms/products/gemini/gemini-live-camera-how-to/), accessed 22 October 2026.
8. OpenAI, [ChatGPT Voice FAQ](https://help.openai.com/en/articles/20001274-gpt-live-faq) and [Voice with video](https://m.chatgpt.com/features/voice-with-video/), accessed 22 October 2026.
9. Divya et al. (2021), [Snake Detection and Classification using Deep Learning](https://aisel.aisnet.org/hicss-54/da/mobile_services/3/), HICSS-54. Describes offline on-device and cloud-assisted iOS prototypes.
10. [Smart Snake Identification System using Video Processing](https://rda.sliit.lk/entities/publication/a7482c0f-95a5-441d-8340-ba9750477b44), SLIIT project record describing offline and online video recognition.
11. [Serpent-ID: A Deep Learning-Based Mobile Application for Rapid Snake Identification and Bite Management Guidance](https://ieeexplore.ieee.org/abstract/document/11507553), IEEE Xplore. Implementation details still need checking against the full paper before we use it as a benchmark.
12. [Transfer Learning with Self-Supervised Vision Transformers for Snake Identification](https://arxiv.org/abs/2407.06178) (2024) and the authors' [SnakeCLEF 2024 code](https://github.com/dsgt-arc/snakeclef-2024).
13. Naresh E., Ananda Babu J., Shiva Darshan S. L., Murthy S. V. N. (2024). [A Novel Framework for Detection of Harmful Snakes Using YOLO Algorithm](https://doi.org/10.1007/s42979-023-02366-z). *SN Computer Science*.
14. Najm H., Mohsin Z. R., Abdulhussien W. R. (2025). [Real-Time Detection of Multiple Snake Species in Natural Environments Using YOLOv8-Nano](https://doi.org/10.18280/isi.301025). *Ingénierie des Systèmes d'Information* 30(10): 2807-2814. CC BY 4.0.
15. World Health Organization, Regional Office for South-East Asia (2016). [Guidelines for the Management of Snakebites, 2nd edition](https://cdn.who.int/media/docs/default-source/searo/india/health-topic-pdf/who-guidance-on-management-of-snakebites.pdf). India species categories and the Million Death Study estimate are on PDF p. 87.
16. iNaturalist API v1, [observations and species counts](https://api.inaturalist.org/v1/docs/), queried 22 October 2026 (place 6681 = India; taxon 85553 = Serpentes; research grade; CC0 and CC-BY family photo licences).
17. Google, [Open Images V7 boxable class list](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv) (`/m/078jl` = Snake).
18. ImageCLEF, [SnakeCLEF 2024 dataset and task page](https://www.imageclef.org/node/319).
19. Roboflow Universe, for example [yolo_annotate_snake](https://universe.roboflow.com/thisisnew/yolo_annotate_snake) (community-labelled; licence set per project).
20. Guo C., Pleiss G., Sun Y., Weinberger K. Q. (2017). [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). ICML.
21. Zhang Y. et al. (2022). [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864). ECCV.
22. Ultralytics, [YOLO multi-object tracking](https://docs.ultralytics.com/modes/track/).
23. Mehta S., Rastegari M. (2022). [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178). ICLR.
24. Hu E. J. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

<div style="page-break-before: always"></div>

# Team review and acknowledgment

**This is the final page of the report. Each team member must complete and sign one row before submission.** By signing, each member confirms that they have reviewed the submitted document, understand its contents, and agree with the submission.

| Team member name | Roll Number | Signature / acknowledgment | Date |
|---|---|---|---|
|Jayesh Bansal|24f100185|![Jayesh Bansal](https://drive.google.com/uc?export=download&id=1imepTm_t9vgJVL367tWU-LPYcLq9Zet0)|22/10/2026|
|Vikas|23f3001800|  |  |
|Puneet Bajaj|23f3002416|Puneet Bajaj|22/10/2026|
|Manas Rastogi|  |  |  |
