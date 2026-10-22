# Current scoring profile

The current mogzilla scoring profile has revision `33c59a04cba4`. Additional portrait annotations use a 1–10 scale. Their source is described by the project maintainer as model-scout ratings; individual rater identities and agreement statistics are not recorded in the supplied annotation export.

## Model and preprocessing

The image encoder starts from the SCUT-FBP5500 beauty-regression checkpoint described in [the upstream model card](https://huggingface.co/evanlyhf/scut-fbp5500-beauty), revision `44c77683f3640a259aacff59a2d635875ec65729`. The implementation loads its parameters into a ResNet-50. The current profile updates the last `layer4.2` bottleneck and a regularized regression head; preceding blocks and BatchNorm parameters/statistics remain fixed. This is transfer learning, not a model trained entirely from random initialization.

The geometry pipeline uses the Google Face Landmarker bundle with 478 landmarks. It runs separately from image scoring. Face and wider-head crops, horizontal reflections and normalized visual features are used for the current 1–10 output. Geometric measurements do not form a weighted explanation of that neural score.

The upstream checkpoint URLs and SHA-256 checks are pinned in `scripts/download_holistic_models.py`. The public repository includes the trained scoring head and last-block weights in `training/checkpoints/holistic/`; the upstream encoder and landmark bundle are downloaded by setup.

## Data and evaluation

The additional rated collection contains photographs from MEBeauty and supplied portrait collections, including Pinterest-sourced digitals and other portraits. These labels are separate from the original average MEBeauty scores.

The latest technically usable set contains 917 photographs: 717 for fitting and 200 held out for evaluation. Four epochs were selected on an internal development split before fitting the final profile on the 717 photographs. On the held-out set, MAE is 1.2011 points, RMSE is 1.5075, 51% of predictions are within one point, and Spearman correlation is 0.6926.

Twenty-seven held-out photographs came from the latest added collection. MAE on these is 1.2958; the preceding profile had MAE 2.3483 on the same photographs. Results on the other 173 photographs had already been inspected in earlier experiments, so the combined set is a retrospective audit rather than a new blind test. The adapted visual block improved the latest small subset relative to a new fixed-feature head but worsened overall MAE relative to that control.

Potential repeat views were grouped by visual review; full identity independence and non-overlap with upstream training data are not established. The small selected collection does not establish agreement across the wider public or a representative panel of scouts. Synthetic crop/reflection checks do not measure consistency across independently captured photographs of the same person.

WFLW was used for the original landmark experiments and a technical geometry audit; AFW/300-W and WIDER FACE belong to earlier landmark/detection experiments. They are not additional sources of attractiveness labels for the current visual score. See [the geometry audit](HOLISTIC.md) and [the original dataset notes](DATASETS.md).

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| Upstream image encoder | `54f55b2d03fe68b9bdb992d4992ea7d245b4dd25e0d376dca8f599463ff4a312` |
| Landmark bundle | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| Scoring head | `ec9c55e3bdd87f930ba77aafd5db95e99d163543d7cb7d6d351dddf9f65ab48f` |
| Updated visual block | `36afe3303698f0e21de341934046268deac46009815611991cb2924c34e9dda8` |

Upstream data and weights retain their research-use conditions. Third-party photographs retain the rights of their respective owners. Public repository visibility does not grant new rights to those assets.
