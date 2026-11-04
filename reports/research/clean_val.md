# Exploratory: the compute-matched comparison on validation documents without overlap

Not part of the analysis plan; the pre-registered decision rule uses the full validation set (see results.md). Each Turkish validation document gets the share of its 13-grams that occur in the repeated training windows of either seed; subsets drop documents above a threshold. The 20% threshold was chosen after seeing the distribution below, so 50% is shown as well.

The subsets are clean for every model listed: the loader reads Turkish windows in one seed-fixed order, so tr030 and tr030_seed1 saw exactly the x3 pools of their seeds and tr_only_control saw the first 175,616 of the seed-0 pool (English windows of the tr030 runs are not scanned).

## Overlap per document

| Share of the document's 13-grams in the training windows | Documents | Validation tokens | Mean document length |
|---|---|---|---|
| none | 2,526 | 49.3% | 434 |
| 0% < share <= 1% | 297 | 18.4% | 1,375 |
| 1% < share <= 5% | 349 | 13.5% | 861 |
| 5% < share <= 20% | 271 | 11.4% | 934 |
| 20% < share <= 50% | 178 | 5.4% | 673 |
| 50% < share <= 100% | 88 | 2.0% | 499 |

Dropping every document with any overlap removes long documents disproportionately (a single shared phrase is enough), so that subset also shifts the validation distribution towards short documents.

## Turkish BPB per subset

| Run | full validation | without documents >50% overlapping | without documents >20% overlapping | without any overlapping document |
|---|---|---|---|---|
| documents kept | 3,709 | 3,621 | 3,443 | 2,526 |
| tokens kept | 100% | 98% | 93% | 49% |
| tr030 | 1.2158 | 1.2167 | 1.2180 | 1.2195 |
| tr030_seed1 | 1.2170 | 1.2179 | 1.2191 | 1.2209 |
| tr_only_control_x3 | 1.1293 | 1.1311 | 1.1336 | 1.1403 |
| tr_only_control_x3_seed1 | 1.1302 | 1.1321 | 1.1346 | 1.1410 |
| tr_only_control | 1.2780 | 1.2789 | 1.2798 | 1.2793 |
| **x3 control minus tr030 (seed means)** | **-0.0866** | **-0.0857** | **-0.0845** | **-0.0795** |

All values from one float16 evaluation pass per model; the full-validation column can differ from the training logs in the last digit because of evaluation order and precision.
