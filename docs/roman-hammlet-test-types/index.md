# Roman/Hammlet test-case galleries

Type-wise galleries for the two 100-event Roman static-lens test batches. Each gallery has the same per-event figure used in the batch result reports, together with the GULLS truth parameters used for the numerical grouping.

## Scope

- The gallery contains all 200 events: 100 events from the `9910000`–`9910099` bound-planet batch and 100 events from the `9920000`–`9920099` broad binary-signal batch.
- Every event appears exactly once, in a bin defined by its mass ratio `q` and separation `s`.
- The labels are parameter-space tags, not recovery or failure labels. The manifests do not contain the later `lcbinint`/`\vbm{}` inference outcome, so no event is called a success or failure here.
- The `s` coordinate is retained in the GULLS truth convention: for the high-`q` binary sample it is expressed in the host-only Einstein scale. No page claims a geometric caustic crossing; the figures are provided for inspection.

The original batch-level pages are [`991` bound-planet results](../roman-hammlet-batch100-results.md) and [`992` binary-signal results](../roman-binary-hammlet-signal-batch100-results.md).

## Grouping rules

| Quantity | Classes |
|---|---|
| `q` | `planetary`: q < 1e-2; `intermediate`: 1e-2 <= q < 1e-1; `stellar`: q >= 1e-1 |
| `s` | `very_close`: s < 0.45; `close`: 0.45 <= s < 0.70; `resonant_range`: 0.70 <= s < 1.35; `wide`: 1.35 <= s < 2.20; `very_wide`: s >= 2.20 |

## Galleries

| `q` class | `s` regime | Events | 991 | 992 | Page |
|---|---:|---:|---:|---:|---|
| planetary | close | 18 | 12 | 6 | [open gallery](planetary-close.md) |
| planetary | resonant_range | 64 | 52 | 12 | [open gallery](planetary-resonant-range.md) |
| planetary | wide | 24 | 16 | 8 | [open gallery](planetary-wide.md) |
| intermediate | very_close | 8 | 0 | 8 | [open gallery](intermediate-very-close.md) |
| intermediate | close | 9 | 4 | 5 | [open gallery](intermediate-close.md) |
| intermediate | resonant_range | 16 | 12 | 4 | [open gallery](intermediate-resonant-range.md) |
| intermediate | wide | 8 | 4 | 4 | [open gallery](intermediate-wide.md) |
| intermediate | very_wide | 8 | 0 | 8 | [open gallery](intermediate-very-wide.md) |
| stellar | very_close | 12 | 0 | 12 | [open gallery](stellar-very-close.md) |
| stellar | close | 9 | 0 | 9 | [open gallery](stellar-close.md) |
| stellar | resonant_range | 4 | 0 | 4 | [open gallery](stellar-resonant-range.md) |
| stellar | wide | 8 | 0 | 8 | [open gallery](stellar-wide.md) |
| stellar | very_wide | 12 | 0 | 12 | [open gallery](stellar-very-wide.md) |

## Reading the pages

The 991 and 992 figures are diagnostic snapshots from the corresponding batch reports. They are included here to make the parameter-space coverage auditable; the pages do not replace the inference results or assign an algorithmic status to an event.

