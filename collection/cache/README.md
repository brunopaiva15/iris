# What survives the dataset

`dataset/` is not versioned: hundreds of thousands of images, tens of
gigabytes. Three files are extracted here because they cost hours to rebuild
and weigh less than a megabyte.

| File | What it is |
|---|---|
| `species.json` | the GBIF taxon of every catalogue name, 1,685 resolved out of 1,688 |
| `species_inat.json` | the iNaturalist taxon, 868 resolved out of 886 requested |
| `stats.json` | the state of one set: 1,513 species, 235,909 images kept |

To start again from here, copy them into `dataset/` before running
`build_dataset.py`: the names already resolved do not go back over the network.
A `null` entry is never memoised as a failure, so an unresolved name will be
retried.
