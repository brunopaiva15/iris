"""Building the image set for plant recognition.

The package is independent of the application: it reads the plant list
(`plants.csv`, exported from the app's catalogue), fetches images whose licence
allows commercial use, checks them, deduplicates them, splits them into
train / validation / test, and keeps a manifest where every image holds on to
its source, its author and its licence.
"""

__version__ = '0.1.0'
